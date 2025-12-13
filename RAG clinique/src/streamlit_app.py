# streamlit_app.py
"""
Interface Streamlit pour le chatbot médical RAG.
Remplace flask_api.py + medical_chatbot.html
"""
import streamlit as st
import time
from pathlib import Path
from datetime import datetime

# Import du pipeline RAG existant
from rag_pipeline import (
    load_vectorstore,
    rag_query,
    VECTOR_DIR,
    retrieve_from_vectorstore,
    rerank_hits
)

# =========================================
# CONFIGURATION DE LA PAGE
# =========================================
st.set_page_config(
    page_title="Medical Assistant AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# STYLES PERSONNALISÉS
# =========================================
st.markdown("""
<style>
    /* Thème principal */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Zone de chat */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Messages utilisateur */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: right;
    }
    
    /* Messages bot */
    .bot-message {
        background: #f8fafc;
        color: #1e293b;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Tags de source */
    .source-tag {
        display: inline-block;
        background: #e2e8f0;
        color: #475569;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 12px;
        margin: 5px 5px 5px 0;
    }
    
    /* Header */
    .main-header {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# INITIALISATION DE LA SESSION
# =========================================
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.vectorstore_loaded = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0

# =========================================
# FONCTION DE CHARGEMENT DU VECTORSTORE
# =========================================
@st.cache_resource
def initialize_vectorstore():
    """Charge le vectorstore (une seule fois grâce au cache)."""
    try:
        if not VECTOR_DIR.exists():
            st.error(f"❌ Vectorstore introuvable: {VECTOR_DIR}")
            st.info("👉 Lancez d'abord: `python rebuild_vectorstore.py`")
            return None
        
        vectorstore = load_vectorstore()
        return vectorstore
    except Exception as e:
        st.error(f"❌ Erreur chargement vectorstore: {e}")
        return None

# =========================================
# SIDEBAR (PANNEAU LATÉRAL)
# =========================================
with st.sidebar:
    st.image("https://em-content.zobj.net/thumbs/120/apple/354/hospital_1f3e5.png", width=100)
    st.title("🏥 Medical Assistant")
    st.markdown("---")
    
    # Statut du système
    st.subheader("📊 System Status")
    
    if st.session_state.vectorstore_loaded:
        st.success("✅ Vectorstore: Loaded")
    else:
        st.warning("⚠️ Vectorstore: Not loaded")
    
    st.info(f"💬 Queries: {st.session_state.total_queries}")
    
    st.markdown("---")
    
    # Paramètres
    st.subheader("⚙️ Settings")
    
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=3,
        max_value=20,
        value=10,
        help="More chunks = more context but slower"
    )
    
    use_rerank = st.checkbox(
        "Enable re-ranking",
        value=True,
        help="CrossEncoder for better accuracy"
    )
    
    st.markdown("---")
    
    # Boutons d'action
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()
    
    if st.button("📥 Export Conversation"):
        if st.session_state.messages:
            export_text = "\n\n".join([
                f"[{msg['timestamp']}]\n{msg['role'].upper()}: {msg['content']}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                label="💾 Download TXT",
                data=export_text,
                file_name=f"medical_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.warning("No conversation to export")
    
    st.markdown("---")
    st.caption("Powered by RAG Pipeline")
    st.caption("📚 Gale Encyclopedia of Medicine")

# =========================================
# MAIN CONTENT
# =========================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Medical Assistant AI</h1>
    <p>Powered by RAG Pipeline - Ask medical questions in English</p>
</div>
""", unsafe_allow_html=True)

# Chargement du vectorstore
if not st.session_state.vectorstore_loaded:
    with st.spinner("🔮 Loading vectorstore..."):
        vectorstore = initialize_vectorstore()
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.vectorstore_loaded = True
            st.success("✅ Vectorstore loaded successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.stop()

# =========================================
# QUICK QUESTIONS
# =========================================
if len(st.session_state.messages) == 0:
    st.markdown("### 🚀 Quick Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💉 What are the symptoms of diabetes?"):
            st.session_state.quick_question = "What are the symptoms of diabetes?"
            st.rerun()
        
        if st.button("🫁 What is asthma?"):
            st.session_state.quick_question = "What is asthma?"
            st.rerun()
    
    with col2:
        if st.button("❤️ How to treat hypertension?"):
            st.session_state.quick_question = "How to treat hypertension?"
            st.rerun()
        
        if st.button("💔 What causes heart disease?"):
            st.session_state.quick_question = "What causes heart disease?"
            st.rerun()

# =========================================
# AFFICHAGE DES MESSAGES
# =========================================
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                👤 <strong>You</strong><br>
                {message['content']}<br>
                <small style="opacity: 0.8;">{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if message.get('sources'):
                sources_html = "<br>".join([
                    f'<span class="source-tag">📄 Page {s["page"]} (Score: {s["score"]:.3f})</span>'
                    for s in message['sources']
                ])
            
            st.markdown(f"""
            <div class="bot-message">
                🤖 <strong>Medical Assistant</strong><br>
                {message['content']}<br>
                {sources_html}<br>
                <small style="color: #94a3b8;">{message['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)

# =========================================
# INPUT UTILISATEUR
# =========================================
def process_query(user_input):
    """Traite la question de l'utilisateur."""
    if not user_input.strip():
        return
    
    # Ajoute le message utilisateur
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Génère la réponse
    with st.spinner("🤖 Thinking..."):
        try:
            # Appel au pipeline RAG
            answer = rag_query(
                question=user_input,
                vectordb=st.session_state.vectorstore,
                rerank=use_rerank,
                top_k=top_k,
                debug=False
            )
            
            # Récupère les sources
            hits = retrieve_from_vectorstore(user_input, st.session_state.vectorstore, top_k=top_k)
            if use_rerank:
                hits = rerank_hits(user_input, hits, top_n=3)
            
            sources = [
                {
                    'page': h.get('page'),
                    'chunk_id': h.get('chunk_id'),
                    'score': h.get('rerank_score', h.get('score', 0))
                }
                for h in hits[:3]
            ]
            
            # Ajoute la réponse
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            st.session_state.total_queries += 1
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sorry, I encountered an error processing your question.",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

# Gestion des quick questions
if 'quick_question' in st.session_state:
    process_query(st.session_state.quick_question)
    del st.session_state.quick_question
    st.rerun()

# Chat input
user_input = st.chat_input("Ask a medical question in English...")

if user_input:
    process_query(user_input)
    st.rerun()

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This is an educational tool. Always consult a healthcare professional for medical advice.")