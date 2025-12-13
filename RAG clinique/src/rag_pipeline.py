# rag_pipeline.py
import os
import re
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import logging

import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

# =========================================
# CONFIGURATION DU LOGGING
# =========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# =========================================
load_dotenv()

# =========================================
# PARAMÈTRES DE CONFIGURATION
# =========================================
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = Path("../data/Medical_book.pdf")
VECTOR_DIR = BASE_DIR / "faiss_index"

POSTGRES_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "rag_db"),
    "user": os.getenv("POSTGRES_USER", "rag_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "changeme"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
}

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Ollama / LLM config
OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE}/api/generate"
LLM_MODEL = "llama3.1:8b"

# Retrieval / RAG
TOP_K = 10
MAX_PROMPT_CHARS = 20000
CHUNKS_TO_KEEP_FOR_PROMPT = 6

# =========================================
# UTILITAIRES
# =========================================
def looks_like_token_dump(s: str) -> bool:
    """Détecte les dumps de tokens numériques."""
    if re.search(r"^\s*[\d,\s]{50,}\s*$", s):
        return True
    tokens = re.findall(r"\b\d+\b", s)
    if len(tokens) > 200:
        return True
    if '"tokens"' in s or '"ids"' in s or '"prompt_eval_count"' in s:
        return True
    return False

# =========================================
# CHARGEMENT ET SPLIT DU PDF (AMÉLIORÉ)
# =========================================
def load_and_split_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Charge le PDF avec logging détaillé et filtrage intelligent.
    """
    logger.info(f"🔍 Chargement du PDF: {pdf_path}")
    logger.info(f"📁 Chemin absolu: {pdf_path.absolute()}")
    logger.info(f"✅ Fichier existe: {pdf_path.exists()}")
    
    if not pdf_path.exists():
        logger.error(f"❌ ERREUR: Le fichier PDF n'existe pas!")
        return []
    
    # Keywords à ignorer - DÉSACTIVÉ PAR DÉFAUT pour debug
    ignore_keywords = []  # Vide = accepte tout
    
    pages_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"📄 Nombre total de pages: {total_pages}")
            
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                page_num = i + 1
                
                # Log des premières pages pour diagnostic
                if i < 3:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"PAGE {page_num} - Longueur: {len(txt)} caractères")
                    logger.info(f"Aperçu (200 premiers chars):")
                    logger.info(txt[:200].replace('\n', ' '))
                    logger.info(f"{'='*60}\n")
                
                # Filtrage (désactivé si ignore_keywords est vide)
                if ignore_keywords:
                    has_ignore = any(k.lower() in txt.lower() for k in ignore_keywords)
                    if has_ignore:
                        logger.debug(f"⏭️  Page {page_num} ignorée (contient keyword)")
                        continue
                
                # Garde toutes les pages avec contenu
                if len(txt.strip()) > 50:  # Au moins 50 caractères
                    pages_text.append((page_num, txt))
                    logger.debug(f"✅ Page {page_num} conservée ({len(txt)} chars)")
                else:
                    logger.debug(f"⚠️  Page {page_num} ignorée (trop courte)")
                    
    except Exception as e:
        logger.error(f"❌ ERREUR lors de la lecture du PDF: {e}", exc_info=True)
        return []
    
    logger.info(f"\n📊 RÉSULTAT: {len(pages_text)} pages conservées sur {total_pages}")
    
    # =========================================
    # SPLIT EN CHUNKS
    # =========================================
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    logger.info(f"✂️  Découpage en chunks (size=1000, overlap=200)...")
    
    for p_no, txt in pages_text:
        raw_chunks = splitter.split_text(txt)
        logger.debug(f"Page {p_no}: {len(raw_chunks)} chunks créés")
        
        for idx, c in enumerate(raw_chunks):
            cleaned = c.strip()
            if len(cleaned) > 20:  # Minimum 20 caractères
                chunks.append({
                    "text": cleaned,
                    "page": p_no,
                    "chunk_id": f"p{p_no}_c{idx}"
                })
    
    logger.info(f"✅ TOTAL: {len(chunks)} chunks créés")
    
    # Affiche quelques exemples de chunks
    if chunks:
        logger.info(f"\n📝 Exemples de chunks:")
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"\nChunk {i+1} (page {chunk['page']}):")
            logger.info(chunk['text'][:150] + "...")
    else:
        logger.warning(f"⚠️  ATTENTION: Aucun chunk créé!")
    
    return chunks

# =========================================
# EMBEDDINGS
# =========================================
def create_embeddings(chunks_texts: List[str]) -> List[List[float]]:
    logger.info(f"🧠 Création des embeddings pour {len(chunks_texts)} chunks...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(chunks_texts, show_progress_bar=True)
    logger.info(f"✅ Embeddings créés: {len(embeddings)} vecteurs")
    return embeddings

# =========================================
# POSTGRESQL
# =========================================
def ensure_postgres_tables():
    logger.info("🗄️  Vérification des tables PostgreSQL...")
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents_texts (
            id SERIAL PRIMARY KEY,
            title TEXT,
            page_number INT,
            chunk_id TEXT,
            text_chunk TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id SERIAL PRIMARY KEY,
            query TEXT,
            result TEXT,
            retrieved_meta JSONB,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        logger.info("✅ Tables PostgreSQL prêtes")
    except Exception as e:
        logger.error(f"❌ Erreur PostgreSQL: {e}", exc_info=True)

def insert_chunks_metadata_to_postgres(chunks: List[Dict[str, Any]]):
    if not chunks:
        logger.warning("⚠️  Pas de chunks à insérer dans PostgreSQL")
        return
    
    logger.info(f"💾 Insertion de {len(chunks)} chunks dans PostgreSQL...")
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()
        sql = """
        INSERT INTO documents_texts (title, page_number, chunk_id, text_chunk)
        VALUES %s
        """
        values = [("Medical Book", c["page"], c["chunk_id"], c["text"]) for c in chunks]
        execute_values(cur, sql, values)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("✅ Chunks insérés dans PostgreSQL")
    except Exception as e:
        logger.error(f"❌ Erreur insertion PostgreSQL: {e}", exc_info=True)

def log_query_to_postgres(query: str, result_text: str, retrieved_meta: List[Dict[str, Any]]):
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()
        sql = "INSERT INTO query_logs (query, result, retrieved_meta) VALUES (%s, %s, %s)"
        cur.execute(sql, (query, result_text, json.dumps(retrieved_meta, ensure_ascii=False)))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Erreur log query: {e}")

# =========================================
# FAISS VECTORSTORE
# =========================================
def create_vectorstore(chunks: List[Dict[str, Any]]):
    if not chunks:
        logger.error("❌ Impossible de créer vectorstore: aucun chunk!")
        return None
    
    logger.info(f"🔮 Création du vectorstore FAISS...")
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks]
    
    vectordb = FAISS.from_texts(texts=texts, embedding=embedding_fn, metadatas=metadatas)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(VECTOR_DIR))
    logger.info(f"✅ Vectorstore sauvegardé dans: {VECTOR_DIR}")
    return vectordb

def load_vectorstore():
    logger.info(f"📂 Chargement du vectorstore depuis: {VECTOR_DIR}")
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    try:
        # Essaie avec le nouveau paramètre (LangChain >= 0.1.0)
        vectordb = FAISS.load_local(str(VECTOR_DIR), embedding_fn, allow_dangerous_deserialization=True)
    except TypeError:
        # Fallback pour les anciennes versions de LangChain
        vectordb = FAISS.load_local(str(VECTOR_DIR), embedding_fn)
    logger.info("✅ Vectorstore chargé")
    return vectordb

def get_or_create_vectorstore(chunks: List[Dict[str, Any]]):
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        try:
            return load_vectorstore()
        except Exception as e:
            logger.warning(f"⚠️  Échec chargement vectorstore, recréation: {e}")
            return create_vectorstore(chunks)
    else:
        return create_vectorstore(chunks)

def retrieve_from_vectorstore(query: str, vectordb: FAISS, top_k: int = TOP_K):
    logger.info(f"🔍 Recherche dans vectorstore: '{query[:50]}...'")
    results = vectordb.similarity_search_with_score(query, k=top_k)
    hits = []
    for doc, score in results:
        hits.append({
            "text": doc.page_content,
            "score": float(score),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id")
        })
    logger.info(f"✅ {len(hits)} résultats trouvés")
    return hits

def rerank_hits(query: str, hits: List[Dict[str, Any]], top_n: int = None):
    try:
        logger.info(f"🔄 Re-ranking des résultats...")
        cross = CrossEncoder(CROSS_ENCODER_NAME)
        pairs = [[query, h["text"]] for h in hits]
        scores = cross.predict(pairs)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        hits_sorted = sorted(hits, key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        logger.info(f"✅ Re-ranking terminé")
        return hits_sorted[: top_n or len(hits_sorted)]
    except Exception as e:
        logger.warning(f"⚠️  Re-ranking échoué: {e}")
        return hits

# =========================================
# PROMPT TEMPLATE (EN ANGLAIS)
# =========================================
PROMPT_TEMPLATE = """You are a medical assistant strictly limited to the content provided in the CONTEXT below.
You are NOT allowed to use your external knowledge.
You are NOT allowed to invent, extrapolate or deduce information that is not explicitly present in the CONTEXT.

IMPORTANT RULES:
1. Only answer if the answer is clearly found in the CONTEXT.
2. Cite passages from the CONTEXT used (reformulating if necessary) indicating the page: [page X].
3. If the CONTEXT does not contain the requested information, respond exactly:
   "I cannot find this information in the book."
4. Do not make any medical assumptions.
5. Do not give any medical advice outside the provided text.

CONTEXT:
{retrieved_text}

QUESTION:
{user_question}

ANSWER (based only on the book):
"""

def build_prompt(question: str, hits: List[Dict[str, Any]]):
    if not hits:
        logger.warning("⚠️  Pas de hits pour construire le prompt")
        return None
    
    hits_trim = hits[:CHUNKS_TO_KEEP_FOR_PROMPT]
    context_parts = []
    
    for i, h in enumerate(hits_trim, start=1):
        p = h.get("page", "N/A")
        txt = h["text"]
        max_chunk = 2000
        if len(txt) > max_chunk:
            txt = txt[:max_chunk] + " ... [TRUNCATED]"
        context_parts.append(f"[{i}] (page {p}) {txt}")
    
    retrieved_text = "\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(retrieved_text=retrieved_text, user_question=question)
    
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS - 50] + "\n\n...[PROMPT TRUNCATED]..."
    
    logger.info(f"📝 Prompt construit ({len(prompt)} caractères)")
    return prompt

# =========================================
# OLLAMA CALLER (ROBUSTE) - FIX ERROR 500
# =========================================
def query_llm(prompt_text: str, max_tokens: int = 512, debug: bool = False) -> str:
    """Appel robuste à Ollama avec fallback HTTP -> CLI."""
    if not prompt_text:
        return "I cannot find this information in the book."
    
    logger.info(f"🤖 Appel LLM (modèle: {LLM_MODEL})...")
    
    # FIX pour l'erreur 500: utiliser options au lieu de max_tokens
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
        }
    }
    
    # ===== HTTP ATTEMPT =====
    try:
        logger.debug(f"🌐 Tentative HTTP: {OLLAMA_API_GENERATE}")
        resp = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=120)
        
        if resp.status_code == 200:
            data = resp.json()
            
            # Ollama retourne {"response": "text"}
            if "response" in data:
                txt = data["response"].strip()
                if looks_like_token_dump(txt):
                    raise RuntimeError("Token dump détecté")
                logger.info("✅ Réponse LLM reçue (HTTP)")
                return txt
            
            # Fallback
            logger.warning(f"⚠️  Format inattendu: {list(data.keys())}")
            return str(data)
        else:
            logger.warning(f"⚠️  HTTP {resp.status_code}: {resp.text[:200]}")
            raise RuntimeError(f"HTTP {resp.status_code}")
            
    except Exception as e_http:
        logger.warning(f"⚠️  HTTP échoué: {e_http}")
        logger.info("🔄 Fallback vers CLI...")
    
    # ===== CLI FALLBACK =====
    tmp_path = None
    try:
        prompt_for_cli = prompt_text
        if len(prompt_for_cli) > MAX_PROMPT_CHARS:
            prompt_for_cli = prompt_for_cli[:MAX_PROMPT_CHARS - 50] + "\n\n...[TRUNCATED]..."
        
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as tmpf:
            tmpf.write(prompt_for_cli)
            tmpf.flush()
            tmp_path = tmpf.name
        
        cmd_stdin = ["ollama", "run", LLM_MODEL]
        logger.debug(f"🖥️  Commande CLI: {' '.join(cmd_stdin)}")
        
        proc = subprocess.run(
            cmd_stdin,
            input=prompt_for_cli,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        
        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"CLI failed ({proc.returncode}): {err_msg}")
        
        out = proc.stdout.strip()
        if looks_like_token_dump(out):
            raise RuntimeError("Token dump détecté")
        
        logger.info("✅ Réponse LLM reçue (CLI)")
        return out
        
    except Exception as e_cli:
        logger.error(f"❌ Erreur CLI: {e_cli}", exc_info=True)
        return "Error generating response."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

# =========================================
# RAG QUERY
# =========================================
def rag_query(question: str, vectordb: FAISS, rerank: bool = True, top_k: int = TOP_K, debug: bool = False):
    logger.info(f"\n{'='*70}")
    logger.info(f"🔎 REQUÊTE RAG: {question}")
    logger.info(f"{'='*70}")
    
    if not vectordb:
        logger.error("❌ Vectorstore non disponible")
        return "Error: database not available."
    
    hits = retrieve_from_vectorstore(question, vectordb, top_k=top_k)
    
    if rerank and hits:
        hits = rerank_hits(question, hits, top_n=top_k)
    
    if not hits:
        result = "I cannot find this information in the book."
        logger.info(f"📭 Aucun résultat trouvé")
        log_query_to_postgres(question, result, [])
        return result
    
    # Log des meilleurs résultats
    logger.info(f"\n📊 Top 3 résultats:")
    for i, hit in enumerate(hits[:3], 1):
        logger.info(f"{i}. Score: {hit.get('rerank_score', hit.get('score', 0)):.4f} | Page: {hit.get('page')} | Chunk: {hit.get('chunk_id')}")
        logger.info(f"   Texte: {hit['text'][:100]}...")
    
    prompt = build_prompt(question, hits)
    if not prompt:
        result = "I cannot find this information in the book."
        log_query_to_postgres(question, result, hits)
        return result
    
    llm_out = query_llm(prompt_text=prompt, max_tokens=512, debug=debug)
    
    if "cannot find this information" in llm_out.lower():
        final = "I cannot find this information in the book."
    else:
        final = llm_out
    
    logger.info(f"\n✅ RÉPONSE FINALE:\n{final}\n")
    log_query_to_postgres(question, final, hits)
    return final

# =========================================
# MAIN PIPELINE
# =========================================
def main():
    logger.info(f"\n{'#'*70}")
    logger.info(f"# 🏥 PIPELINE RAG MÉDICAL - DÉMARRAGE")
    logger.info(f"{'#'*70}\n")
    
    # 1. Chargement PDF
    chunks = load_and_split_pdf(PDF_PATH)
    
    if not chunks:
        logger.error(f"\n❌ ERREUR CRITIQUE: Aucun chunk créé!")
        logger.error(f"Vérifiez:")
        logger.error(f"  1. Le chemin du PDF: {PDF_PATH.absolute()}")
        logger.error(f"  2. Le contenu du PDF (lisible avec pdfplumber?)")
        logger.error(f"  3. Les logs ci-dessus pour plus de détails")
        return
    
    # 2. PostgreSQL
    ensure_postgres_tables()
    
    # 3. Vectorstore
    vectordb = get_or_create_vectorstore(chunks)
    
    if not vectordb:
        logger.error("❌ Impossible de créer/charger le vectorstore")
        return
    
    # 4. Insertion metadata
    insert_chunks_metadata_to_postgres(chunks)
    
    # 5. Test RAG (EN ANGLAIS - le livre est en anglais!)
    logger.info(f"\n{'='*70}")
    logger.info(f"🧪 TEST DU SYSTÈME RAG")
    logger.info(f"{'='*70}\n")
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "How to treat hypertension?",
        "What is asthma and its symptoms?",
        "What are the causes of heart disease?",
        "How to diagnose pneumonia?"
    ]
    
    for query in test_queries:
        answer = rag_query(query, vectordb, rerank=True, top_k=TOP_K, debug=False)
        logger.info(f"\n{'─'*70}\n")
        time.sleep(1)  # Pause entre les requêtes
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# ✅ PIPELINE TERMINÉ - Consultez rag_pipeline.log pour détails")
    logger.info(f"{'#'*70}\n")
    logger.info(f"💡 NOTE: Le livre est en anglais, posez vos questions en anglais!")

if __name__ == "__main__":
    main()