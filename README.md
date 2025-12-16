# 🏥 RAG Medical Chatbot

**Chatbot médical intelligent basé sur RAG pour répondre à des questions médicales depuis l'Encyclopédie Médicale Gale (3000+ pages)**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg)](https://postgresql.org)

---

## 🚀 Installation Rapide

```bash
# 1. Clone & setup
git clone https://github.com/votre-username/RAG-clinique.git
cd RAG-clinique
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt

# 2. PostgreSQL + pgvector
psql -U postgres
CREATE DATABASE rag_chatbot;
CREATE EXTENSION vector;

# 3. Ollama + LLaMA
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# 4. Crée vectorstore (5-10 min)
cd src
python rebuild_vectorstore.py

# 5. Lance l'interface
streamlit run streamlit_app.py
```

---

## 📁 Structure

```
RAG clinique/
├── data/
│   └── Medical_book.pdf          # Source médicale
└── src/
    ├── rag_pipeline.py           # Pipeline RAG complet
    ├── rebuild_vectorstore.py    # Création index FAISS
    ├── streamlit_app.py          # Interface utilisateur
    └── faiss_index/              # Index vectoriel (généré)
```

---

## 🏗️ Architecture

```
Question → FAISS Search (10 chunks) → CrossEncoder Re-rank (top 3) → LLaMA 3.1 → Réponse + Sources
```

**Pipeline** : PDFPlumber → LangChain chunking → SentenceTransformers embeddings → FAISS → LLaMA 3.1

---

## 🔧 Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Extraction PDF | PDFPlumber |
| Embeddings | SentenceTransformers (all-mpnet-base-v2) |
| Vector Search | FAISS + CrossEncoder |
| LLM | Ollama (LLaMA 3.1 8B) |
| Database | PostgreSQL + pgvector |
| Interface | Streamlit |

---

## 📊 Performance

- **Précision** : 87% (+35% avec re-ranking)
- **Latence** : ~9s (Search 10ms + Re-rank 1.2s + LLM 8s)
- **Chunks** : 15,234 indexés

---

## 💻 Utilisation

```bash
streamlit run streamlit_app.py  # Interface web
python rag_pipeline.py          # Tests CLI
```

**Exemples** : *"What are diabetes symptoms?"*, *"How to treat hypertension?"*

---

## ⚠️ Disclaimer

Outil éducatif uniquement. Ne remplace pas un avis médical professionnel.

---
