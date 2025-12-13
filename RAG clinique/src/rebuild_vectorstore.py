# rebuild_vectorstore.py
"""
Script pour reconstruire complètement le vectorstore FAISS
avec les métadonnées correctes.
"""
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paramètres
PDF_PATH = Path("../data/Medical_book.pdf")
VECTOR_DIR = Path("faiss_index")
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_and_split_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Charge et découpe le PDF en chunks avec métadonnées."""
    logger.info(f"📖 Chargement du PDF: {pdf_path}")
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")
    
    pages_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"📄 Nombre total de pages: {len(pdf.pages)}")
        
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            page_num = i + 1
            
            # Garde les pages avec minimum 50 caractères
            if len(txt.strip()) > 50:
                pages_text.append((page_num, txt))
    
    logger.info(f"✅ {len(pages_text)} pages conservées")
    
    # Découpage en chunks
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    logger.info("✂️  Découpage en chunks...")
    
    for p_no, txt in pages_text:
        raw_chunks = splitter.split_text(txt)
        
        for idx, c in enumerate(raw_chunks):
            cleaned = c.strip()
            if len(cleaned) > 20:  # Minimum 20 caractères
                chunks.append({
                    "text": cleaned,
                    "page": p_no,
                    "chunk_id": f"p{p_no}_c{idx}"
                })
    
    logger.info(f"✅ {len(chunks)} chunks créés")
    
    # Affiche statistiques
    pages_with_chunks = set(c["page"] for c in chunks)
    logger.info(f"📊 Pages avec contenu: {len(pages_with_chunks)}")
    
    # Affiche exemples de chunks médical (pas juste titre)
    medical_chunks = [c for c in chunks if len(c["text"]) > 100 and c["page"] > 10]
    if medical_chunks:
        logger.info(f"\n📝 Exemples de chunks médicaux:")
        for i, chunk in enumerate(medical_chunks[:3], 1):
            logger.info(f"\n{i}. Page {chunk['page']} ({chunk['chunk_id']}):")
            logger.info(f"   {chunk['text'][:200]}...")
    
    return chunks

def create_vectorstore(chunks: List[Dict[str, Any]], vector_dir: Path):
    """Crée un nouveau vectorstore FAISS."""
    if not chunks:
        raise ValueError("Aucun chunk à indexer!")
    
    logger.info(f"🔮 Création du vectorstore FAISS...")
    logger.info(f"   - Modèle: {EMBED_MODEL_NAME}")
    logger.info(f"   - Nombre de chunks: {len(chunks)}")
    
    # Prépare les textes et métadonnées
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "page": c["page"],
            "chunk_id": c["chunk_id"],
            "text_length": len(c["text"])
        } 
        for c in chunks
    ]
    
    # Vérifie que les métadonnées sont bien présentes
    logger.info(f"📋 Exemple de métadonnées:")
    for i in range(min(3, len(metadatas))):
        logger.info(f"   {i+1}. {metadatas[i]}")
    
    # Crée l'embedding function
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    
    # Crée le vectorstore
    logger.info("⏳ Création des embeddings (cela peut prendre quelques minutes)...")
    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embedding_fn,
        metadatas=metadatas
    )
    
    # Sauvegarde
    if vector_dir.exists():
        logger.warning(f"⚠️  Suppression de l'ancien index: {vector_dir}")
        shutil.rmtree(vector_dir)
    
    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))
    
    logger.info(f"✅ Vectorstore sauvegardé: {vector_dir.absolute()}")
    
    return vectordb

def test_vectorstore(vectordb: FAISS):
    """Test le vectorstore avec plusieurs requêtes."""
    logger.info(f"\n{'='*70}")
    logger.info(f"🧪 TEST DU VECTORSTORE")
    logger.info(f"{'='*70}\n")
    
    test_queries = [
        "Quels sont les symptômes du diabète ?",
        "Comment traiter l'hypertension ?",
        "Qu'est-ce que l'asthme ?",
        "Symptômes de la grippe",
        "Traitement du cancer"
    ]
    
    for query in test_queries:
        logger.info(f"\n🔍 Requête: '{query}'")
        
        results = vectordb.similarity_search_with_score(query, k=10)
        logger.info(f"   Résultats trouvés: {len(results)}")
        
        if results:
            logger.info(f"   Top 3:")
            for i, (doc, score) in enumerate(results[:3], 1):
                metadata = doc.metadata
                page = metadata.get("page", "N/A")
                chunk_id = metadata.get("chunk_id", "N/A")
                text_preview = doc.page_content[:100].replace('\n', ' ')
                
                logger.info(f"   {i}. Score: {score:.4f} | Page: {page} | ID: {chunk_id}")
                logger.info(f"      Texte: {text_preview}...")
        else:
            logger.warning(f"   ⚠️  Aucun résultat trouvé!")
        
        logger.info("")

def main():
    logger.info(f"\n{'#'*70}")
    logger.info(f"# 🔧 RECONSTRUCTION DU VECTORSTORE")
    logger.info(f"{'#'*70}\n")
    
    try:
        # 1. Charge le PDF
        chunks = load_and_split_pdf(PDF_PATH)
        
        if not chunks:
            logger.error("❌ Aucun chunk créé. Vérifiez le PDF.")
            return
        
        # 2. Crée le vectorstore
        vectordb = create_vectorstore(chunks, VECTOR_DIR)
        
        # 3. Test
        test_vectorstore(vectordb)
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"# ✅ RECONSTRUCTION TERMINÉE")
        logger.info(f"{'#'*70}\n")
        logger.info(f"👉 Le vectorstore est prêt dans: {VECTOR_DIR.absolute()}")
        logger.info(f"👉 Vous pouvez maintenant relancer rag_pipeline.py")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)

if __name__ == "__main__":
    main()