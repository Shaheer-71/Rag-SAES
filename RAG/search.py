import faiss
import numpy as np
import pickle
import os
from embedder import model
from vectorstore import INDEX_PATH, CHUNKS_PATH

def loadVectorStore():
    if not os.path.exists(INDEX_PATH):
        print("[ERROR] No vector store found!")
        return None, []
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {index.ntotal} vectors")
    return index, chunks

def search(query: str, top_k: int = 3):
    index, chunks = loadVectorStore()
    if index is None:
        return []

    query_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_emb, top_k)

    return [
        {
            "content": chunks[i].page_content,
            "score":   round(float(s), 4),
            "source":  chunks[i].metadata.get("source")
        }
        for s, i in zip(scores[0], indices[0])
        if i != -1
    ]