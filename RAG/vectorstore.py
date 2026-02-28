import faiss
import numpy as np
import pickle
import os
from typing import List
from langchain.schema import Document

STORE_DIR   = "vector_store"
INDEX_PATH  = f"{STORE_DIR}/index.faiss"
CHUNKS_PATH = f"{STORE_DIR}/chunks.pkl"

def saveVectorStore(chunks: List[Document], embeddings: np.ndarray):
    os.makedirs(STORE_DIR, exist_ok=True)
    embeddings = np.array(embeddings).astype("float32")

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            existing_chunks = pickle.load(f)
        existing_hashes = {c.metadata.get("hash_id") for c in existing_chunks}
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        existing_chunks = []
        existing_hashes = set()

    filtered = [(c, e) for c, e in zip(chunks, embeddings)
                if c.metadata.get("hash_id") not in existing_hashes]

    if not filtered:
        print("✅ No new chunks — everything already stored!")
        return

    new_chunks, new_embeddings = zip(*filtered)
    index.add(np.array(new_embeddings).astype("float32"))
    all_chunks = existing_chunks + list(new_chunks)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Added: {len(new_chunks)} | Total: {index.ntotal} vectors")