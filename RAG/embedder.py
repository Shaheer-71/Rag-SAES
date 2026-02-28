from sentence_transformers import SentenceTransformer
from typing import List
from langchain.schema import Document
from config import CONFIG

model = SentenceTransformer(CONFIG["model"])
seen_hashes = set()

def EmbeddingChunkedData(chunks: List[Document]):
    
    if not chunks:
        print("[WARNING] No chunks to embed!")
        return [], []

    unique_chunks = []
    skipped = 0
    for chunk in chunks:
        hash_id = chunk.metadata.get("hash_id")
        if hash_id in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(hash_id)
        unique_chunks.append(chunk)

    print(f"✅ Unique : {len(unique_chunks)} | 🔁 Skipped : {skipped}")

    if not unique_chunks:
        print("[WARNING] All chunks were duplicates!")
        return [], []

    embeddings = model.encode(
        [chunk.page_content for chunk in unique_chunks],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"✅ Embedded {len(embeddings)} chunks")
    print(f"📐 Shape: {len(embeddings)} x {len(embeddings[0])}")
    return unique_chunks, embeddings