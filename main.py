import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "RAG"))  # ✅ points to RAG/ folder

from loader      import loadDocs
from chunker     import chunkDocs
from embedder    import EmbeddingChunkedData
from vectorstore import saveVectorStore
from search      import search
from generator   import GenerateAnswer
from config      import CONFIG

def ingest():
    docs                      = loadDocs(CONFIG["data_dir"])
    chunks                    = chunkDocs(docs)
    unique_chunks, embeddings = EmbeddingChunkedData(chunks)
    saveVectorStore(unique_chunks, embeddings)
    print("✅ Ingestion complete!")

def chat():
    print("Bot ready! Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        results = search(question)
        answer  = GenerateAnswer(question, results)
        print(f"\nBot: {answer}\n")

if __name__ == "__main__":
    # ingest()  # ← uncomment and run once to load docs
    chat()      # ← chat with your docs