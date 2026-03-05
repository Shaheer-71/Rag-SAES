import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "RAG"))

from loader      import loadDocs
from chunker     import chunkDocs
from embedder    import EmbeddingChunkedData
from vectorstore import saveVectorStore
from search      import search
from generator   import GenerateAnswer
from config      import CONFIG

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    history:  list = []

# ✅ API endpoint — React calls this
@app.post("/chat")
def chat(req: ChatRequest):
    results = search(req.question)
    response = GenerateAnswer(req.question, results, req.history)

    return {
        "answer":    response["answer"],
        "source":    response["source"],
        "from_docs": response["from_docs"]  # ✅ tell frontend where answer came from
    }

# ✅ run once to ingest docs
def ingest():
    docs = loadDocs(CONFIG["data_dir"])
    chunks = chunkDocs(docs)
    unique_chunks, embeddings = EmbeddingChunkedData(chunks)
    saveVectorStore(unique_chunks, embeddings)
    print("✅ Ingestion complete!")
    
    
# def chat():
#     print("Bot ready! Type 'exit' to quit.\n")
#     while True:
#         question = input("You: ").strip()
#         if not question:
#             continue
#         if question.lower() == "exit":
#             break
#         results = search(question)
#         answer  = GenerateAnswer(question, results)
#         print(f"\nBot: {answer}\n")

if __name__ == "__main__":
    # ingest()  # ← uncomment and run once
    # chat() # ← uncomment and do chat locally
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)