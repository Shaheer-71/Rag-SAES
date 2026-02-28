import requests
from typing import List
from dotenv import load_dotenv
import os
from search import search
from config import CONFIG

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = os.getenv("API_URL")

SYSTEM_PROMPT = """You are a technical engineering assistant.
Answer ONLY from the context provided.
If answer is not in context, say "Not found in the documents."
Do NOT use your own knowledge."""

def GenerateAnswer(query: str, results: List[dict]) -> str:

    top_results = [r for r in results if r["score"] > 0.35]

    if not top_results:
        return "No relevant documents found."

    context = "\n\n".join([
        f"Source: {r['source'].split(chr(92))[-1]}\n{r['content']}"
        for r in top_results
    ])

    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer strictly from the context:"

    payload = {
        "model": CONFIG["llm_model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "RAG App"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        result = response.json()
        if "error" in result:
            print(f"[ERROR] API: {result['error']}")
            return None
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return None