import requests
from typing import List
from dotenv import load_dotenv
import os
from search import search
from config import CONFIG

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = os.getenv("API_URL")

# ============================================================
# CONSTANTS
# ============================================================

SMALL_TALK = [
    "hi", "hello", "how are you", "hey", "good morning", "good evening",
    "thanks", "thank you", "bye", "goodbye", "ok", "okay", "aoa",
    "assalam o alikum", "assalamualaikum", "wa alaikum assalam"
]

SYSTEM_PROMPT = """You are a senior technical engineering assistant specializing in Saudi Aramco Engineering Standards (SAES).

Rules:
- Answer EXACTLY what was asked — nothing more, nothing less
- Use chat history to understand follow-up questions like "it", "this", "them"
- Format answers with markdown: ## headings, **bold**, bullet points
- Keep answers proportional — short question = short answer, detailed question = detailed answer
- If not found in documents, answer from general engineering knowledge and mention it
"""

# ============================================================
# HELPER — rewrite follow-up questions
# ============================================================

def rewriteQuery(query: str, history: List[dict], headers: dict) -> str:

    if not history:
        return query

    followup_signals = ["it", "this", "them", "they", "those", "that", "same", "also", "more", "which", "and"]
    is_followup      = any(w in query.lower().split() for w in followup_signals) or len(query.split()) < 6

    if not is_followup:
        return query

    payload = {
        "model": CONFIG["llm_model"],
        "messages": [
            {"role": "system", "content": "Rewrite the follow-up question to be standalone. Return ONLY the rewritten question, nothing else."},
            *history[-6:],
            {"role": "user", "content": f"Rewrite: {query}"}
        ],
        "max_tokens": 80,
        "temperature": 0.1
    }

    try:
        result    = requests.post(API_URL, json=payload, headers=headers).json()
        rewritten = result["choices"][0]["message"]["content"].strip()
        print(f"🔄 {query} → {rewritten}")
        return rewritten
    except:
        return query


# ============================================================
# HELPER — call LLM
# ============================================================

def callLLM(messages: list, headers: dict, max_tokens: int = 800, temperature: float = 0.2) -> str | None:

    payload = {
        "model":       CONFIG["llm_model"],
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature
    }

    try:
        result = requests.post(API_URL, json=payload, headers=headers).json()

        if "error" in result:
            print(f"[ERROR] API: {result['error']}")
            return None

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return None


# ============================================================
# MAIN FUNCTION
# ============================================================

def GenerateAnswer(query: str, results: List[dict], history: List[dict] = []) -> dict:

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost",
        "X-Title":       "RAG App"
    }

    # ── Step 1: small talk ──────────────────────────────────
    if query.strip().lower() in SMALL_TALK or len(query.strip()) < 8:
        answer = callLLM([
            {"role": "system", "content": "You are a friendly engineering assistant called Nexus RAG. Respond briefly to greetings."},
            {"role": "user",   "content": query}
        ], headers, max_tokens=100, temperature=0.7)

        return {"answer": answer or "Hello! How can I help?", "source": None, "from_docs": False}

    # ── Step 2: rewrite query for better search ─────────────
    search_query = rewriteQuery(query, history, headers)

    # ── Step 3: search documents ────────────────────────────
    search_results = search(search_query)
    top_results    = [r for r in search_results if r["score"] > CONFIG["score_threshold"]]

    # ── Step 4: answer from docs ────────────────────────────
    if top_results:
        context = "\n\n".join([
            f"Source: {r['source'].split(chr(92))[-1]}\n{r['content']}"
            for r in top_results
        ])

        prompt = f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer exactly what was asked using the context and chat history."""

        answer = callLLM([
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user",   "content": prompt}
        ], headers, max_tokens=800, temperature=0.2)

        print(f"📄 From docs: {query[:60]}")
        return {"answer": answer, "source": top_results[0]["source"].split("\\")[-1], "from_docs": True}

    # ── Step 5: answer from general knowledge ───────────────
    answer = callLLM([
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user",   "content": f"Answer from general engineering knowledge (mention it's not from documents):\n{query}"}
    ], headers, max_tokens=800, temperature=0.4)

    print(f"🌐 General: {query[:60]}")
    return {"answer": answer, "source": "General Knowledge", "from_docs": False}
