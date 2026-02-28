CONFIG = {
    "data_dir":     "data/",              # where your docs are
    "model":        "BAAI/bge-small-en-v1.5",  # embedding model
    "llm_model":    "openai/gpt-3.5-turbo",    # LLM model
    "chunk_size":   500,                  # chunk size
    "top_k":        5,                    # results to retrieve
    "score_threshold": 0.35,             # min score
    "system_prompt": "You are a technical engineering assistant. Answer ONLY from the context provided. If answer is not in context, say \"Not found in the documents.\" Do NOT use your own knowledge."
}