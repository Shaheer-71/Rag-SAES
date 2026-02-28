from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import hashlib

CHUNK_CONFIG = {
    "pdf": (500, 100), "docx": (500, 100), "txt": (400, 80),
    "csv": (300, 50),  "xlsx": (300, 50),  "json": (300, 50),
    "pptx": (400, 80), "html": (500, 100), "md": (500, 100),
    "py": (1500, 200), "js": (1500, 200),
}

def cleanText(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunkDocs(docs):
    if not docs:
        return []

    chunks = []
    for doc in docs:
        doc.page_content = cleanText(doc.page_content)
        ext = doc.metadata.get("source", "").split(".")[-1].lower()
        size, overlap = CHUNK_CONFIG.get(ext, (500, 100))
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        doc_chunks = splitter.split_documents([doc])

        for chunk in doc_chunks:
            if len(chunk.page_content.strip()) > 50:
                chunk.metadata["chunk_id"] = len(chunks)
                chunk.metadata["file_type"] = ext
                chunk.metadata["hash_id"] = hashlib.md5(chunk.page_content.encode()).hexdigest()
                chunks.append(chunk)

    print(f"✅ {len(chunks)} chunks from {len(docs)} docs")
    return chunks