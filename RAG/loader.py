from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredExcelLoader,
    CSVLoader, JSONLoader, UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader, UnstructuredRTFLoader,
    UnstructuredXMLLoader, UnstructuredEmailLoader, UnstructuredEPubLoader,
    UnstructuredImageLoader,
)
from pathlib import Path
import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)

def loadDocs(dir: str):

    data_path = (Path.cwd() / dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    if not data_path.exists():
        print("[ERROR] Folder not found!")
        return []

    FILE_LOADERS = {
        PyPDFLoader:                  ["*.pdf"],
        Docx2txtLoader:               ["*.docx", "*.doc"],
        UnstructuredExcelLoader:      ["*.xlsx", "*.xls"],
        CSVLoader:                    ["*.csv"],
        UnstructuredPowerPointLoader: ["*.pptx", "*.ppt"],
        UnstructuredHTMLLoader:       ["*.html", "*.htm"],
        UnstructuredMarkdownLoader:   ["*.md", "*.markdown"],
        UnstructuredRTFLoader:        ["*.rtf"],
        UnstructuredXMLLoader:        ["*.xml"],
        UnstructuredEmailLoader:      ["*.eml", "*.msg"],
        UnstructuredEPubLoader:       ["*.epub"],
        UnstructuredImageLoader:      ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"],
        TextLoader:                   ["*.txt", "*.yaml", "*.yml"],
    }

    for loader_class, extensions in FILE_LOADERS.items():
        files = []
        for ext in extensions:
            files.extend(data_path.glob(f'**/{ext}'))
        print(f"[DEBUG] Found {len(files)} {extensions} files")
        for file in files:
            try:
                loader = loader_class(str(file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"[ERROR] {file.name}: {e}")

    print(f"\n Total documents loaded: {len(documents)}")
    return documents