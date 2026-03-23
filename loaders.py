from pathlib import Path
from typing import List

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader, TextLoader

from logger import logger

def load_document(path: str) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
        logger.info(f'Файл {file_path} загружен')
        return loader.load()
    if suffix == ".docx":
        loader = Docx2txtLoader(file_path)
        logger.info(f'Файл {file_path} загружен')
        return loader.load()
    if suffix == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        logger.info(f'Файл {file_path} загружен')
        return loader.load()
    
    raise ValueError(f"Unsupported file type: {suffix}")
