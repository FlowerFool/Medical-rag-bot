"""Загрузка PDF-документов с сохранением метаданных (номер страницы)."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Загружает PDF и возвращает список Document (по одному на страницу).
    
    Каждый Document содержит:
    - page_content: текст страницы
    - metadata: {source: путь, page: номер_страницы (0-indexed)}
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    if not pages:
        raise ValueError(f"PDF пустой или нечитаемый: {pdf_path}")

    return pages