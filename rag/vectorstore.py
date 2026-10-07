import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

"""Векторное хранилище на ChromaDB."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from rag.embeddings import get_embeddings


def build_vectorstore(chunks: list[Document], force_recreate: bool = True) -> Chroma:
    """
    Создаёт (или пересоздаёт) векторное хранилище из чанков.

    force_recreate=True (по умолчанию) — удаляет старую коллекцию и создаёт заново.
    Это защита от дублей: если запустить индексацию дважды, не получим 2× чанков.
    """
    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Удаляем старую коллекцию, если она есть — иначе Chroma будет копить дубли
    if force_recreate and persist_dir.exists():
        try:
            old_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=get_embeddings(),
                collection_name="medical_docs",
            )
            old_store.delete_collection()
            print("   → Старая коллекция удалена (защита от дублей)")
        except Exception:
            # Первый запуск — коллекции ещё нет, это нормально
            pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name="medical_docs",
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """Загружает уже существующее хранилище (без пересоздания)."""
    return Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name="medical_docs",
    )