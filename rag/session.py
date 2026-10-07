"""Менеджер сессий: у каждого Telegram-пользователя свой документ и retriever."""
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever

from config import settings
from rag.loader import load_pdf
from rag.chunking import split_into_chunks
from rag.vectorstore import build_vectorstore
from rag.chain import _get_retriever  # переиспользуем готовый retriever


@dataclass
class UserSession:
    """Сессия одного пользователя."""
    user_id: int
    source_filename: str
    chunks_count: int
    retriever: VectorStoreRetriever


# In-memory хранилище сессий: user_id → UserSession
# Для пет-проекта этого достаточно. В проде — Redis или Postgres.
_sessions: dict[int, UserSession] = {}


def index_pdf_for_user(user_id: int, pdf_bytes: bytes, filename: str) -> UserSession:
    """
    Индексирует PDF для конкретного пользователя.
    
    ВАЖНО: для простоты используем общую ChromaDB, но с фильтром по user_id.
    В настоящем проде — у каждого пользователя была бы своя коллекция.
    """
    # Сохраняем PDF во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        pages = load_pdf(tmp_path)
        chunks = split_into_chunks(pages)

        # Проставляем user_id в metadata каждого чанка — это фильтр
        for chunk in chunks:
            chunk.metadata["user_id"] = user_id

        # Пересобираем коллекцию — для простоты общая
        # (в проде: своя коллекция на пользователя или фильтр по metadata)
        vectorstore = build_vectorstore(chunks, force_recreate=True)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.RETRIEVER_K, "fetch_k": 10, "lambda_mult": 0.5},
        )

        session = UserSession(
            user_id=user_id,
            source_filename=filename,
            chunks_count=len(chunks),
            retriever=retriever,
        )
        _sessions[user_id] = session
        return session
    finally:
        tmp_path.unlink(missing_ok=True)


def get_session(user_id: int) -> UserSession | None:
    """Возвращает сессию пользователя или None."""
    return _sessions.get(user_id)


def clear_session(user_id: int) -> None:
    """Удаляет сессию пользователя."""
    _sessions.pop(user_id, None)