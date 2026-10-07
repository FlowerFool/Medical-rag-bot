"""Локальная модель эмбеддингов (HuggingFace)."""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Возвращает синглтон модели эмбеддингов.
    
    Модель all-MiniLM-L6-v2:
    - ~90 МБ
    - 384-мерный вектор
    - Быстрая на CPU (наш случай)
    - Хорошее качество для семантического поиска
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # для косинусной близости
    )