"""Разбивка страниц PDF на чанки для эмбеддингов."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def split_into_chunks(pages: list[Document]) -> list[Document]:
    """
    Режет страницы на перекрывающиеся чанки.
    
    - chunk_size=500: ~500 символов на чанк (золотая середина для медтекстов)
    - chunk_overlap=50: соседние чанки пересекаются на 50 символов,
      чтобы не терять смысл на границе
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # Разделители в порядке приоритета: сначала режем по абзацам,
        # потом по предложениям, потом по словам
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(pages)

    # Нумеруем чанки — пригодится для отладки и кнопки "показать чанки"
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks