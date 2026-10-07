"""Быстрый тест RAG-пайплайна без Telegram."""
from rag.loader import load_pdf
from rag.chunking import split_into_chunks
from rag.vectorstore import build_vectorstore
from config import settings


def main():
    print("1. Загружаю PDF...")
    pages = load_pdf(settings.SAMPLE_PDF_PATH)
    print(f"   → Страниц: {len(pages)}")

    print("2. Режу на чанки...")
    chunks = split_into_chunks(pages)
    print(f"   → Чанков: {len(chunks)}")
    print(f"   → Пример первого чанка:\n{chunks[0].page_content[:200]}...")

    print("3. Строю векторное хранилище...")
    vectorstore = build_vectorstore(chunks)
    print(f"   → Готово. Чанков в БД: {vectorstore._collection.count()}")

    print("4. Тестовый поиск...")
    query = "симптомы хронического ринита"  # замени на что-то из твоего PDF
    results = vectorstore.similarity_search(query, k=3)
    print(f"   → Найдено: {len(results)}")
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page", "?")
        print(f"   [{i}] стр.{page}: {doc.page_content[:150]}...")


if __name__ == "__main__":
    main()