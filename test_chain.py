"""Тест полной RAG-цепочки с Groq."""
from rag.chain import ask


def main():
    questions = [
        "Какие симптомы артериальной гипертензии?",
        "Какие препараты первой линии для лечения АГ?",
        "Какая целевая цифра артериального давления?",
        "Какая погода на Марсе?",  # проверка на "не знаю"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ Вопрос: {q}")
        print(f"{'='*60}")
        result = ask(q)
        print(f"\n💬 Ответ:\n{result['answer']}")
        print(f"\n📚 Источников найдено: {len(result['sources'])}")
        for i, doc in enumerate(result["sources"], 1):
            page = doc.metadata.get("page", "?")
            page_num = page + 1 if isinstance(page, int) else page
            print(f"   [{i}] стр. {page_num}: {doc.page_content[:80]}...")


if __name__ == "__main__":
    main()