"""Утилиты для форматирования ответов в Telegram."""
from langchain_core.documents import Document


def format_answer_for_telegram(answer: str, sources: list[Document]) -> str:
    """
    Форматирует ответ RAG для отправки в Telegram.
    
    Telegram MarkdownV2 капризен к спецсимволам, поэтому используем HTML —
    он проще в экранировании.
    """
    # Обрезаем ответ если слишком длинный (лимит Telegram — 4096 символов)
    if len(answer) > 3500:
        answer = answer[:3500] + "..."

    return f"{answer}"


def format_chunks_for_telegram(sources: list[Document]) -> str:
    """
    Форматирует найденные чанки для кнопки "Показать чанки".
    """
    if not sources:
        return "Ничего не найдено."

    parts = ["📚 <b>Найденные фрагменты:</b>\n"]
    for i, doc in enumerate(sources, 1):
        page = doc.metadata.get("page", "?")
        page_num = page + 1 if isinstance(page, int) else page
        text = doc.page_content.strip().replace("\n", " ")
        # Обрезаем до 300 символов, чтобы влезло в сообщение
        if len(text) > 300:
            text = text[:300] + "..."
        parts.append(f"\n<b>[{i}] Стр. {page_num}</b>\n<i>{text}</i>")

    return "\n".join(parts)


def get_pages_summary(sources: list[Document]) -> str:
    """Короткая сводка — какие страницы нашлись. Для inline-кнопки."""
    pages = []
    for doc in sources:
        page = doc.metadata.get("page", "?")
        page_num = page + 1 if isinstance(page, int) else page
        if page_num not in pages:
            pages.append(page_num)
    return ", ".join(f"стр. {p}" for p in pages)