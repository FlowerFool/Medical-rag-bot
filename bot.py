"""Telegram-бот: медицинский RAG-ассистент по загруженному PDF."""
import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from config import settings
from rag.chain import ask
from rag.session import index_pdf_for_user, get_session, clear_session
from bot_utils import (
    format_answer_for_telegram,
    format_chunks_for_telegram,
    get_pages_summary,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dp = Dispatcher()

# Храним последние sources для кнопки "показать чанки" — по user_id
_last_sources: dict[int, list] = {}


# ============================================================
# /start — приветствие
# ============================================================

@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 <b>Привет!</b>\n\n"
        "Я — медицинский RAG-ассистент. Я отвечаю на вопросы "
        "<b>только по тому документу, который ты мне пришлёшь</b>, "
        "и всегда указываю источник.\n\n"
        "📄 <b>Как пользоваться:</b>\n"
        "1. Отправь мне PDF с медицинским документом\n"
        "2. Дождись индексации (30–60 секунд)\n"
        "3. Задавай любые вопросы по этому документу\n\n"
        "⚠️ Я не ставлю диагнозы и не назначаю лечение — "
        "только помогаю найти информацию в тексте.",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "ℹ️ <b>Что я умею:</b>\n\n"
        "• Принимать PDF и индексировать его содержимое\n"
        "• Отвечать на вопросы <b>по этому PDF</b>\n"
        "• Показывать, какие фрагменты документа я использовал\n"
        "• Говорить «нет ответа», если в PDF его нет\n\n"
        "<b>Команды:</b>\n"
        "/start — приветствие\n"
        "/help — эта справка\n"
        "/reset — забыть текущий документ"
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    clear_session(message.from_user.id)
    _last_sources.pop(message.from_user.id, None)
    await message.answer("🗑 Документ удалён. Отправь новый PDF.")


# ============================================================
# Приём PDF — индексация
# ============================================================

@dp.message(F.document)
async def handle_document(message: Message) -> None:
    doc = message.document

    if not doc.file_name.lower().endswith(".pdf"):
        await message.answer("⚠️ Я работаю только с PDF. Пришли PDF-файл.")
        return

    status = await message.answer("⏳ Скачиваю и индексирую документ… Это займёт 30–60 секунд.")

    try:
        file = await message.bot.get_file(doc.file_id)
        pdf_bytes = await message.bot.download_file(file.file_path)
        pdf_bytes = pdf_bytes.read() if hasattr(pdf_bytes, "read") else pdf_bytes

        session = await asyncio.to_thread(
            index_pdf_for_user,
            message.from_user.id,
            pdf_bytes,
            doc.file_name,
        )

        await status.edit_text(
            f"✅ <b>Документ готов!</b>\n\n"
            f"📄 {session.source_filename}\n"
            f"🧩 Чанков: <b>{session.chunks_count}</b>\n\n"
            f"Теперь задавай вопросы по этому документу.",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.exception("Ошибка индексации")
        await status.edit_text(
            f"❌ Не удалось обработать PDF.\n\n<code>{type(e).__name__}: {e}</code>",
            parse_mode=ParseMode.HTML,
        )


# ============================================================
# Обработка текстовых вопросов
# ============================================================

@dp.message(F.text)
async def handle_question(message: Message) -> None:
    user_id = message.from_user.id

    if not get_session(user_id):
        await message.answer(
            "📄 Сначала пришли мне PDF-документ, по которому ты хочешь задавать вопросы."
        )
        return

    thinking = await message.answer("🤔 Думаю…")

    try:
        result = await asyncio.to_thread(ask, message.text)
        answer = result["answer"]
        sources = result["sources"]

        # Запоминаем sources — для кнопки "показать чанки"
        _last_sources[user_id] = sources

        # Кнопка "показать чанки"
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text=f"📚 Показать чанки ({get_pages_summary(sources)})",
                callback_data="show_chunks",
            )]
        ])

        await thinking.edit_text(
            format_answer_for_telegram(answer, sources),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )
    except Exception as e:
        logger.exception("Ошибка при генерации ответа")
        await thinking.edit_text(
            f"❌ Ошибка: <code>{type(e).__name__}: {e}</code>",
            parse_mode=ParseMode.HTML,
        )


# ============================================================
# Callback: показать чанки
# ============================================================

@dp.callback_query(F.data == "show_chunks")
async def show_chunks(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    sources = _last_sources.get(user_id, [])

    if not sources:
        await callback.answer("Нет данных для показа", show_alert=True)
        return

    await callback.message.answer(
        format_chunks_for_telegram(sources),
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()


# ============================================================
# Запуск
# ============================================================

async def main() -> None:
    settings.validate()
    bot = Bot(
        token=settings.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())