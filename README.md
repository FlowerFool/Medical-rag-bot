# 🩺 Medical RAG Assistant

Telegram-бот, который отвечает на вопросы по медицинским документам **только на основе загруженного PDF**, с указанием источника.

## 🚀 Стек
- Python 3.11
- LangChain
- ChromaDB (векторная БД)
- HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Groq API (LLM)
- aiogram (Telegram)

## 📦 Как запустить

```bash
git clone <repo>
cd medical-rag-bot
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
copy .env.example .env
# вставь свои ключи в .env
python bot.py