"""Загрузка настроек из .env."""
import os

# Отключаем телеметрию ChromaDB (иначе спамит в консоль)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "3"))

    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    SAMPLE_PDF_PATH: str = os.getenv("SAMPLE_PDF_PATH", "./data/sample_meddoc.pdf")

    @classmethod
    def validate(cls) -> None:
        """Проверка, что критичные секреты на месте."""
        missing = []
        if not cls.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if missing:
            raise ValueError(f"Отсутствуют переменные в .env: {', '.join(missing)}")


settings = Settings()