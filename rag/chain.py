"""RAG-цепочка: retrieval → prompt → LLM → ответ с источником."""
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from config import settings
from rag.vectorstore import load_vectorstore


def format_docs(docs: list[Document]) -> str:
    """
    Склеивает найденные чанки в один текст с указанием страницы.
    Формат: [стр. N] текст чанка
    """
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        page_num = page + 1 if isinstance(page, int) else page
        parts.append(f"[стр. {page_num}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def load_system_prompt() -> str:
    """Читает системный промпт из файла."""
    prompt_path = Path("prompts/system.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Промпт не найден: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Синглтон LLM-клиента Groq."""
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
        max_tokens=500,
    )


@lru_cache(maxsize=1)
def _get_retriever():
    """
    Retriever с MMR (Maximum Marginal Relevance).
    
    MMR берёт fetch_k кандидатов, затем выбирает k так, чтобы они были
    релевантны вопросу, но НЕ дублировали друг друга. Решает проблему
    "три почти одинаковых чанка в ответе".
    """
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.RETRIEVER_K,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


def build_chain():
    """
    Собирает RAG-цепочку:
    вопрос → retriever → top-k чанков → prompt → LLM → текст ответа
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_system_prompt()),
        ("human", "{question}"),
    ])

    chain = (
        {"context": _get_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain


def ask(question: str) -> dict:
    source_docs = _get_retriever().invoke(question)

    chain = build_chain()
    try:
        answer = chain.invoke(question)
    except Exception as e:
        print(f"⚠️  Ошибка LLM: {type(e).__name__}: {e}")
        answer = ""

    # Логируем что вернула модель
    print(f"🔍 DEBUG: len(answer)={len(answer)}, repr={repr(answer[:100])}")

    if not answer or not answer.strip():
        answer = "Не удалось получить ответ от модели. Попробуйте переформулировать вопрос."

    return {"answer": answer, "sources": source_docs}