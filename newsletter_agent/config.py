"""Environment helpers. LLM client wired after GROQ_API_KEY is provided."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def get_groq_api_key() -> str | None:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key or None


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()


def require_groq_api_key() -> str:
    key = get_groq_api_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key."
        )
    return key


@lru_cache(maxsize=1)
def get_chat_model():
    """
    Return a LangChain ChatGroq instance.

    Called only by LLM-backed nodes (plan / summarize / write / critique)
    once the key is available — not required for search or HTML tools.
    """
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=require_groq_api_key(),
        model=get_groq_model(),
        temperature=0.3,
    )
