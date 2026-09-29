"""Environment helpers for Groq LLM and search backends.

Reads Streamlit Cloud secrets first, then process env / local .env.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    """Prefer st.secrets (Streamlit Cloud), then environment variables."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value is None:
                return default
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


def get_groq_api_key() -> str | None:
    key = _secret("GROQ_API_KEY")
    return key or None


def get_groq_model() -> str:
    return _secret("GROQ_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"


def get_tavily_api_key() -> str | None:
    key = _secret("TAVILY_API_KEY")
    return key or None


def get_search_backend() -> str:
    """Return 'tavily' when keyed, otherwise 'duckduckgo'."""
    return "tavily" if get_tavily_api_key() else "duckduckgo"


def require_groq_api_key() -> str:
    key = get_groq_api_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to Streamlit Secrets (cloud) "
            "or to your local .env file."
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
