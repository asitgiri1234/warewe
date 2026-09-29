"""Shared LangGraph state for the Newsletter Agent."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class Article(TypedDict, total=False):
    title: str
    url: str
    snippet: str
    source: str
    summary: str
    relevance: str


class CritiqueResult(TypedDict, total=False):
    score: float
    feedback: str
    approved: bool


class NewsletterState(TypedDict, total=False):
    """Mutable graph state passed between nodes."""

    goal: str
    mode: Literal["autonomous", "hitl"]
    plan: list[str]
    search_queries: list[str]
    raw_articles: list[dict[str, Any]]
    top_articles: list[Article]
    draft_subject: str
    draft_markdown: str
    draft_html: str
    critique: CritiqueResult
    revision_count: int
    human_feedback: str | None
    awaiting_human: bool
    output_path: str | None
    logs: list[str]


def initial_state(
    goal: str,
    mode: Literal["autonomous", "hitl"] = "autonomous",
) -> NewsletterState:
    return {
        "goal": goal,
        "mode": mode,
        "plan": [],
        "search_queries": [],
        "raw_articles": [],
        "top_articles": [],
        "draft_subject": "",
        "draft_markdown": "",
        "draft_html": "",
        "critique": {"score": 0.0, "feedback": "", "approved": False},
        "revision_count": 0,
        "human_feedback": None,
        "awaiting_human": False,
        "output_path": None,
        "logs": [],
    }


def append_log(state: NewsletterState, message: str) -> list[str]:
    logs = list(state.get("logs") or [])
    logs.append(message)
    return logs
