"""Derive search queries and labels from the user's goal (no hardcoded topic)."""

from __future__ import annotations

import re

_BOILERPLATE = (
    r"\b(create|build|write|make|generate|draft|produce)\b",
    r"\b(a|an|the|weekly|daily|monthly)\b",
    r"\b(newsletter|email|digest|bulletin|update)\b",
    r"\b(on|about|related to|regarding|covering|for)\b",
    r"\b(latest|recent|top|all|current|this week|today)\b",
    r"\b(news|headlines|stories|updates|articles)\b",
    r"\b(and|or|to|our|my|your|their|its|send|it|them|users|subscribers|readers)\b",
)

_GENERIC_FALLBACK = (
    "top news headlines this week",
    "latest breaking news today",
    "weekly news roundup",
)


def _clean_goal(goal: str) -> str:
    text = (goal or "").strip().lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    for pattern in _BOILERPLATE:
        text = re.sub(pattern, " ", text)
    words: list[str] = []
    seen: set[str] = set()
    for w in re.sub(r"\s+", " ", text).strip().split():
        if w and w not in seen:
            seen.add(w)
            words.append(w)
    return " ".join(words)


def derive_topic_label(goal: str) -> str:
    """Short label for UI/template, e.g. 'Sports Weekly'."""
    topic = _clean_goal(goal)
    if not topic:
        return "Newsletter"
    words = topic.split()[:3]
    label = " ".join(w.capitalize() for w in words)
    return f"{label} Weekly" if label else "Newsletter"


def derive_search_queries(goal: str, max_queries: int = 5) -> list[str]:
    """
    Build web search queries from the user's goal.
    Used as fallback when the plan LLM returns nothing useful.
    """
    topic = _clean_goal(goal)
    if not topic:
        return list(_GENERIC_FALLBACK[:max_queries])

    queries = [
        f"{topic} news this week",
        f"latest {topic} headlines",
        f"top {topic} news today",
        f"{topic} breaking news",
        f"{topic} updates",
    ]
    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out[:max_queries]
