"""Research tools: Tavily when keyed, else DuckDuckGo (resilient)."""

from __future__ import annotations

import os
from typing import Any


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the public web for `query`.

    Prefers Tavily if TAVILY_API_KEY is set; otherwise uses DuckDuckGo
    via the `ddgs` package (no API key required). Returns [] on soft failures.
    """
    query = (query or "").strip()
    if not query:
        return []

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            return _search_tavily(query, max_results=max_results, api_key=tavily_key)
        except Exception:
            # Fall through to DuckDuckGo if Tavily fails.
            pass

    try:
        return _search_duckduckgo(query, max_results=max_results)
    except Exception:
        return []


def search_many(queries: list[str], max_results_per_query: int = 5) -> list[dict[str, Any]]:
    """Run multiple queries and dedupe by URL. Never raises on empty backends."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for q in queries:
        try:
            items = web_search(q, max_results=max_results_per_query)
        except Exception:
            items = []
        for item in items:
            url = (item.get("url") or "").rstrip("/")
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(item)
    return merged


def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict[str, Any]]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
    )
    results: list[dict[str, Any]] = []
    for row in response.get("results") or []:
        results.append(
            {
                "title": row.get("title") or "Untitled",
                "url": row.get("url") or "",
                "snippet": row.get("content") or "",
                "source": "tavily",
            }
        )
    return results


def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    results: list[dict[str, Any]] = []
    # Try a couple of backends; DDG rate-limits / returns empty intermittently.
    backends = (None, "duckduckgo", "bing", "google")
    last_error: Exception | None = None
    for backend in backends:
        try:
            kwargs: dict[str, Any] = {"max_results": max_results}
            if backend:
                kwargs["backend"] = backend
            with DDGS() as ddgs:
                rows = list(ddgs.text(query, **kwargs))
            if not rows:
                continue
            for row in rows:
                results.append(
                    {
                        "title": row.get("title") or "Untitled",
                        "url": row.get("href") or row.get("url") or "",
                        "snippet": row.get("body") or row.get("snippet") or "",
                        "source": f"duckduckgo:{backend or 'auto'}",
                    }
                )
            if results:
                return results
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error and not results:
        raise last_error
    return results
