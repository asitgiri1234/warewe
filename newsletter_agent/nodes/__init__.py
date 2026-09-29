"""Graph nodes. LLM-backed steps raise until Groq integration is enabled."""

from __future__ import annotations

from newsletter_agent.config import get_groq_api_key
from newsletter_agent.state import NewsletterState, append_log
from newsletter_agent.tools.html_builder import build_html_newsletter, save_newsletter
from newsletter_agent.tools.search import search_many

_LLM_PENDING = (
    "LLM nodes are not active yet. Add GROQ_API_KEY to .env to enable "
    "plan / summarize / write / critique."
)


def _require_llm() -> None:
    if not get_groq_api_key():
        raise RuntimeError(_LLM_PENDING)


def plan_node(state: NewsletterState) -> dict:
    """Break the goal into steps + search queries (LLM)."""
    _require_llm()
    # Wired in next PR once Groq key is available.
    raise NotImplementedError("plan_node: Groq integration pending")


def research_node(state: NewsletterState) -> dict:
    """Run web search for planned queries (no LLM)."""
    queries = list(state.get("search_queries") or [])
    if not queries:
        # Sensible defaults so research can be exercised before the plan LLM lands.
        queries = [
            "AI agents news this week",
            "LangGraph multi-agent systems",
            "autonomous AI agent frameworks",
        ]
    articles = search_many(queries, max_results_per_query=5)
    logs = append_log(
        state,
        f"Research: {len(queries)} queries -> {len(articles)} unique articles",
    )
    return {
        "search_queries": queries,
        "raw_articles": articles,
        "logs": logs,
    }


def summarize_node(state: NewsletterState) -> dict:
    """Pick and summarize top 5–7 articles (LLM)."""
    _require_llm()
    raise NotImplementedError("summarize_node: Groq integration pending")


def write_node(state: NewsletterState) -> dict:
    """Draft subject + markdown, then render HTML (LLM + Jinja)."""
    _require_llm()
    raise NotImplementedError("write_node: Groq integration pending")


def critique_node(state: NewsletterState) -> dict:
    """Self-reflect on the draft; approve or request revision (LLM)."""
    _require_llm()
    raise NotImplementedError("critique_node: Groq integration pending")


def output_node(state: NewsletterState) -> dict:
    """Simulate send: persist HTML + log subject (no LLM)."""
    html = state.get("draft_html") or ""
    subject = state.get("draft_subject") or "AI Agents Weekly"
    if not html:
        # Fallback demo path: render from raw research snippets.
        articles = state.get("top_articles") or state.get("raw_articles") or []
        preview = [
            {
                "title": a.get("title", "Untitled"),
                "url": a.get("url", ""),
                "summary": a.get("summary") or a.get("snippet") or "",
                "source": a.get("source", ""),
                "relevance": a.get("relevance", ""),
            }
            for a in articles[:7]
        ]
        html = build_html_newsletter(
            subject=subject,
            intro="Draft newsletter built from research results (LLM writing pending).",
            articles=preview,
        )
    path = save_newsletter(html)
    logs = append_log(state, f"Simulated send -> {path} | Subject: {subject}")
    return {
        "draft_html": html,
        "draft_subject": subject,
        "output_path": str(path),
        "logs": logs,
    }
