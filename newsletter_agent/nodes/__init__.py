"""LangGraph nodes: plan, research, summarize, write, critique, HITL, output."""

from __future__ import annotations

import json
from typing import Any

from newsletter_agent.llm import chat_json
from newsletter_agent.state import NewsletterState, append_log
from newsletter_agent.tools.html_builder import build_html_newsletter, save_newsletter
from newsletter_agent.tools.search import search_many

MAX_REVISIONS = 2
CRITIQUE_PASS_SCORE = 7.0


def plan_node(state: NewsletterState) -> dict:
    """Break the goal into steps + search queries (Groq)."""
    goal = state.get("goal") or ""
    data = chat_json(
        system=(
            "You are the planning module of a Newsletter Agent. "
            "Given a plain-English goal, produce a short execution plan and "
            "3-5 focused web search queries about recent AI agent news."
        ),
        user=json.dumps(
            {
                "goal": goal,
                "required_schema": {
                    "plan": ["step strings"],
                    "search_queries": ["query strings focused on AI agents / multi-agent news"],
                },
            }
        ),
    )
    plan = [str(x) for x in (data.get("plan") or [])][:8]
    queries = [str(x) for x in (data.get("search_queries") or [])][:5]
    if not queries:
        queries = [
            "AI agents news this week",
            "LangGraph multi-agent systems",
            "autonomous AI agent frameworks",
        ]
    if not plan:
        plan = [
            "Research latest AI agent news",
            "Summarize top articles",
            "Write and critique newsletter",
            "Simulate send",
        ]
    logs = append_log(state, f"Plan: {len(plan)} steps, {len(queries)} search queries")
    return {"plan": plan, "search_queries": queries, "logs": logs}


def research_node(state: NewsletterState) -> dict:
    """Run web search for planned queries (no LLM)."""
    queries = list(state.get("search_queries") or [])
    if not queries:
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
    """Pick and summarize top 5-7 articles (Groq)."""
    raw = list(state.get("raw_articles") or [])
    compact = [
        {
            "index": i,
            "title": a.get("title"),
            "url": a.get("url"),
            "snippet": (a.get("snippet") or "")[:500],
            "source": a.get("source"),
        }
        for i, a in enumerate(raw)
    ]
    data = chat_json(
        system=(
            "You are a news editor for an AI Agents weekly newsletter. "
            "Select the 5-7 most relevant, non-duplicate articles about AI agents, "
            "agent frameworks, multi-agent systems, or agent tooling. "
            "Write a crisp 2-3 sentence summary and a one-line relevance note for each."
        ),
        user=json.dumps(
            {
                "goal": state.get("goal"),
                "candidates": compact,
                "required_schema": {
                    "articles": [
                        {
                            "index": 0,
                            "title": "string",
                            "url": "string",
                            "summary": "string",
                            "relevance": "string",
                            "source": "string",
                        }
                    ]
                },
            }
        ),
    )
    selected: list[dict[str, Any]] = []
    for item in data.get("articles") or []:
        idx = item.get("index")
        base: dict[str, Any] = {}
        if isinstance(idx, int) and 0 <= idx < len(raw):
            base = dict(raw[idx])
        selected.append(
            {
                "title": item.get("title") or base.get("title") or "Untitled",
                "url": item.get("url") or base.get("url") or "",
                "snippet": base.get("snippet") or "",
                "source": item.get("source") or base.get("source") or "",
                "summary": item.get("summary") or base.get("snippet") or "",
                "relevance": item.get("relevance") or "",
            }
        )
        if len(selected) >= 7:
            break

    if not selected and raw:
        for a in raw[:6]:
            selected.append(
                {
                    "title": a.get("title") or "Untitled",
                    "url": a.get("url") or "",
                    "snippet": a.get("snippet") or "",
                    "source": a.get("source") or "",
                    "summary": a.get("snippet") or "",
                    "relevance": "Selected as fallback",
                }
            )

    logs = append_log(state, f"Summarize: kept top {len(selected)} articles")
    return {"top_articles": selected, "logs": logs}


def write_node(state: NewsletterState) -> dict:
    """Draft subject + intro/outro, render HTML via Jinja (Groq + template)."""
    articles = list(state.get("top_articles") or [])
    revision = int(state.get("revision_count") or 0)
    feedback = state.get("human_feedback") or ""
    critique = state.get("critique") or {}
    prior_feedback = critique.get("feedback") or ""

    data = chat_json(
        system=(
            "You write a clean weekly newsletter about AI agents. "
            "Return a punchy email subject, a short intro paragraph, and a short outro. "
            "Do not invent articles — only reference the provided ones. "
            "Tone: professional, clear, slightly enthusiastic."
        ),
        user=json.dumps(
            {
                "goal": state.get("goal"),
                "articles": articles,
                "revision_count": revision,
                "critique_feedback": prior_feedback,
                "human_feedback": feedback,
                "required_schema": {
                    "subject": "string",
                    "intro": "string",
                    "outro": "string",
                    "markdown": "full markdown body string",
                },
            }
        ),
    )
    subject = str(data.get("subject") or "AI Agents Weekly Digest")
    intro = str(data.get("intro") or "Here are this week's top AI agent stories.")
    outro = str(
        data.get("outro")
        or "You're receiving this because you subscribed to our AI Agents digest."
    )
    markdown = str(data.get("markdown") or "")
    if not markdown:
        lines = [f"# {subject}", "", intro, ""]
        for a in articles:
            lines.append(f"## [{a.get('title')}]({a.get('url')})")
            lines.append(a.get("summary") or "")
            lines.append("")
        lines.append(outro)
        markdown = "\n".join(lines)

    html = build_html_newsletter(
        subject=subject,
        intro=intro,
        articles=articles,
        outro=outro,
    )
    logs = append_log(
        state,
        f"Write: subject drafted (revision {revision})",
    )
    return {
        "draft_subject": subject,
        "draft_markdown": markdown,
        "draft_html": html,
        "human_feedback": None,
        "logs": logs,
    }


def critique_node(state: NewsletterState) -> dict:
    """Self-reflect on the draft; approve or request revision (Groq)."""
    data = chat_json(
        system=(
            "You are a strict newsletter editor. Score the draft 1-10 on relevance, "
            "clarity, structure, and usefulness for readers interested in AI agents. "
            "Approve only if score >= 7. If not approved, give concrete revision notes."
        ),
        user=json.dumps(
            {
                "goal": state.get("goal"),
                "subject": state.get("draft_subject"),
                "markdown": state.get("draft_markdown"),
                "article_count": len(state.get("top_articles") or []),
                "required_schema": {
                    "score": 8.0,
                    "feedback": "string",
                    "approved": True,
                },
            }
        ),
    )
    score = float(data.get("score") or 0)
    feedback = str(data.get("feedback") or "")
    approved = bool(data.get("approved")) and score >= CRITIQUE_PASS_SCORE
    revision = int(state.get("revision_count") or 0)
    if not approved and revision >= MAX_REVISIONS:
        approved = True
        feedback = (feedback + " | Max revisions reached; accepting draft.").strip()

    critique = {"score": score, "feedback": feedback, "approved": approved}
    logs = append_log(
        state,
        f"Critique: score={score:.1f} approved={approved}",
    )
    return {"critique": critique, "logs": logs}


def hitl_pause_node(state: NewsletterState) -> dict:
    """Pause for human approval in HITL mode."""
    logs = append_log(state, "HITL: awaiting human approve / revise")
    return {"logs": logs, "awaiting_human": True}


def output_node(state: NewsletterState) -> dict:
    """Simulate send: persist HTML + log subject."""
    html = state.get("draft_html") or ""
    subject = state.get("draft_subject") or "AI Agents Weekly"
    if not html:
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
            intro="Weekly AI Agents digest.",
            articles=preview,
        )
    path = save_newsletter(html)
    logs = append_log(state, f"Simulated send -> {path} | Subject: {subject}")
    return {
        "draft_html": html,
        "draft_subject": subject,
        "output_path": str(path),
        "awaiting_human": False,
        "logs": logs,
    }


def route_after_critique(state: NewsletterState) -> str:
    critique = state.get("critique") or {}
    approved = bool(critique.get("approved"))
    revision = int(state.get("revision_count") or 0)
    if not approved and revision < MAX_REVISIONS:
        return "revise"
    if (state.get("mode") or "autonomous") == "hitl":
        return "hitl"
    return "output"


def bump_revision_node(state: NewsletterState) -> dict:
    revision = int(state.get("revision_count") or 0) + 1
    logs = append_log(state, f"Revising draft (attempt {revision})")
    return {"revision_count": revision, "logs": logs}
