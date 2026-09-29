"""
LangGraph Newsletter Agent.

Pre-LLM phase: research + HTML output work without Groq.
Full plan → summarize → write → critique loop lands after GROQ_API_KEY is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from newsletter_agent.config import get_groq_api_key
from newsletter_agent.nodes import output_node, research_node
from newsletter_agent.state import NewsletterState, initial_state


@dataclass
class AgentResult:
    subject: str
    html: str
    markdown: str
    output_path: str | None
    articles: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    mode: str = "autonomous"
    llm_ready: bool = False


def _build_pre_llm_graph():
    """Research → output only (usable before Groq key)."""
    graph = StateGraph(NewsletterState)
    graph.add_node("research", research_node)
    graph.add_node("output", output_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "output")
    graph.add_edge("output", END)
    return graph.compile()


def build_newsletter_graph(*, llm_enabled: bool | None = None):
    """
    Compile the agent graph.

    When Groq is not configured, returns the pre-LLM research→output graph.
    Full multi-step graph is enabled once GROQ_API_KEY is present (next step).
    """
    if llm_enabled is None:
        llm_enabled = bool(get_groq_api_key())

    if not llm_enabled:
        return _build_pre_llm_graph()

    # Full graph will be wired in the Groq integration step:
    # plan → research → summarize → write → critique → (HITL) → output
    raise NotImplementedError(
        "Full LangGraph (plan/summarize/write/critique) pending Groq integration."
    )


def run_newsletter_agent(
    goal: str,
    mode: Literal["autonomous", "hitl"] = "autonomous",
) -> AgentResult:
    """
    Single entrypoint for the Newsletter Agent.

    Pre-LLM: researches default AI-agent queries and writes a draft HTML file.
    Post-LLM: will run the full planning → research → writing → critique pipeline.
    """
    goal = (goal or "").strip()
    if not goal:
        raise ValueError("goal must be a non-empty string")

    llm_ready = bool(get_groq_api_key())
    state = initial_state(goal=goal, mode=mode)
    state["logs"] = [
        f"Goal: {goal}",
        f"Mode: {mode}",
        f"LLM ready: {llm_ready}",
    ]

    if llm_ready:
        raise NotImplementedError(
            "Groq key detected, but LLM nodes are not wired yet. "
            "Complete the Groq integration step next."
        )

    graph = build_newsletter_graph(llm_enabled=False)
    final: NewsletterState = graph.invoke(state)

    return AgentResult(
        subject=final.get("draft_subject") or "",
        html=final.get("draft_html") or "",
        markdown=final.get("draft_markdown") or "",
        output_path=final.get("output_path"),
        articles=list(final.get("raw_articles") or []),
        logs=list(final.get("logs") or []),
        mode=mode,
        llm_ready=False,
    )
