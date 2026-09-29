"""LangGraph Newsletter Agent — full Groq-powered pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from newsletter_agent.config import get_groq_api_key
from newsletter_agent.nodes import (
    bump_revision_node,
    critique_node,
    hitl_pause_node,
    output_node,
    plan_node,
    research_node,
    route_after_critique,
    summarize_node,
    write_node,
)
from newsletter_agent.state import NewsletterState, append_log, initial_state


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
    awaiting_human: bool = False
    critique: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)


def _build_pre_llm_graph():
    graph = StateGraph(NewsletterState)
    graph.add_node("research", research_node)
    graph.add_node("output", output_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "output")
    graph.add_edge("output", END)
    return graph.compile()


def _build_full_graph():
    graph = StateGraph(NewsletterState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("write", write_node)
    graph.add_node("critique", critique_node)
    graph.add_node("bump_revision", bump_revision_node)
    graph.add_node("hitl_pause", hitl_pause_node)
    graph.add_node("output", output_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", "write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "revise": "bump_revision",
            "hitl": "hitl_pause",
            "output": "output",
        },
    )
    graph.add_edge("bump_revision", "write")
    graph.add_edge("hitl_pause", END)
    graph.add_edge("output", END)
    return graph.compile()


def build_newsletter_graph(*, llm_enabled: bool | None = None):
    if llm_enabled is None:
        llm_enabled = bool(get_groq_api_key())
    if llm_enabled:
        return _build_full_graph()
    return _build_pre_llm_graph()


def _to_result(final: NewsletterState, mode: str, llm_ready: bool) -> AgentResult:
    return AgentResult(
        subject=final.get("draft_subject") or "",
        html=final.get("draft_html") or "",
        markdown=final.get("draft_markdown") or "",
        output_path=final.get("output_path"),
        articles=list(final.get("top_articles") or final.get("raw_articles") or []),
        logs=list(final.get("logs") or []),
        mode=mode,
        llm_ready=llm_ready,
        awaiting_human=bool(final.get("awaiting_human")),  # type: ignore[arg-type]
        critique=dict(final.get("critique") or {}),
        state=dict(final),
    )


def run_newsletter_agent(
    goal: str,
    mode: Literal["autonomous", "hitl"] = "autonomous",
) -> AgentResult:
    """
    Single entrypoint: plan -> research -> summarize -> write -> critique -> send.

    In HITL mode, returns after critique with awaiting_human=True until
    resume_newsletter_agent(...) is called.
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
    # Extra field used by HITL pause (declared loosely on state dict)
    state["awaiting_human"] = False  # type: ignore[typeddict-item]

    graph = build_newsletter_graph(llm_enabled=llm_ready)
    final: NewsletterState = graph.invoke(state)
    return _to_result(final, mode=mode, llm_ready=llm_ready)


def resume_newsletter_agent(
    prior_state: dict[str, Any],
    decision: Literal["approve", "revise"],
    feedback: str = "",
) -> AgentResult:
    """Continue after HITL pause: approve (simulate send) or revise with feedback."""
    state: NewsletterState = dict(prior_state)  # type: ignore[assignment]
    mode = state.get("mode") or "hitl"
    llm_ready = bool(get_groq_api_key())

    if decision == "approve":
        state["logs"] = append_log(state, "HITL: human approved")
        state["awaiting_human"] = False  # type: ignore[typeddict-item]
        updated = output_node(state)
        state.update(updated)  # type: ignore[arg-type]
        return _to_result(state, mode=str(mode), llm_ready=llm_ready)

    state["human_feedback"] = (feedback or "").strip() or "Please improve clarity and relevance."
    state["revision_count"] = int(state.get("revision_count") or 0) + 1
    state["awaiting_human"] = False  # type: ignore[typeddict-item]
    state["logs"] = append_log(state, f"HITL: human requested revise - {state['human_feedback']}")

    # Re-run write -> critique -> route
    for node in (write_node, critique_node):
        state.update(node(state))  # type: ignore[arg-type]

    route = route_after_critique(state)
    if route == "revise":
        state.update(bump_revision_node(state))  # type: ignore[arg-type]
        state.update(write_node(state))  # type: ignore[arg-type]
        state.update(critique_node(state))  # type: ignore[arg-type]
        route = route_after_critique(state)

    if route == "hitl" or (mode == "hitl" and route != "output"):
        state.update(hitl_pause_node(state))  # type: ignore[arg-type]
    else:
        state.update(output_node(state))  # type: ignore[arg-type]

    return _to_result(state, mode=str(mode), llm_ready=llm_ready)
