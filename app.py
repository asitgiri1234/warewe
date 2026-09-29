"""Streamlit UI for the Newsletter Agent (Groq + LangGraph)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from newsletter_agent.config import get_groq_api_key, get_groq_model, get_search_backend
from newsletter_agent.graph import resume_newsletter_agent, stream_newsletter_agent

DEFAULT_GOAL = (
    "Create a weekly newsletter on latest AI agent news and send it to our subscribers."
)

PIPELINE_STEPS = [
    ("plan", "Plan", "Break goal into search queries"),
    ("research", "Research", "Fetch news for your goal topic"),
    ("summarize", "Summarize", "Pick top 5-7 articles"),
    ("write", "Write", "Draft subject + newsletter"),
    ("critique", "Critique", "Self-review and revise"),
    ("output", "Send", "Simulate email delivery"),
]

NODE_TO_STEP = {
    "plan": "plan",
    "research": "research",
    "summarize": "summarize",
    "write": "write",
    "critique": "critique",
    "bump_revision": "write",
    "hitl_pause": "output",
    "output": "output",
}

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
    .hero {
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 55%, #ffffff 100%);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
    }
    .hero h1 { margin: 0 0 0.35rem 0; font-size: 2rem; color: #0f172a; }
    .hero p { margin: 0; color: #475569; font-size: 1rem; line-height: 1.55; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        min-height: 88px;
    }
    .metric-label { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { color: #0f172a; font-size: 1.35rem; font-weight: 700; margin-top: 0.15rem; }
    .step-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.75rem 0 1rem 0; }
    .step-chip {
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.82rem;
        border: 1px solid #cbd5e1;
        background: #f8fafc;
        color: #475569;
    }
    .step-chip.done { background: #dcfce7; border-color: #86efac; color: #166534; }
    .step-chip.active { background: #dbeafe; border-color: #93c5fd; color: #1d4ed8; }
    .step-chip.waiting { background: #fff7ed; border-color: #fdba74; color: #9a3412; }
    .panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    .log-line {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.82rem;
        color: #334155;
        padding: 0.2rem 0;
        border-bottom: 1px dashed #eef2f7;
    }
    div[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
</style>
"""


def _init_session() -> None:
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "running" not in st.session_state:
        st.session_state.running = False


def _pipeline_chips_html(status: dict[str, str]) -> str:
    chips = []
    for key, label, _ in PIPELINE_STEPS:
        state = status.get(key, "pending")
        css = "step-chip"
        if state == "done":
            css += " done"
        elif state == "waiting":
            css += " waiting"
        elif state == "active":
            css += " active"
        chips.append(f'<span class="{css}">{label}</span>')
    return f'<div class="step-row">{"".join(chips)}</div>'


def _step_status(logs: list[str], awaiting_human: bool, has_output: bool) -> dict[str, str]:
    text = " ".join(logs).lower()
    status = {key: "pending" for key, _, _ in PIPELINE_STEPS}

    if "plan:" in text:
        status["plan"] = "done"
    if "research" in text and "queries" in text:
        status["research"] = "done"
    if "summarize:" in text:
        status["summarize"] = "done"
    if "write:" in text:
        status["write"] = "done"
    if "critique:" in text:
        status["critique"] = "done"
    if "simulated send" in text or has_output:
        status["output"] = "done"

    if awaiting_human:
        status["output"] = "waiting"

    return status


def _live_status(completed_nodes: set[str], active_node: str | None) -> dict[str, str]:
    status = {key: "pending" for key, _, _ in PIPELINE_STEPS}
    order = [key for key, _, _ in PIPELINE_STEPS]
    for node in completed_nodes:
        step = NODE_TO_STEP.get(node)
        if step:
            status[step] = "done"
    if active_node:
        step = NODE_TO_STEP.get(active_node)
        if step and status.get(step) != "done":
            status[step] = "active"
        # Mark earlier steps done when later ones start
        if step in order:
            idx = order.index(step)
            for earlier in order[:idx]:
                if status[earlier] == "pending":
                    status[earlier] = "done"
    return status


def _render_pipeline(status: dict[str, str]) -> None:
    st.markdown(_pipeline_chips_html(status), unsafe_allow_html=True)


def _render_metrics(result) -> None:
    score = result.critique.get("score") if result.critique else None
    cols = st.columns(4)
    metrics = [
        ("Mode", result.mode.replace("_", " ").title()),
        ("Articles", str(len(result.articles))),
        ("Critique score", f"{score:.1f}/10" if isinstance(score, (int, float)) else "—"),
        ("Status", "Awaiting you" if result.awaiting_human else ("Sent" if result.output_path else "Draft")),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )


def _render_sidebar() -> tuple[str, str | None]:
    with st.sidebar:
        st.markdown("### Controls")
        goal = st.text_area(
            "Newsletter goal",
            value=DEFAULT_GOAL,
            height=120,
            help="Plain-English instruction for the agent.",
        )
        hitl = st.toggle(
            "Human-in-the-Loop",
            value=False,
            help="Pause for your approval before simulated send.",
        )
        mode = "hitl" if hitl else "autonomous"

        st.markdown("---")
        st.markdown("**Runtime**")
        llm_ready = bool(get_groq_api_key())
        backend = get_search_backend()
        st.markdown(f"- LLM: {'Connected' if llm_ready else 'Missing key'}")
        st.markdown(f"- Model: `{get_groq_model()}`")
        if backend == "tavily":
            st.markdown("- Search: **Tavily**")
        else:
            st.markdown("- Search: DuckDuckGo (add `TAVILY_API_KEY` for Tavily)")

        st.markdown("---")
        run = st.button("Run Agent", type="primary", use_container_width=True)
        clear = st.button("Clear results", use_container_width=True)

        if hitl:
            st.info("HITL pauses after critique for approve / revise.")

    if clear:
        st.session_state.last_result = None
        st.session_state.running = False
        st.rerun()

    return goal, mode if run else None


def _render_hitl(result):
    st.markdown("#### Human review")
    st.warning("Draft ready. Approve to simulate send, or request a revision.")
    feedback = st.text_area(
        "Revision feedback",
        placeholder="e.g. Shorten the intro and highlight open-source agent frameworks.",
        key="hitl_feedback",
        height=90,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Approve & Send", type="primary", use_container_width=True):
            with st.spinner("Simulating send..."):
                updated = resume_newsletter_agent(result.state, "approve")
            st.session_state.last_result = updated
            st.rerun()
    with c2:
        if st.button("Request Revision", use_container_width=True):
            with st.spinner("Revising with your feedback..."):
                updated = resume_newsletter_agent(result.state, "revise", feedback=feedback)
            st.session_state.last_result = updated
            st.rerun()


def _run_with_live_stream(goal: str, mode: str):
    """Run the agent and stream each LangGraph step into the UI."""
    st.session_state.running = True
    pipeline_slot = st.empty()
    log_slot = st.empty()
    completed: set[str] = set()
    live_logs: list[str] = []
    result = None

    pipeline_slot.markdown(
        _pipeline_chips_html(_live_status(completed, None)),
        unsafe_allow_html=True,
    )

    with st.status("Running agent pipeline...", expanded=True) as status_box:
        try:
            for event in stream_newsletter_agent(goal, mode=mode):
                if event.done and event.result is not None:
                    result = event.result
                    status_box.update(label="Agent run complete", state="complete")
                    break

                status_box.write(f"**{event.label}** — {event.message}")
                live_logs = event.logs
                completed.add(event.node)
                pipeline_slot.markdown(
                    _pipeline_chips_html(_live_status(completed, event.node)),
                    unsafe_allow_html=True,
                )
                # Show last few log lines live
                recent = live_logs[-6:] if live_logs else []
                if recent:
                    log_slot.code("\n".join(recent), language=None)

                status_box.update(label=f"{event.label}...", state="running")
        except Exception as exc:  # noqa: BLE001
            status_box.update(label="Agent run failed", state="error")
            st.error(str(exc))
            st.session_state.running = False
            return None

    st.session_state.running = False
    return result


def main() -> None:
    st.set_page_config(
        page_title="Newsletter Agent",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _init_session()

    st.markdown(
        """
        <div class="hero">
            <h1>Newsletter Agent</h1>
            <p>
                Autonomous LangGraph agent that plans, researches news for your goal,
                writes a newsletter, self-critiques, and simulates delivery — with live step streaming.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    goal, run_mode = _render_sidebar()

    if run_mode is not None:
        result = _run_with_live_stream(goal, run_mode)
        if result is not None:
            st.session_state.last_result = result

    result = st.session_state.get("last_result")
    if not result:
        st.markdown(
            """
            <div class="panel">
                <strong>Get started</strong><br/>
                Set your goal in the sidebar, choose Autonomous or Human-in-the-Loop,
                then click <em>Run Agent</em>. Watch each pipeline step stream live.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Pipeline")
        _render_pipeline({key: "pending" for key, _, _ in PIPELINE_STEPS})
        return

    status = _step_status(result.logs, result.awaiting_human, bool(result.output_path))
    st.markdown("#### Pipeline progress")
    _render_pipeline(status)
    _render_metrics(result)

    if result.output_path:
        st.success(f"Simulated send complete -> `{result.output_path}`")
    elif result.awaiting_human:
        st.info("Waiting for human approval before send.")

    if result.awaiting_human:
        _render_hitl(result)

    st.markdown(f"### {result.subject or 'Untitled draft'}")

    if result.critique:
        approved = result.critique.get("approved")
        feedback = result.critique.get("feedback") or ""
        st.caption(
            f"Self-critique: score {result.critique.get('score', '?')} | "
            f"approved={approved}"
        )
        if feedback:
            with st.expander("Critique notes", expanded=False):
                st.write(feedback)

    tab_preview, tab_articles, tab_markdown, tab_logs = st.tabs(
        ["Preview", "Articles", "Markdown", "Logs"]
    )

    with tab_preview:
        if result.html:
            components.html(result.html, height=760, scrolling=True)
            st.download_button(
                "Download HTML",
                data=result.html,
                file_name="newsletter.html",
                mime="text/html",
                use_container_width=False,
            )
            if result.output_path and Path(result.output_path).exists():
                st.caption(f"Saved file: {result.output_path}")
        else:
            st.info("No HTML preview yet.")

    with tab_articles:
        if not result.articles:
            st.info("No articles selected.")
        for i, article in enumerate(result.articles, start=1):
            title = article.get("title") or "Untitled"
            url = article.get("url") or ""
            summary = article.get("summary") or article.get("snippet") or ""
            relevance = article.get("relevance") or ""
            with st.container(border=True):
                if url:
                    st.markdown(f"**{i}. [{title}]({url})**")
                else:
                    st.markdown(f"**{i}. {title}**")
                if relevance:
                    st.caption(relevance)
                if summary:
                    st.write(summary)

    with tab_markdown:
        if result.markdown:
            st.code(result.markdown, language="markdown")
            st.download_button(
                "Download Markdown",
                data=result.markdown,
                file_name="newsletter.md",
                mime="text/markdown",
            )
        else:
            st.info("No markdown body generated.")

    with tab_logs:
        if result.logs:
            for line in result.logs:
                st.markdown(f'<div class="log-line">{line}</div>', unsafe_allow_html=True)
        else:
            st.info("No logs yet.")


if __name__ == "__main__":
    main()
