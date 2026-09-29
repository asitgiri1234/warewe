"""Streamlit UI for the Newsletter Agent (Groq + LangGraph)."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from newsletter_agent.graph import resume_newsletter_agent, run_newsletter_agent

DEFAULT_GOAL = (
    "Create a weekly newsletter on latest AI agent news and send it to our subscribers."
)


def main() -> None:
    st.set_page_config(
        page_title="Newsletter Agent",
        page_icon="📰",
        layout="wide",
    )
    st.title("Newsletter Agent")
    st.caption("Autonomous LangGraph agent · Groq · Search + HTML tools · Self-critique")

    goal = st.text_area("Goal", value=DEFAULT_GOAL, height=100)
    hitl = st.toggle("Human-in-the-Loop", value=False)
    mode = "hitl" if hitl else "autonomous"

    if hitl:
        st.info(
            "HITL mode: after research, writing, and self-critique, "
            "you approve or request changes before the simulated send."
        )

    col_run, col_status = st.columns([1, 3])
    with col_run:
        run_clicked = st.button("Run Agent", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running newsletter agent (plan -> research -> write -> critique)…"):
            try:
                result = run_newsletter_agent(goal, mode=mode)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Agent failed: {exc}")
                return
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        st.markdown(
            "Click **Run Agent** to research AI-agent news and generate a newsletter."
        )
        return

    # HITL controls
    if result.awaiting_human:
        st.warning("Draft ready — approve to simulate send, or request a revision.")
        feedback = st.text_area(
            "Revision feedback (optional)",
            placeholder="e.g. Make the intro shorter and highlight open-source frameworks.",
            key="hitl_feedback",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Approve & Send", type="primary", use_container_width=True):
                with st.spinner("Simulating send…"):
                    result = resume_newsletter_agent(result.state, "approve")
                st.session_state["last_result"] = result
                st.rerun()
        with c2:
            if st.button("Request Revision", use_container_width=True):
                with st.spinner("Revising with your feedback…"):
                    result = resume_newsletter_agent(
                        result.state, "revise", feedback=feedback
                    )
                st.session_state["last_result"] = result
                st.rerun()

    with col_status:
        if result.output_path:
            st.success(f"Sent (simulated) -> `{result.output_path}`")
        elif result.awaiting_human:
            st.info("Waiting for your approval")
        else:
            st.success("Draft ready")

    st.subheader("Subject")
    st.write(result.subject or "(untitled draft)")

    if result.critique:
        st.subheader("Self-critique")
        st.write(
            f"Score: {result.critique.get('score', '?')} | "
            f"Approved: {result.critique.get('approved', '?')}"
        )
        if result.critique.get("feedback"):
            st.caption(result.critique["feedback"])

    st.subheader("Step logs")
    for line in result.logs:
        st.text(f"* {line}")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Articles ({len(result.articles)})")
        for i, article in enumerate(result.articles[:12], start=1):
            title = article.get("title") or "Untitled"
            url = article.get("url") or ""
            summary = article.get("summary") or article.get("snippet") or ""
            if url:
                st.markdown(f"**{i}. [{title}]({url})**")
            else:
                st.markdown(f"**{i}. {title}**")
            if summary:
                st.caption(summary[:320] + ("…" if len(summary) > 320 else ""))

    with right:
        st.subheader("HTML preview")
        if result.html:
            components.html(result.html, height=720, scrolling=True)
            st.download_button(
                "Download HTML",
                data=result.html,
                file_name="newsletter.html",
                mime="text/html",
            )
        else:
            st.write("No HTML produced.")


if __name__ == "__main__":
    main()
