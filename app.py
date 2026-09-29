"""
Streamlit UI for the Newsletter Agent.

Pre-LLM: run research + HTML draft preview.
HITL / full critique loop activates after Groq integration.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from newsletter_agent.graph import run_newsletter_agent

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
    st.caption(
        "Autonomous LangGraph agent · Groq LLM (pending key) · "
        "Research + HTML tools ready now"
    )

    goal = st.text_area("Goal", value=DEFAULT_GOAL, height=100)
    hitl = st.toggle("Human-in-the-Loop", value=False)
    mode = "hitl" if hitl else "autonomous"

    if hitl:
        st.info(
            "HITL pause/approve will activate after Groq integration. "
            "For now the agent runs research → draft HTML only."
        )

    col_run, col_status = st.columns([1, 3])
    with col_run:
        run_clicked = st.button("Run Agent", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running newsletter agent…"):
            try:
                result = run_newsletter_agent(goal, mode=mode)
            except NotImplementedError as exc:
                st.warning(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 — surface cleanly in UI
                st.error(f"Agent failed: {exc}")
                return

        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        st.markdown(
            "Click **Run Agent** to research AI-agent news and generate a draft HTML newsletter."
        )
        return

    with col_status:
        st.success(
            f"Draft saved"
            + (f" -> `{result.output_path}`" if result.output_path else "")
        )

    st.subheader("Subject")
    st.write(result.subject or "(untitled draft)")

    st.subheader("Step logs")
    for line in result.logs:
        st.text(f"• {line}")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Articles ({len(result.articles)})")
        for i, article in enumerate(result.articles[:12], start=1):
            title = article.get("title") or "Untitled"
            url = article.get("url") or ""
            snippet = article.get("snippet") or ""
            if url:
                st.markdown(f"**{i}. [{title}]({url})**")
            else:
                st.markdown(f"**{i}. {title}**")
            if snippet:
                st.caption(snippet[:280] + ("…" if len(snippet) > 280 else ""))

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
