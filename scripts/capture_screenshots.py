"""Capture README screenshots of the Streamlit UI and sample newsletter."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
APP = "http://localhost:8503"
NEWSLETTER = (OUT / "sample-newsletter.html").resolve().as_uri()


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto(APP, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "01-ui-home.png"), full_page=True)

        # Run agent for a results screenshot (live steps visible during run)
        goal = page.locator("textarea").first
        goal.fill(
            "Create a weekly newsletter on latest AI agent news and send it to our subscribers."
        )
        page.get_by_role("button", name="Run Agent").click()
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT / "02-live-streaming.png"), full_page=True)

        # Wait until run finishes (status complete / subject appears)
        page.get_by_text("Simulated send complete").wait_for(timeout=180000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT / "03-results.png"), full_page=True)

        # Articles tab
        page.get_by_role("tab", name="Articles").click()
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "04-articles.png"), full_page=True)

        # Newsletter HTML preview standalone
        page2 = browser.new_page(viewport={"width": 900, "height": 1200})
        page2.goto(NEWSLETTER, wait_until="networkidle")
        page2.wait_for_timeout(1000)
        page2.screenshot(path=str(OUT / "05-newsletter-html.png"), full_page=True)

        browser.close()
    print(f"Saved screenshots to {OUT}")


if __name__ == "__main__":
    main()
