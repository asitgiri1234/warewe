"""Jinja2 HTML newsletter renderer (no LLM required)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def build_html_newsletter(
    subject: str,
    intro: str,
    articles: list[dict[str, Any]],
    outro: str = "You're receiving this because you subscribed to our AI Agents digest.",
    week_label: str | None = None,
) -> str:
    """
    Render a clean HTML newsletter from structured content.

    `articles` items should include: title, url, summary (or snippet),
    and optionally source / relevance.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("newsletter.html.j2")
    now = datetime.now(timezone.utc)
    return template.render(
        subject=subject,
        intro=intro,
        articles=articles,
        outro=outro,
        week_label=week_label or now.strftime("Week of %B %d, %Y"),
        generated_at=now.strftime("%Y-%m-%d %H:%M UTC"),
    )


def save_newsletter(html: str, output_dir: str | Path = "output", stem: str | None = None) -> Path:
    """Write HTML to disk and return the path (simulates send)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not stem:
        stem = datetime.now(timezone.utc).strftime("newsletter_%Y%m%d_%H%M%S")
    path = out / f"{stem}.html"
    path.write_text(html, encoding="utf-8")
    return path
