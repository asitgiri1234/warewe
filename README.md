# Newsletter Agent

Mini autonomous agent that researches the latest **AI agent** news, drafts a weekly newsletter, and simulates sending it (saves HTML + subject).

Built with **LangGraph**, **Groq** (LLM — key pending), **Tavily / DuckDuckGo** search, and **Streamlit**.

## Status (this commit)

| Piece | Status |
|-------|--------|
| Project scaffold | Done |
| Web search tool (Tavily or DuckDuckGo) | Done |
| HTML newsletter builder (Jinja2) | Done |
| Pre-LLM graph (`research → output`) | Done |
| Streamlit UI shell + mode toggle | Done |
| Groq plan / summarize / write / critique | **Next** (after `GROQ_API_KEY`) |
| Full HITL approve/revise gate | **Next** |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # then add GROQ_API_KEY when ready
```

### Environment

| Variable | Required | Notes |
|----------|----------|-------|
| `GROQ_API_KEY` | For LLM nodes | Get from [console.groq.com](https://console.groq.com/) |
| `GROQ_MODEL` | No | Default `llama-3.3-70b-versatile` |
| `TAVILY_API_KEY` | No | If unset, search uses DuckDuckGo |

## Run (pre-LLM demo)

Works **without** a Groq key — researches public news and writes a draft HTML file:

```bash
streamlit run app.py
```

Or from Python:

```python
from newsletter_agent import run_newsletter_agent

result = run_newsletter_agent(
    "Create a weekly newsletter on latest AI agent news and send it to our subscribers."
)
print(result.subject)
print(result.output_path)
print(result.logs)
```

## Architecture (target)

```
goal → plan → research → summarize → write → critique ⇄ write
                                              ↓
                                    HITL? → human approve
                                              ↓
                                         simulate send (HTML file)
```

Entrypoint: `run_newsletter_agent(goal, mode="autonomous"|"hitl")`.

## Layout

```
app.py
newsletter_agent/
  config.py          # Groq env helpers
  graph.py           # LangGraph + run_newsletter_agent()
  state.py
  nodes/             # plan, research, summarize, write, critique, output
  tools/
    search.py
    html_builder.py
  templates/
    newsletter.html.j2
output/              # generated newsletters (gitignored)
```

## License

MIT
