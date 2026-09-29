# Newsletter Agent

Mini autonomous agent that researches the latest **AI agent** news, drafts a weekly newsletter, self-critiques, and simulates sending it (saves HTML + subject).

Built with **LangGraph**, **Groq**, **Tavily / DuckDuckGo** search, and **Streamlit**.

## Pipeline

```
goal -> plan -> research -> summarize -> write -> critique <-> write
                                              |
                                    HITL? -> human approve/revise
                                              |
                                         simulate send (HTML file)
```

Entrypoint: `run_newsletter_agent(goal, mode="autonomous"|"hitl")`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
copy .env.example .env          # set GROQ_API_KEY
```

### Environment

| Variable | Required | Notes |
|----------|----------|-------|
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com/) |
| `GROQ_MODEL` | No | Default `openai/gpt-oss-120b` |
| `TAVILY_API_KEY` | No | If unset, search uses DuckDuckGo |

`.env` is gitignored — never commit keys.

## Run

```bash
streamlit run app.py
```

Or:

```python
from newsletter_agent import run_newsletter_agent

result = run_newsletter_agent(
    "Create a weekly newsletter on latest AI agent news and send it to our subscribers."
)
print(result.subject)
print(result.output_path)
print(result.critique)
```

## Layout

```
app.py
newsletter_agent/
  config.py          # Groq env helpers
  llm.py             # chat_text / chat_json
  graph.py           # LangGraph + run/resume
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
