# Copilot Instructions

## Build & Test

```bash
# Activate the virtualenv first
source .venv/bin/activate

# Run all tests
python -m pytest test_writing_tool.py -v

# Run a single test class or test
python -m pytest test_writing_tool.py::TestRewrite -v
python -m pytest test_writing_tool.py::TestRewrite::test_returns_model_response -v
```

There is no linter or CI pipeline configured.

## Architecture

Single-file macOS menu bar app (`writing_tool.py`) built with **rumps**. The app reads clipboard text, sends it to an LLM for rewriting, presents variants in a native macOS picker (osascript `choose from list`), and copies the chosen result back to the clipboard.

### LLM backend dispatch

`_call_model()` routes to one of three backends based on the `BACKEND` env var: `_call_ollama` (default), `_call_openai`, or `_call_anthropic`. The OpenAI and Anthropic SDKs are lazy-imported inside their respective functions to avoid requiring API keys when using Ollama.

### Rewrite flow

`rewrite_multiple()` asks the LLM for N numbered variants in a single call, then parses them with a regex. If parsing fails (wrong count), it falls back to returning the raw response as a single option. The user picks from these via `pick_result()`.

### Anki integration

After a rewrite is chosen, a background thread calls the LLM again to generate an explanation of what changed, then posts the original/corrected pair as a flashcard to AnkiConnect. The "Learn This" mode creates vocabulary cards instead. The "Practice Weak Spots" feature fetches all cards from the Writing Errors deck, sends them to the LLM to identify recurring error patterns, generates novel sentence-correction exercises, and creates them in a separate exercise deck. Anki calls are fire-and-forget — failures are logged but never shown to the user.

### Threading model

Menu callbacks acquire `self.lock` to set a `processing` flag, then spawn a daemon thread for the LLM work. The flag prevents concurrent requests. A rumps Timer drives a spinner animation in the menu bar while processing.

## Conventions

- **All configuration is via environment variables** — defaults are set at module level, with `start.sh` as the canonical place to document and set them.
- **Modes** are defined in the `MODES` dict. Each entry needs `instruction` (str), `label` (str), and `temperature` (float). New modes appear in the menu bar automatically.
- **Tests mock at the HTTP/function boundary** (`requests.post` or `_call_openai`/`_call_anthropic`) — never call a real LLM. The `_make_app()` helper creates a `WritingToolApp` without starting the rumps event loop.
- **macOS-only**: the app depends on `osascript` for notifications and the picker dialog, and `rumps` for the menu bar. There are no cross-platform abstractions.
- **Python type hints** use the `str | None` union syntax (Python 3.10+).
