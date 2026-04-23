#!/bin/bash
# ──────────────────────────────────────────────────────────
# Writing Tool Launcher
# ──────────────────────────────────────────────────────────
# Configure by setting environment variables before running,
# e.g. in your ~/.zshrc or ~/.bash_profile:
#
#   export OLLAMA_HOST="http://192.168.1.x:11434"
#   export OLLAMA_MODEL="qwen3.5:9b"
#
# Defaults are applied below if variables are not already set.

export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:9b}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-120}"

export ANKI_ENABLED="${ANKI_ENABLED:-1}"
export ANKI_URL="${ANKI_URL:-http://127.0.0.1:8765}"
export ANKI_DECK="${ANKI_DECK:-Writing Errors}"
export ANKI_VOCAB_DECK="${ANKI_VOCAB_DECK:-English Vocabulary}"
export ANKI_EXERCISE_DECK="${ANKI_EXERCISE_DECK:-Writing Exercises}"
export ANKI_REGISTER_DECK="${ANKI_REGISTER_DECK:-Register Notes}"
export ANKI_TIMEOUT="${ANKI_TIMEOUT:-5}"

# Daily writing prompt (set DAILY_PROMPT_ENABLED=0 to disable).
# DAILY_PROMPT_HOUR is local time (0–23); the timer fires once on/after this hour each day.
export DAILY_PROMPT_ENABLED="${DAILY_PROMPT_ENABLED:-1}"
export DAILY_PROMPT_HOUR="${DAILY_PROMPT_HOUR:-10}"

# ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
python "$SCRIPT_DIR/writing_tool.py"
