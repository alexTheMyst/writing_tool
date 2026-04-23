# Writing Tool

License: MIT


A macOS menu bar writing assistant powered by [Ollama](https://ollama.com). Copy text, click the ✎ icon, pick a mode — three variants appear in a native picker and the one you choose lands in your clipboard, ready to paste.

Runs entirely on your own hardware. No data leaves your network.

## How it works

1. Copy any text
2. Click **✎** in the menu bar
3. Pick a rewrite mode
4. Choose one of the three variants (or press Escape to cancel)
5. Paste

## Modes

| Mode | What it does |
|------|-------------|
| Make Casual | Natural Slack voice — contractions, fragments, no corporate tone, emojis where fitting |
| Simplify | Shorter — cut every word that doesn't add meaning |
| Soften Tone | Collaborative not cold, without hollow affirmations |
| Make Direct | Lead with the ask, cut the build-up |
| Sound Native | Fixes unnatural collocations, preposition pairings, and word choices so the text sounds like a native speaker wrote it |
| Custom… | Enter any rewrite instruction on the fly |
| Learn This | Explains phrasal verbs, idioms, and nuances in the copied text (saves to Anki) |
| Estimate Level | Rates the copied text on the CEFR scale (A1–C2), explains why, and gives concrete suggestions to reach the next level |
| Check Register | Checks whether the tone matches a chosen audience (Slack peer, manager, email client, etc.) and suggests specific swaps |
| Daily Prompt | Shows a short writing scenario from a 25-prompt bank; you type a response, the LLM corrects it and saves an Anki card |
| Practice Weak Spots | Analyzes your Writing Errors deck, identifies recurring mistake patterns, and generates sentence-correction exercises |

## Setup

### 1. Install Ollama

**Local** (runs on your Mac):

```bash
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```

**Remote server** (Linux / Fedora):

```bash
# On the server
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# Make Ollama listen on all interfaces (default is localhost only)
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama

# Open the firewall (Fedora)
sudo firewall-cmd --zone=FedoraServer --add-port=11434/tcp --permanent
sudo firewall-cmd --reload
```


### 2. Install the writing tool

```bash
cd writing-tool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure and run

Set your Ollama URL in your shell profile (`~/.zshrc` or `~/.bash_profile`):

```bash
export OLLAMA_HOST="http://YOUR_SERVER_IP:11434"   # remote
# or leave unset for http://localhost:11434 (local default)
```

Then run:

```bash
chmod +x start.sh
./start.sh
```

A **✎** icon appears in the menu bar. No Accessibility permission required.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND` | `ollama` | LLM backend: `ollama`, `openai`, or `anthropic` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Ollama model name |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key — required when `BACKEND=openai` |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key — required when `BACKEND=anthropic` |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model name |
| `ANKI_ENABLED` | `1` | Set to `0` to disable Anki integration |
| `ANKI_URL` | `http://127.0.0.1:8765` | AnkiConnect plugin URL |
| `ANKI_DECK` | `Writing Errors` | Deck for grammar/style correction cards |
| `ANKI_VOCAB_DECK` | `English Vocabulary` | Deck for vocabulary/idiom cards (Learn This) |
| `ANKI_EXERCISE_DECK` | `Writing Exercises` | Deck for generated practice exercises (Practice Weak Spots) |
| `ANKI_REGISTER_DECK` | `Register Notes` | Deck for register/audience swap cards (Check Register) |
| `ANKI_TIMEOUT` | `5` | AnkiConnect request timeout in seconds |
| `DAILY_PROMPT_ENABLED` | `1` | Set to `0` to disable the automatic daily writing prompt |
| `DAILY_PROMPT_HOUR` | `10` | Local time hour (0–23) when the daily prompt fires automatically |

## Anki integration

When you pick a rewritten variant, the tool automatically creates an [Anki](https://apps.ankiweb.net) flashcard showing the original text on the front and the corrected version (with an AI-generated explanation of the changes) on the back. The **Learn This** menu item creates a vocabulary card listing any phrasal verbs, idioms, and colloquialisms found in the copied text.

The **Check Register** menu item saves any suggested swaps as cards in the Register Notes deck (`ANKI_REGISTER_DECK`), so you can review formal/casual substitutions later.

The **Daily Prompt** menu item shows a writing scenario and corrects your response. If any corrections were needed, it saves a card to the Writing Errors deck.

The **Practice Weak Spots** menu item analyzes all cards in your Writing Errors deck, uses the LLM to identify recurring error patterns (e.g. missing articles, wrong prepositions), and generates sentence-correction exercise cards in a separate "Writing Exercises" deck. The number of exercises scales with the number of patterns found.

**Requirement:** the [AnkiConnect](https://ankiweb.net/shared/info/2055492159) add-on must be installed in Anki (add-on code `2055492159`). Anki must be running for cards to be created — if it isn't, the tool silently skips card creation.

To disable the feature entirely, set `ANKI_ENABLED=0` in your shell profile.

## Daily Prompt

The tool fires a writing scenario automatically once per day at or after `DAILY_PROMPT_HOUR` (default 10 AM). A native macOS dialog asks you to type a 2–3 sentence response; the LLM corrects it and shows what changed. If your text needed corrections, a card is saved to the Writing Errors deck.

- Trigger manually at any time via **Daily Prompt** in the menu bar
- Disable automatic firing by setting `DAILY_PROMPT_ENABLED=0` in your shell profile
- Change the fire time with `DAILY_PROMPT_HOUR=<0-23>`
- Progress is logged to `~/.writing_tool_progress.jsonl` (one JSON line per CEFR check) and daily completion state to `~/.writing_tool_daily_state`

## Adding a mode

Edit `writing_tool.py` — add an entry to `MODES`:

```python
MODES["formal"] = {
    "instruction": "Rewrite this text to sound professional and formal.",
    "label": "Make Formal",
    "temperature": 0.3,
}
```

The new mode appears in the menu bar automatically on next launch.

## Autostart

`com.local.writing-tool.plist` is a launchd agent. Update the path inside it to point to your `start.sh`, then:

```bash
cp com.local.writing-tool.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.local.writing-tool.plist
```

Logs: `/tmp/writing-tool.log` and `/tmp/writing-tool.err`

## Model recommendations

| Model | Size | Speed (CPU) | Quality |
|-------|------|-------------|---------|
| `phi3:mini` | 2.3 GB | Fast (~3–5 s) | Good |
| `llama3.1:8b` | 4.7 GB | Medium (~30–60 s) | Very good |
| `mistral:7b` | 4.1 GB | Medium (~30–60 s) | Very good |
| `qwen3.5:9b` *(default)* | ~6 GB | Medium (~30–60 s) | Excellent |

GPU inference is much faster — `qwen3.5:9b` runs in ~5 s on a mid-range GPU.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ✎ not in menu bar | Run `./start.sh` — check the terminal for errors |
| "Clipboard is empty" | **Cmd+C** the text before clicking the menu |
| Connection refused | Check `OLLAMA_HOST` in `start.sh` and that Ollama is running |
| Picker shows garbled options | Ollama returned unexpected format — check logs in the terminal |
| Timeout | Increase `OLLAMA_TIMEOUT` in `start.sh`, or try `phi3:mini` |
