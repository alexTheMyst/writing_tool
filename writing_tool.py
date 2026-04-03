#!/usr/bin/env python3
"""
Local Writing Tool — macOS menu bar app powered by Ollama.

Workflow: Copy text → click menu bar icon (✎) → select mode → paste improved text.
"""

import logging
import os
import re
import subprocess
import threading
import sys

import requests
import pyperclip
import rumps


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Set OLLAMA_HOST env var to point at your Fedora server's Tailscale IP, e.g.:
#   export OLLAMA_HOST=http://100.x.y.z:11434
# Falls back to localhost if not set.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

# Timeout in seconds — CPU inference can be slow, so be generous
TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# AnkiConnect — set ANKI_ENABLED=0 to disable
ANKI_ENABLED = os.environ.get("ANKI_ENABLED", "1") == "1"
ANKI_URL = os.environ.get("ANKI_URL", "http://127.0.0.1:8765")
ANKI_DECK = os.environ.get("ANKI_DECK", "Writing Errors")
ANKI_VOCAB_DECK = os.environ.get("ANKI_VOCAB_DECK", "English Vocabulary")
ANKI_TIMEOUT = int(os.environ.get("ANKI_TIMEOUT", "5"))

SYSTEM_PROMPT = (
    "You are a writing assistant helping someone communicate naturally in Slack. "
    "The input is written by a non-native English speaker and may contain spelling mistakes, typos, "
    "wrong verb tenses, or missing articles — always correct these as part of your rewrite. "
    "When a word looks misspelled, choose the most common literal interpretation "
    "(e.g. 'sings' → 'things', 'brake' → 'break'). "
    "Never infer context, domain, or details that are not explicitly in the input. "
    "Rewrite text according to the instruction given. "
    "Return ONLY the rewritten text — no explanations, no quotes, no markdown, no preamble. "
    "Never add filler phrases like 'Hope this helps!', 'Just wanted to', 'Please let me know', "
    "or 'Feel free to reach out'. "
    "Never start with a greeting or sign-off unless the original has one. "
    "If the input is already good, return it with minimal changes."
)


MODES = {
    "casual": {
        "instruction": (
            "Rewrite this for Slack. Sound like a real person typing to a colleague, not a corporate email. "
            "Use contractions (it's, don't, we'll). Fragments and short sentences are fine. "
            "Cut filler and throat-clearing. Start directly with the point. "
            "Warm but not performative — skip the exclamation points unless the original is genuinely excited. "
            "Add a relevant emoji or two where it feels natural (e.g. at the end of a sentence, not mid-word). "
            "Don't force emojis on every sentence — one or two per message is plenty."
        ),
        "label": "Make Casual",
        "temperature": 0.6,
    },
    "simplify": {
        "instruction": (
            "Shorten this for Slack. Cut every word that doesn't add meaning. "
            "Break any sentence over 20 words into two. "
            "Remove hedging phrases like 'in order to', 'it is worth noting that', 'as mentioned previously'. "
            "Keep the core ask or information. Aim for under half the original length if possible."
        ),
        "label": "Simplify",
        "temperature": 0.3,
    },
    "soften": {
        "instruction": (
            "Soften the tone of this message for Slack without changing the meaning or becoming sycophantic. "
            "Make it sound collaborative rather than demanding or cold. "
            "Don't add hollow affirmations or filler phrases. "
            "A small word change or reorder is often enough — don't over-edit."
        ),
        "label": "Soften Tone",
        "temperature": 0.6,
    },
    "direct": {
        "instruction": (
            "Rewrite this to lead with the actual point or ask. "
            "Cut the build-up and get to what matters in the first sentence. "
            "Keep it friendly but don't bury the request. "
            "Don't soften so much that the ask becomes unclear."
        ),
        "label": "Make Direct",
        "temperature": 0.3,
    },
}


# ──────────────────────────────────────────────────────────────
# macOS notification helper
# ──────────────────────────────────────────────────────────────

def notify(title: str, message: str) -> None:
    """Send a macOS notification via osascript."""
    safe_msg = message.replace("\\", "\\\\").replace('"', '\\"')
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{safe_msg}" with title "{safe_title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def pick_result(options: list) -> str | None:
    """Show a native macOS list picker and return the chosen full text, or None if cancelled."""
    labels = [
        f"{i + 1}. {opt[:80]}{'…' if len(opt) > 80 else ''}"
        for i, opt in enumerate(options)
    ]
    def esc(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')
    as_list = "{" + ", ".join(f'"{esc(l)}"' for l in labels) + "}"
    script = (
        f'choose from list {as_list} '
        f'with prompt "Pick a version:" '
        f'without multiple selections allowed'
    )
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    chosen = result.stdout.strip()
    if not chosen or chosen == "false":
        return None
    for i, label in enumerate(labels):
        if chosen == label:
            return options[i]
    return None



# ──────────────────────────────────────────────────────────────
# Ollama interaction
# ──────────────────────────────────────────────────────────────

def rewrite(text: str, instruction: str, temperature: float = 0.3) -> str:
    """Send text to Ollama and return the rewritten version."""
    prompt = f"{instruction}\n\nText to rewrite:\n{text}"
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
        },
    }
    try:
        logging.debug("POST %s (timeout=%ds)", OLLAMA_URL, TIMEOUT)
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        logging.debug("Ollama responded: %r", result[:120])
        return result
    except requests.RequestException as e:
        logging.error("Ollama request failed: %s", e)
        notify("Writing Tool — Error", str(e))
        return ""


def rewrite_multiple(text: str, instruction: str, n: int, temperature: float) -> list:
    """Ask Ollama for n numbered variants. Returns a list of strings (may be shorter than n on parse failure)."""
    prompt = (
        f"{instruction}\n\n"
        f"Provide exactly {n} different rewrites, numbered 1. 2. 3. "
        f"Put a blank line between each. Output only the numbered rewrites, nothing else.\n\n"
        f"Text:\n{text}"
    )
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }
    try:
        logging.debug("POST %s (timeout=%ds, n=%d)", OLLAMA_URL, TIMEOUT, n)
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        logging.debug("Ollama responded: %r", raw[:300])
        variants = re.findall(r'(?m)^\d+\.\s+([\s\S]+?)(?=\n\s*\d+\.|\s*$)', raw)
        variants = [v.strip() for v in variants if v.strip()]
        if len(variants) != n:
            logging.warning("Expected %d variants, parsed %d — falling back to raw", n, len(variants))
            return [raw] if raw else []
        return variants
    except requests.RequestException as e:
        logging.error("Ollama request failed: %s", e)
        notify("Writing Tool — Error", str(e))
        return []


# ──────────────────────────────────────────────────────────────
# Anki integration
# ──────────────────────────────────────────────────────────────

_ANKI_EXPLANATION_PROMPT = """\
You are an English tutor helping a non-native speaker understand their mistakes.

Compare the original text with the corrected version below.
List every correction made and explain WHY it was changed.
Group corrections by category: Spelling, Grammar, Tone/Style.
Skip any category that has no corrections.
Be concise — one short bullet per correction.
Output plain text only — no markdown, no headers, no numbering.
Use a dash (-) for each bullet.

Original:
{original}

Corrected:
{corrected}\
"""

_NUANCE_PROMPT = """\
You are an English tutor. Find every phrasal verb and idiomatic expression in the text below.

If none are found, output exactly: NONE

For each one output a single bullet:
- <exact phrase as it appears> — <meaning>

Text:
{text}\
"""


def generate_explanation(original: str, corrected: str) -> str:
    """Ask Ollama to explain the differences between original and corrected text."""
    prompt = _ANKI_EXPLANATION_PROMPT.format(original=original, corrected=corrected)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        logging.debug("Anki explanation raw response: %r", raw[:300])
        return raw
    except requests.RequestException as e:
        logging.warning("Explanation request failed: %s", e)
        return ""


def generate_nuance_explanation(text: str) -> str:
    """Ask Ollama to explain phrasal verbs, idioms, slang, and other nuances in text."""
    prompt = _NUANCE_PROMPT.format(text=text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        logging.debug("Nuance explanation raw response: %r", raw[:300])
        return raw
    except requests.RequestException as e:
        logging.warning("Nuance explanation request failed: %s", e)
        return ""


def _ensure_anki_deck(deck: str = ANKI_DECK) -> None:
    """Create the Anki deck if it doesn't already exist."""
    payload = {
        "action": "createDeck",
        "version": 6,
        "params": {"deck": deck},
    }
    try:
        resp = requests.post(ANKI_URL, json=payload, timeout=ANKI_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("error") is not None:
            logging.warning("AnkiConnect createDeck error: %s", body["error"])
        else:
            logging.debug("Anki deck ensured: %s", deck)
    except requests.RequestException as e:
        logging.warning("AnkiConnect createDeck failed: %s", e)


def create_anki_card(front: str, back_html: str, deck: str = ANKI_DECK) -> bool:
    """Create a single Basic Anki card with the given front and back HTML."""
    _ensure_anki_deck(deck)
    payload = {
        "action": "addNote",
        "version": 6,
        "params": {
            "note": {
                "deckName": deck,
                "modelName": "Basic",
                "fields": {"Front": front, "Back": back_html},
                "options": {"allowDuplicate": False},
                "tags": ["writing-tool"],
            }
        },
    }
    try:
        resp = requests.post(ANKI_URL, json=payload, timeout=ANKI_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("error") is not None:
            logging.debug("AnkiConnect skipped: %s", body["error"])
            return False
        logging.info("Anki card created id=%s", body.get("result"))
        return True
    except requests.exceptions.ConnectionError:
        logging.debug("AnkiConnect not reachable — Anki likely not running")
        return False
    except requests.RequestException as e:
        logging.warning("AnkiConnect request failed: %s", e)
        return False


def _run_anki_creation(original: str, corrected: str) -> None:
    try:
        explanation = generate_explanation(original, corrected)
        if not explanation:
            logging.debug("Anki: no explanation generated, creating card without one")
            explanation = "(no explanation available)"
        back_html = (
            f"{corrected}"
            f"<hr>"
            f"<small>{explanation.replace(chr(10), '<br>')}</small>"
        )
        if create_anki_card(original, back_html):
            notify("Writing Tool — Anki", f'Card added to "{ANKI_DECK}"')
    except Exception:
        logging.exception("Unexpected error in Anki creation thread")


def _run_learn_card(text: str) -> None:
    try:
        explanation = generate_nuance_explanation(text)
        if not explanation or explanation.strip().upper() == "NONE":
            logging.debug("Anki: no nuances found, using fallback back")
            back_html = "(no explanation available — added manually)"
        else:
            back_html = explanation.replace("\n", "<br>")
        if create_anki_card(text, back_html, deck=ANKI_VOCAB_DECK):
            notify("Writing Tool — Learn", f'Vocab card added to "{ANKI_VOCAB_DECK}"')
        else:
            notify("Writing Tool — Learn", "Card already exists or Anki not running.")
    except Exception:
        logging.exception("Unexpected error in learn card thread")


# ──────────────────────────────────────────────────────────────
# Menu bar app
# ──────────────────────────────────────────────────────────────

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class WritingToolApp(rumps.App):
    def __init__(self):
        self.lock = threading.Lock()
        self.processing = False
        self._spinner_frame = 0

        menu_items = []
        for mode_name, cfg in MODES.items():
            item = rumps.MenuItem(
                cfg["label"],
                callback=self._make_callback(mode_name),
            )
            menu_items.append(item)

        menu_items.append(rumps.MenuItem("Custom…", callback=self._custom_callback))
        menu_items.append(None)  # separator
        menu_items.append(rumps.MenuItem("Learn This", callback=self._learn_callback))

        super().__init__(
            name="Writing Tool",
            title="✎",
            menu=menu_items,
            quit_button="Quit",
        )
        self._spinner_timer = rumps.Timer(self._tick_spinner, 0.1)

    def _tick_spinner(self, _timer):
        if self.processing:
            self.title = _SPINNER_FRAMES[self._spinner_frame % len(_SPINNER_FRAMES)]
            self._spinner_frame += 1

    def _start_spinner(self):
        self._spinner_frame = 0
        self._spinner_timer.start()

    def _stop_spinner(self):
        self._spinner_timer.stop()
        self.title = "✎"

    def _make_callback(self, mode_name: str):
        def callback(sender):
            logging.info("Menu clicked: %s", mode_name)
            with self.lock:
                if self.processing:
                    logging.warning("Already processing, ignoring click")
                    notify("Writing Tool", "Already processing — please wait.")
                    return
                self.processing = True
            self._start_spinner()
            threading.Thread(
                target=self._process, args=(mode_name,), daemon=True
            ).start()
        return callback

    def _get_custom_instruction(self) -> str | None:
        script = (
            'text returned of (display dialog "Enter rewrite instruction:" '
            'default answer "" with title "Writing Tool — Custom")'
        )
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def _custom_callback(self, sender):
        logging.info("Menu clicked: Custom")
        instruction = self._get_custom_instruction()
        if not instruction:
            logging.info("Custom: user cancelled or entered empty instruction")
            return
        with self.lock:
            if self.processing:
                logging.warning("Already processing, ignoring click")
                notify("Writing Tool", "Already processing — please wait.")
                return
            self.processing = True
        self._start_spinner()
        threading.Thread(
            target=self._process, args=(None,), kwargs={"instruction": instruction, "temperature": 0.5}, daemon=True
        ).start()

    def _learn_callback(self, sender):
        logging.info("Menu clicked: Learn This")
        with self.lock:
            if self.processing:
                logging.warning("Already processing, ignoring click")
                notify("Writing Tool", "Already processing — please wait.")
                return
            self.processing = True
        self._start_spinner()
        threading.Thread(target=self._process_learn, daemon=True).start()

    def _process_learn(self):
        try:
            text = pyperclip.paste()
            logging.debug("Clipboard contents (%d chars): %r", len(text or ""), (text or "")[:120])
            if not text or not text.strip():
                logging.warning("Clipboard is empty")
                notify("Writing Tool", "Clipboard is empty — copy some text first.")
                return
            logging.info("Generating nuance explanation: chars=%d", len(text))
            _run_learn_card(text)
        except Exception:
            logging.exception("Unexpected error in _process_learn")
        finally:
            self._stop_spinner()
            self.processing = False

    def _process(self, mode_name: str | None, *, instruction: str | None = None, temperature: float | None = None):
        if mode_name is not None:
            mode = MODES[mode_name]
            instruction = mode["instruction"]
            temperature = mode.get("temperature", 0.3)
        if temperature is None:
            temperature = 0.5
        try:
            original = pyperclip.paste()
            logging.debug("Clipboard contents (%d chars): %r", len(original or ""), (original or "")[:120])
            if not original or not original.strip():
                logging.warning("Clipboard is empty")
                notify("Writing Tool", "Clipboard is empty — copy some text first.")
                return
            logging.info("Sending to Ollama: model=%s mode=%s chars=%d", MODEL, mode_name or "custom", len(original))
            variants = rewrite_multiple(original, instruction, n=3, temperature=temperature)
            if not variants:
                logging.error("Ollama returned no results")
                notify("Writing Tool", "No result — check Ollama is running.")
                return
            logging.info("Got %d variants, showing picker", len(variants))
            chosen = pick_result(variants)
            if chosen:
                logging.info("User picked result (%d chars), copying to clipboard", len(chosen))
                pyperclip.copy(chosen)
                preview = chosen[:80] + ("…" if len(chosen) > 80 else "")
                notify("Writing Tool ✓", preview)
                if ANKI_ENABLED:
                    threading.Thread(
                        target=_run_anki_creation, args=(original, chosen), daemon=True
                    ).start()
            else:
                logging.info("User cancelled picker")
        except Exception:
            logging.exception("Unexpected error in _process")
        finally:
            self._stop_spinner()
            self.processing = False


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("Starting Writing Tool — model=%s ollama=%s", MODEL, OLLAMA_HOST)
    WritingToolApp().run()
