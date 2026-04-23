#!/usr/bin/env python3
"""
Local Writing Tool — macOS menu bar app powered by Ollama.

Workflow: Copy text → click menu bar icon (✎) → select mode → paste improved text.
"""

import json
import logging
import os
import re
import subprocess
import threading
import sys
from datetime import datetime
from pathlib import Path

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

# Backend selection: "ollama" (default), "openai", or "anthropic"
BACKEND = os.environ.get("BACKEND", "ollama").lower()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

# AnkiConnect — set ANKI_ENABLED=0 to disable
ANKI_ENABLED = os.environ.get("ANKI_ENABLED", "1") == "1"
ANKI_URL = os.environ.get("ANKI_URL", "http://127.0.0.1:8765")
ANKI_DECK = os.environ.get("ANKI_DECK", "Writing Errors")
ANKI_VOCAB_DECK = os.environ.get("ANKI_VOCAB_DECK", "English Vocabulary")
ANKI_EXERCISE_DECK = os.environ.get("ANKI_EXERCISE_DECK", "Writing Exercises")
ANKI_REGISTER_DECK = os.environ.get("ANKI_REGISTER_DECK", "Register Notes")
ANKI_TIMEOUT = int(os.environ.get("ANKI_TIMEOUT", "5"))

# Daily writing prompt: set DAILY_PROMPT_ENABLED=0 to disable the scheduled prompt.
# DAILY_PROMPT_HOUR is local-time 0–23 — the timer fires once on/after this hour each day.
DAILY_PROMPT_ENABLED = os.environ.get("DAILY_PROMPT_ENABLED", "1") == "1"
DAILY_PROMPT_HOUR = int(os.environ.get("DAILY_PROMPT_HOUR", "10"))

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
    "native": {
        "instruction": (
            "Rewrite this so it sounds like a native English speaker wrote it. "
            "The input is grammatically close but uses unnatural word choices, collocations, or prepositions. "
            "Focus on: (1) collocations (e.g. 'make a decision' not 'do a decision', "
            "'heavy rain' not 'strong rain'); (2) preposition pairings "
            "(e.g. 'discuss X' not 'discuss about X', 'depend on' not 'depend of'); "
            "(3) idiomatic word choices a native would actually use. "
            "Keep the meaning, length, and register exactly the same. "
            "Do not change sentence structure unless a collocation requires it. "
            "If the text already sounds native, return it unchanged."
        ),
        "label": "Sound Native",
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


_AUDIENCES = [
    "Slack — peer",
    "Slack — manager",
    "Email — client",
    "Email — executive",
    "LinkedIn — public",
]


def pick_audience() -> str | None:
    """Show a native macOS list picker for audience selection, or None if cancelled."""
    def esc(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')
    as_list = "{" + ", ".join(f'"{esc(a)}"' for a in _AUDIENCES) + "}"
    script = (
        f'choose from list {as_list} '
        f'with prompt "Who is the audience?" '
        f'without multiple selections allowed'
    )
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    chosen = result.stdout.strip()
    if not chosen or chosen == "false":
        return None
    return chosen if chosen in _AUDIENCES else None



# ──────────────────────────────────────────────────────────────
# Model dispatcher
# ──────────────────────────────────────────────────────────────

def _call_model(
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the configured LLM backend and return the response text, or '' on error."""
    if BACKEND == "openai":
        return _call_openai(prompt, system, temperature, max_tokens)
    if BACKEND == "anthropic":
        return _call_anthropic(prompt, system, temperature, max_tokens)
    return _call_ollama(prompt, system, temperature, max_tokens)


def _call_ollama(
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if system:
        payload["system"] = system
    try:
        logging.debug("POST %s (timeout=%ds)", OLLAMA_URL, TIMEOUT)
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        logging.debug("Ollama responded: %r", result[:120])
        return result
    except requests.RequestException as e:
        logging.error("Ollama request failed: %s", e)
        raise


def _call_openai(
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    import openai  # lazy import — only required when BACKEND=openai
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = (response.choices[0].message.content or "").strip()
        logging.debug("OpenAI responded: %r", result[:120])
        return result
    except openai.OpenAIError as e:
        logging.error("OpenAI request failed: %s", e)
        raise


def _call_anthropic(
    prompt: str,
    system: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    import anthropic  # lazy import — only required when BACKEND=anthropic
    client = anthropic.Anthropic()
    kwargs = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    try:
        response = client.messages.create(**kwargs)
        result = (response.content[0].text or "").strip()
        logging.debug("Anthropic responded: %r", result[:120])
        return result
    except anthropic.APIError as e:
        logging.error("Anthropic request failed: %s", e)
        raise


# ──────────────────────────────────────────────────────────────
# Ollama interaction
# ──────────────────────────────────────────────────────────────

def rewrite(text: str, instruction: str, temperature: float = 0.3) -> str:
    """Send text to the configured LLM backend and return the rewritten version."""
    prompt = f"{instruction}\n\nText to rewrite:\n{text}"
    try:
        return _call_model(prompt, SYSTEM_PROMPT, temperature, 1024)
    except Exception as e:
        logging.error("Rewrite failed: %s", e)
        notify("Writing Tool — Error", str(e))
        return ""


def rewrite_multiple(text: str, instruction: str, n: int, temperature: float) -> list:
    """Ask the LLM for n numbered variants. Returns a list of strings (may be shorter than n on parse failure)."""
    prompt = (
        f"{instruction}\n\n"
        f"Provide exactly {n} different rewrites, numbered 1. 2. 3. "
        f"Put a blank line between each. Output only the numbered rewrites, nothing else.\n\n"
        f"Text:\n{text}"
    )
    try:
        raw = _call_model(prompt, SYSTEM_PROMPT, temperature, 1024)
    except Exception as e:
        logging.error("Rewrite failed: %s", e)
        notify("Writing Tool — Error", str(e))
        return []
    if not raw:
        return []
    logging.debug("LLM responded: %r", raw[:300])
    variants = re.findall(r'(?m)^\d+\.\s+([\s\S]+?)(?=\n\s*\d+\.|\s*$)', raw)
    variants = [v.strip() for v in variants if v.strip()]
    if len(variants) != n:
        logging.warning("Expected %d variants, parsed %d — falling back to raw", n, len(variants))
        return [raw]
    return variants


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
You are an English tutor helping a non-native speaker learn natural English.

Find every phrase in the text below that is worth explaining to a learner:
- Phrasal verbs (e.g. "speak up", "give away")
- Idiomatic expressions (e.g. "break a leg")
- Common collocations — natural word pairings that may not be obvious (e.g. "naturally loud", "make a decision")
- Informal or colloquial expressions

If none are found, output exactly: NONE

For each one output a single bullet:
- <exact phrase as it appears> — <meaning>

Text:
{text}\
"""


def generate_explanation(original: str, corrected: str) -> str:
    """Ask the LLM to explain the differences between original and corrected text."""
    prompt = _ANKI_EXPLANATION_PROMPT.format(original=original, corrected=corrected)
    try:
        raw = _call_model(prompt, None, 0.1, 512)
        logging.debug("Anki explanation raw response: %r", raw[:300])
        return raw
    except Exception as e:
        logging.warning("Explanation request failed: %s", e)
        return ""


def generate_nuance_explanation(text: str) -> str:
    """Ask the LLM to find phrasal verbs, idioms, collocations, and informal expressions in text."""
    prompt = _NUANCE_PROMPT.format(text=text)
    try:
        raw = _call_model(prompt, None, 0.1, 1024)
        logging.debug("Nuance explanation raw response: %r", raw[:300])
        return raw
    except Exception as e:
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
# CEFR level feedback
# ──────────────────────────────────────────────────────────────

_CEFR_PROMPT = """\
You are an English examiner assessing a non-native speaker's writing against the CEFR scale (A1-C2).

Assess the text below. Respond in this exact format - no markdown, no preamble:
LEVEL: <A1|A2|B1|B2|C1|C2>
RATIONALE: <one sentence citing specific evidence from the text>
NEXT LEVEL: <the level one step above, or C2 if already at C2>
SUGGESTIONS:
- <concrete rewrite of a phrase from the text that would lift it toward the next level>
- ... (2-3 suggestions total)

Text:
{text}\
"""

_CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

_PROGRESS_LOG = Path.home() / ".writing_tool_progress.jsonl"


def _parse_cefr_response(raw: str) -> tuple[str | None, str, str | None, list[str]]:
    """Parse the LLM's CEFR response into (level, rationale, next_level, suggestions)."""
    level: str | None = None
    rationale = ""
    next_level: str | None = None
    suggestions: list[str] = []
    in_suggestions = False
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("LEVEL:"):
            in_suggestions = False
            value = stripped.split(":", 1)[1].strip().upper()
            if value in _CEFR_LEVELS:
                level = value
        elif upper.startswith("RATIONALE:"):
            in_suggestions = False
            rationale = stripped.split(":", 1)[1].strip()
        elif upper.startswith("NEXT LEVEL:"):
            in_suggestions = False
            value = stripped.split(":", 1)[1].strip().upper()
            if value in _CEFR_LEVELS:
                next_level = value
        elif upper.startswith("SUGGESTIONS:"):
            in_suggestions = True
        elif in_suggestions and stripped.startswith("-"):
            suggestions.append(stripped.lstrip("-").strip())
    return level, rationale, next_level, suggestions


def _append_progress_log(level: str | None, text: str) -> None:
    entry = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "chars": len(text),
        "excerpt": text[:200],
    }
    try:
        with _PROGRESS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logging.warning("Failed to append progress log: %s", e)


def _show_cefr_dialog(level: str | None, rationale: str, next_level: str | None, suggestions: list[str]) -> None:
    header = f"Level: {level or 'unknown'}"
    if next_level and next_level != level:
        header += f"  →  aim for {next_level}"
    body_lines = [header, ""]
    if rationale:
        body_lines.append(rationale)
        body_lines.append("")
    if suggestions:
        body_lines.append("To level up, try:")
        for s in suggestions:
            body_lines.append(f"• {s}")
    body = "\n".join(body_lines)
    safe = body.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display dialog "{safe}" with title "Writing Tool — CEFR Level" buttons {{"OK"}} default button "OK"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def _run_cefr_check(text: str) -> None:
    try:
        raw = _call_model(_CEFR_PROMPT.format(text=text), None, 0.2, 512)
        if not raw:
            notify("Writing Tool", "No result — check your LLM backend is reachable.")
            return
        level, rationale, next_level, suggestions = _parse_cefr_response(raw)
        _append_progress_log(level, text)
        _show_cefr_dialog(level, rationale, next_level, suggestions)
    except Exception:
        logging.exception("Unexpected error in CEFR check")


# ──────────────────────────────────────────────────────────────
# Register (audience) check
# ──────────────────────────────────────────────────────────────

_REGISTER_PROMPT = """\
You are an English tutor helping a non-native speaker match register to audience.

Audience: {audience}

Text:
{text}

Respond in this exact format - no markdown, no preamble:
VERDICT: <Too formal | Matches | Too casual>
RATIONALE: <one short sentence>
SWAPS:
- "<phrase from text>" -> "<better phrase>" — <why>
- ... (0-3 swaps total; omit the SWAPS section entirely if VERDICT is Matches)\
"""

_REGISTER_VERDICTS = ("Too formal", "Matches", "Too casual")

_SWAP_LINE_RE = re.compile(r'^[-•]\s*"(?P<original>[^"]+)"\s*(?:->|→)\s*"(?P<better>[^"]+)"\s*(?:[-—–]\s*(?P<why>.+))?$')


def _parse_register_response(raw: str) -> tuple[str | None, str, list[dict]]:
    """Parse the LLM's register response into (verdict, rationale, swaps).

    Each swap is a dict: {"original": ..., "better": ..., "why": ...}.
    """
    verdict: str | None = None
    rationale = ""
    swaps: list[dict] = []
    in_swaps = False
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("VERDICT:"):
            in_swaps = False
            value = stripped.split(":", 1)[1].strip()
            for v in _REGISTER_VERDICTS:
                if value.lower().startswith(v.lower()):
                    verdict = v
                    break
        elif upper.startswith("RATIONALE:"):
            in_swaps = False
            rationale = stripped.split(":", 1)[1].strip()
        elif upper.startswith("SWAPS:"):
            in_swaps = True
        elif in_swaps:
            m = _SWAP_LINE_RE.match(stripped)
            if m:
                swaps.append({
                    "original": m.group("original"),
                    "better": m.group("better"),
                    "why": (m.group("why") or "").strip(),
                })
    return verdict, rationale, swaps


def _show_register_dialog(audience: str, verdict: str | None, rationale: str, swaps: list[dict]) -> None:
    lines = [f"Audience: {audience}", f"Verdict: {verdict or 'unknown'}", ""]
    if rationale:
        lines.append(rationale)
        lines.append("")
    if swaps:
        lines.append("Suggested swaps:")
        for s in swaps:
            tail = f" — {s['why']}" if s['why'] else ""
            lines.append(f'• "{s["original"]}" → "{s["better"]}"{tail}')
    body = "\n".join(lines)
    safe = body.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display dialog "{safe}" with title "Writing Tool — Register" buttons {{"OK"}} default button "OK"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def _run_register_check(text: str, audience: str) -> None:
    try:
        raw = _call_model(
            _REGISTER_PROMPT.format(audience=audience, text=text),
            None, 0.2, 512,
        )
        if not raw:
            notify("Writing Tool", "No result — check your LLM backend is reachable.")
            return
        verdict, rationale, swaps = _parse_register_response(raw)
        _show_register_dialog(audience, verdict, rationale, swaps)
        if verdict and verdict != "Matches":
            for swap in swaps:
                front = f'"{swap["original"]}" (to {audience})'
                back = f'Use instead: "{swap["better"]}"'
                if swap["why"]:
                    back += f"<br><small>{swap['why']}</small>"
                create_anki_card(front, back, deck=ANKI_REGISTER_DECK)
    except Exception:
        logging.exception("Unexpected error in register check")


# ──────────────────────────────────────────────────────────────
# Daily writing prompt
# ──────────────────────────────────────────────────────────────

_DAILY_PROMPTS = [
    "In 2-3 sentences, tell a colleague you disagree with their proposal.",
    "In 2-3 sentences, decline a meeting politely.",
    "In 2-3 sentences, ask your manager for feedback on a recent project.",
    "In 2-3 sentences, explain to a teammate why you missed a deadline.",
    "In 2-3 sentences, introduce yourself to a new team on Slack.",
    "In 2-3 sentences, ask a colleague to review your pull request.",
    "In 2-3 sentences, apologize for sending a message with the wrong information.",
    "In 2-3 sentences, tell your team you will be out of office next week.",
    "In 2-3 sentences, congratulate a colleague on a promotion.",
    "In 2-3 sentences, ask a client to clarify their requirements.",
    "In 2-3 sentences, push back on a deadline you think is unrealistic.",
    "In 2-3 sentences, thank a coworker who helped you with a tricky bug.",
    "In 2-3 sentences, propose a short call to resolve a long email thread.",
    "In 2-3 sentences, summarize today's standup for someone who missed it.",
    "In 2-3 sentences, describe a recent book or article to a friend.",
    "In 2-3 sentences, ask a neighbor to keep the noise down without sounding rude.",
    "In 2-3 sentences, tell a friend you can't make it to their event.",
    "In 2-3 sentences, recommend a restaurant to a coworker visiting your city.",
    "In 2-3 sentences, explain your weekend plans to a colleague.",
    "In 2-3 sentences, ask a manager for a day off.",
    "In 2-3 sentences, announce a small win in your team chat.",
    "In 2-3 sentences, write a LinkedIn post about something you learned this week.",
    "In 2-3 sentences, respond to a vague request for 'an update ASAP'.",
    "In 2-3 sentences, tell a vendor their proposal is too expensive.",
    "In 2-3 sentences, invite a coworker to grab coffee.",
]

_DAILY_CORRECTION_PROMPT = """\
You are an English tutor correcting a non-native speaker's short response to a writing prompt.

Return ONLY the corrected version of the response — no explanations, no quotes, no markdown, no preamble.
Fix spelling, grammar, articles, prepositions, verb tenses, and unnatural phrasing.
Keep the original meaning and tone. Do not add new sentences or ideas.
If the response is already correct, return it unchanged.

Prompt: {prompt}

Response: {response}\
"""

_DAILY_STATE_FILE = Path.home() / ".writing_tool_daily_state"


def _pick_daily_prompt() -> str:
    import random
    return random.choice(_DAILY_PROMPTS)


def _ask_daily_prompt_response(prompt: str) -> str | None:
    safe = prompt.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'text returned of (display dialog "{safe}" '
        f'default answer "" with title "Writing Tool — Daily Prompt")'
    )
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _correct_daily_response(prompt: str, response: str) -> str:
    try:
        corrected = _call_model(
            _DAILY_CORRECTION_PROMPT.format(prompt=prompt, response=response),
            None, 0.2, 512,
        )
        return corrected.strip()
    except Exception as e:
        logging.warning("Daily correction failed: %s", e)
        return ""


def _show_daily_result_dialog(prompt: str, original: str, corrected: str, explanation: str) -> None:
    lines = [f"Prompt: {prompt}", "", f"You wrote:\n{original}", "", f"Corrected:\n{corrected}"]
    if explanation:
        lines.append("")
        lines.append("Changes:")
        lines.append(explanation)
    body = "\n".join(lines)
    safe = body.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display dialog "{safe}" with title "Writing Tool — Daily Review" buttons {{"OK"}} default button "OK"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def _run_daily_prompt() -> None:
    try:
        prompt = _pick_daily_prompt()
        response = _ask_daily_prompt_response(prompt)
        if not response:
            logging.info("Daily prompt: user cancelled or empty response")
            return
        corrected = _correct_daily_response(prompt, response)
        if not corrected:
            notify("Writing Tool", "No result — check your LLM backend is reachable.")
            return
        if corrected == response:
            _show_daily_result_dialog(prompt, response, corrected, "No changes needed — well done!")
            _mark_daily_prompt_done()
            return
        explanation = generate_explanation(response, corrected)
        _show_daily_result_dialog(prompt, response, corrected, explanation or "")
        if ANKI_ENABLED:
            back_html = (
                f"{corrected}"
                f"<hr>"
                f"<small>Prompt: {prompt}<br>"
                f"{(explanation or '(no explanation)').replace(chr(10), '<br>')}</small>"
            )
            create_anki_card(response, back_html, deck=ANKI_DECK)
        _mark_daily_prompt_done()
    except Exception:
        logging.exception("Unexpected error in daily prompt flow")


def _read_daily_state() -> str | None:
    try:
        return _DAILY_STATE_FILE.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None
    except OSError as e:
        logging.warning("Failed to read daily state: %s", e)
        return None


def _mark_daily_prompt_done() -> None:
    today = datetime.now().date().isoformat()
    try:
        _DAILY_STATE_FILE.write_text(today, encoding="utf-8")
    except OSError as e:
        logging.warning("Failed to write daily state: %s", e)


def _should_fire_daily_prompt(now: datetime | None = None) -> bool:
    if not DAILY_PROMPT_ENABLED:
        return False
    now = now or datetime.now()
    if now.hour < DAILY_PROMPT_HOUR:
        return False
    return _read_daily_state() != now.date().isoformat()


# ──────────────────────────────────────────────────────────────
# Error pattern analysis & exercise generation
# ──────────────────────────────────────────────────────────────

def fetch_deck_cards(deck: str = ANKI_DECK) -> list[dict]:
    """Fetch all card fronts/backs from an Anki deck via AnkiConnect.

    Returns a list of {"front": ..., "back": ...} dicts.
    """
    # Step 1: find all note IDs in the deck
    find_payload = {
        "action": "findNotes",
        "version": 6,
        "params": {"query": f'"deck:{deck}"'},
    }
    try:
        resp = requests.post(ANKI_URL, json=find_payload, timeout=ANKI_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("error") is not None:
            logging.warning("AnkiConnect findNotes error: %s", body["error"])
            return []
        note_ids = body.get("result", [])
    except requests.RequestException as e:
        logging.warning("AnkiConnect findNotes failed: %s", e)
        return []

    if not note_ids:
        logging.info("No notes found in deck %r", deck)
        return []

    # Step 2: get note details
    info_payload = {
        "action": "notesInfo",
        "version": 6,
        "params": {"notes": note_ids},
    }
    try:
        resp = requests.post(ANKI_URL, json=info_payload, timeout=ANKI_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("error") is not None:
            logging.warning("AnkiConnect notesInfo error: %s", body["error"])
            return []
        notes = body.get("result", [])
    except requests.RequestException as e:
        logging.warning("AnkiConnect notesInfo failed: %s", e)
        return []

    cards = []
    for note in notes:
        fields = note.get("fields", {})
        front = fields.get("Front", {}).get("value", "")
        back = fields.get("Back", {}).get("value", "")
        if front:
            cards.append({"front": front, "back": back})
    logging.info("Fetched %d cards from deck %r", len(cards), deck)
    return cards


_ANALYZE_PATTERNS_PROMPT = """\
You are an English tutor analyzing a student's writing mistakes.

Below are pairs of original text (with errors) and corrected text from past writing sessions.
Identify the recurring error PATTERNS — not individual mistakes.

Group them into categories like: Article errors, Preposition errors, Verb tense errors, \
Word order errors, Spelling/typo patterns, Tone issues, etc.

For each pattern:
- Name it concisely
- Describe what the student keeps getting wrong
- Give 1-2 examples from the data

Output format (one pattern per block, separated by blank lines):
PATTERN: <name>
DESCRIPTION: <what goes wrong>
EXAMPLES: <1-2 brief examples from the data>

If fewer than 3 cards are provided, still try to identify any patterns you can.
If no clear patterns exist, output exactly: NO_PATTERNS

Cards:
{cards_text}\
"""


def analyze_error_patterns(cards: list[dict]) -> str:
    """Send card data to the LLM and get back a description of recurring error patterns."""
    cards_text = "\n\n".join(
        f"Original: {c['front']}\nCorrected: {c['back']}"
        for c in cards
    )
    prompt = _ANALYZE_PATTERNS_PROMPT.format(cards_text=cards_text)
    try:
        raw = _call_model(prompt, None, 0.2, 2048)
        logging.debug("Pattern analysis response: %r", raw[:300])
        return raw
    except Exception as e:
        logging.error("Pattern analysis failed: %s", e)
        return ""


_GENERATE_EXERCISES_PROMPT = """\
You are an English tutor creating sentence-correction exercises.

Based on the error patterns below, generate practice exercises. For each pattern, \
create 1-2 exercises where the student must fix a sentence that contains that type of error.

The exercises must use NEW sentences — do not copy from the examples.
Make the sentences realistic (workplace Slack messages, emails, casual professional writing).

Output format (one exercise per block, separated by blank lines):
PATTERN: <which pattern this targets>
BROKEN: <sentence with the error>
FIXED: <corrected sentence>
HINT: <short hint about what to look for>

Error patterns:
{patterns}\
"""


def generate_exercises(patterns: str) -> list[dict]:
    """Ask the LLM to generate sentence-correction exercises for the given patterns.

    Returns a list of {"pattern": ..., "broken": ..., "fixed": ..., "hint": ...} dicts.
    """
    prompt = _GENERATE_EXERCISES_PROMPT.format(patterns=patterns)
    try:
        raw = _call_model(prompt, None, 0.5, 2048)
    except Exception as e:
        logging.error("Exercise generation failed: %s", e)
        return []
    if not raw:
        return []
    logging.debug("Exercise generation response: %r", raw[:300])

    exercises = []
    blocks = re.split(r'\n\s*\n', raw)
    for block in blocks:
        pattern_m = re.search(r'PATTERN:\s*(.+)', block)
        broken_m = re.search(r'BROKEN:\s*(.+)', block)
        fixed_m = re.search(r'FIXED:\s*(.+)', block)
        hint_m = re.search(r'HINT:\s*(.+)', block)
        if broken_m and fixed_m:
            exercises.append({
                "pattern": pattern_m.group(1).strip() if pattern_m else "General",
                "broken": broken_m.group(1).strip(),
                "fixed": fixed_m.group(1).strip(),
                "hint": hint_m.group(1).strip() if hint_m else "",
            })
    logging.info("Parsed %d exercises", len(exercises))
    return exercises


def create_exercise_cards(exercises: list[dict], deck: str = ANKI_EXERCISE_DECK) -> int:
    """Create sentence-correction exercise cards in Anki. Returns number of cards created."""
    created = 0
    for ex in exercises:
        front = (
            f"<b>Fix this sentence:</b><br><br>"
            f"{ex['broken']}"
        )
        if ex.get("hint"):
            front += f"<br><br><small>Hint: {ex['hint']}</small>"
        back = (
            f"{ex['fixed']}<br><br>"
            f"<small>Pattern: {ex['pattern']}</small>"
        )
        if create_anki_card(front, back, deck=deck):
            created += 1
    return created


def _run_practice_generation() -> None:
    """Full pipeline: fetch cards → analyze patterns → generate exercises → create cards."""
    try:
        cards = fetch_deck_cards(ANKI_DECK)
        if not cards:
            notify("Writing Tool", "No cards found in your Writing Errors deck.")
            return

        notify("Writing Tool", f"Analyzing {len(cards)} cards for patterns…")
        patterns = analyze_error_patterns(cards)
        if not patterns or patterns.strip() == "NO_PATTERNS":
            notify("Writing Tool", "No clear error patterns found yet — keep writing!")
            return

        exercises = generate_exercises(patterns)
        if not exercises:
            notify("Writing Tool", "Could not generate exercises — try again later.")
            return

        created = create_exercise_cards(exercises)
        notify(
            "Writing Tool — Practice",
            f"Created {created} exercise card{'s' if created != 1 else ''} "
            f'in "{ANKI_EXERCISE_DECK}"',
        )
    except Exception:
        logging.exception("Unexpected error in practice generation")


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
        menu_items.append(rumps.MenuItem("Estimate Level", callback=self._cefr_callback))
        menu_items.append(rumps.MenuItem("Check Register", callback=self._register_callback))
        menu_items.append(None)  # separator
        menu_items.append(rumps.MenuItem("Daily Prompt", callback=self._daily_prompt_callback))
        menu_items.append(rumps.MenuItem("Practice Weak Spots", callback=self._practice_callback))

        super().__init__(
            name="Writing Tool",
            title="✎",
            menu=menu_items,
            quit_button="Quit",
        )
        self._spinner_timer = rumps.Timer(self._tick_spinner, 0.1)
        self._daily_timer = rumps.Timer(self._tick_daily, 600)  # check every 10 min
        if DAILY_PROMPT_ENABLED:
            self._daily_timer.start()

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

    def _cefr_callback(self, sender):
        logging.info("Menu clicked: Estimate Level")
        with self.lock:
            if self.processing:
                logging.warning("Already processing, ignoring click")
                notify("Writing Tool", "Already processing — please wait.")
                return
            self.processing = True
        self._start_spinner()
        threading.Thread(target=self._process_cefr, daemon=True).start()

    def _process_cefr(self):
        try:
            text = pyperclip.paste()
            if not text or not text.strip():
                logging.warning("Clipboard is empty")
                notify("Writing Tool", "Clipboard is empty — copy some text first.")
                return
            logging.info("Running CEFR check: chars=%d", len(text))
            _run_cefr_check(text)
        except Exception:
            logging.exception("Unexpected error in _process_cefr")
        finally:
            self._stop_spinner()
            self.processing = False

    def _tick_daily(self, _timer):
        if self.processing or not _should_fire_daily_prompt():
            return
        logging.info("Daily timer firing")
        self._start_daily_prompt()

    def _daily_prompt_callback(self, sender):
        logging.info("Menu clicked: Daily Prompt")
        self._start_daily_prompt()

    def _start_daily_prompt(self):
        with self.lock:
            if self.processing:
                logging.warning("Already processing, skipping daily prompt")
                return
            self.processing = True
        self._start_spinner()
        threading.Thread(target=self._process_daily_prompt, daemon=True).start()

    def _process_daily_prompt(self):
        try:
            _run_daily_prompt()
        except Exception:
            logging.exception("Unexpected error in _process_daily_prompt")
        finally:
            self._stop_spinner()
            self.processing = False

    def _register_callback(self, sender):
        logging.info("Menu clicked: Check Register")
        text = pyperclip.paste()
        if not text or not text.strip():
            notify("Writing Tool", "Clipboard is empty — copy some text first.")
            return
        audience = pick_audience()
        if not audience:
            logging.info("Register: user cancelled audience picker")
            return
        with self.lock:
            if self.processing:
                logging.warning("Already processing, ignoring click")
                notify("Writing Tool", "Already processing — please wait.")
                return
            self.processing = True
        self._start_spinner()
        threading.Thread(
            target=self._process_register, args=(text, audience), daemon=True
        ).start()

    def _process_register(self, text: str, audience: str):
        try:
            logging.info("Running register check: audience=%s chars=%d", audience, len(text))
            _run_register_check(text, audience)
        except Exception:
            logging.exception("Unexpected error in _process_register")
        finally:
            self._stop_spinner()
            self.processing = False

    def _practice_callback(self, sender):
        logging.info("Menu clicked: Practice Weak Spots")
        if not ANKI_ENABLED:
            notify("Writing Tool", "Anki integration is disabled.")
            return
        with self.lock:
            if self.processing:
                logging.warning("Already processing, ignoring click")
                notify("Writing Tool", "Already processing — please wait.")
                return
            self.processing = True
        self._start_spinner()
        threading.Thread(target=self._process_practice, daemon=True).start()

    def _process_practice(self):
        try:
            _run_practice_generation()
        except Exception:
            logging.exception("Unexpected error in _process_practice")
        finally:
            self._stop_spinner()
            self.processing = False

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
            logging.info("Sending to LLM backend=%s mode=%s chars=%d", BACKEND, mode_name or "custom", len(original))
            variants = rewrite_multiple(original, instruction, n=3, temperature=temperature)
            if not variants:
                logging.error("LLM returned no results")
                notify("Writing Tool", "No result — check your LLM backend is reachable.")
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
    _active_model = {"ollama": MODEL, "openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL}.get(BACKEND, MODEL)
    logging.info("Starting Writing Tool — backend=%s model=%s", BACKEND, _active_model)
    WritingToolApp().run()
