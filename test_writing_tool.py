"""
Tests for writing_tool.py.

Tests the full pipeline: clipboard read → Ollama API call → picker → clipboard write,
using mocked HTTP requests so no real Ollama instance is required.

Run with:
    python -m pytest test_writing_tool.py -v
    # or
    python test_writing_tool.py
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

import requests

import writing_tool as wt
from writing_tool import WritingToolApp, MODES, SYSTEM_PROMPT, rewrite, rewrite_multiple, generate_nuance_explanation, _run_learn_card


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _mock_ollama_response(text: str) -> MagicMock:
    """Return a mock requests.Response that looks like a successful Ollama reply."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": text}
    return resp


def _make_app() -> WritingToolApp:
    """Create a WritingToolApp without starting the rumps event loop."""
    with patch("rumps.App.__init__", return_value=None), \
         patch("rumps.MenuItem", return_value=MagicMock()):
        app = WritingToolApp()
    app.title = "✎"
    return app


# ──────────────────────────────────────────────────────────────
# rewrite() — unit tests for the HTTP layer
# ──────────────────────────────────────────────────────────────

class TestRewrite(unittest.TestCase):

    @patch("writing_tool.requests.post")
    def test_returns_model_response(self, mock_post):
        mock_post.return_value = _mock_ollama_response("cleaned up text")
        result = rewrite("raw text", "Make this better.")
        self.assertEqual(result, "cleaned up text")

    @patch("writing_tool.requests.post")
    def test_strips_whitespace_from_response(self, mock_post):
        mock_post.return_value = _mock_ollama_response("  trimmed  \n")
        result = rewrite("text", "instruction")
        self.assertEqual(result, "trimmed")

    @patch("writing_tool.requests.post")
    def test_passes_system_prompt(self, mock_post):
        mock_post.return_value = _mock_ollama_response("ok")
        rewrite("text", "instruction")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(payload["system"], SYSTEM_PROMPT)

    @patch("writing_tool.requests.post")
    def test_passes_temperature(self, mock_post):
        mock_post.return_value = _mock_ollama_response("ok")
        rewrite("text", "instruction", temperature=0.6)
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.6)

    @patch("writing_tool.requests.post")
    def test_default_temperature_is_0_3(self, mock_post):
        mock_post.return_value = _mock_ollama_response("ok")
        rewrite("text", "instruction")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.3)

    @patch("writing_tool.requests.post")
    def test_instruction_included_in_prompt(self, mock_post):
        mock_post.return_value = _mock_ollama_response("ok")
        rewrite("my text", "Do something specific.")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertIn("Do something specific.", payload["prompt"])
        self.assertIn("my text", payload["prompt"])

    @patch("writing_tool.notify")
    @patch("writing_tool.requests.post",
           side_effect=requests.RequestException("connection refused"))
    def test_returns_empty_string_on_error(self, mock_post, mock_notify):
        result = rewrite("text", "instruction")
        self.assertEqual(result, "")

    @patch("writing_tool.notify")
    @patch("writing_tool.requests.post",
           side_effect=requests.RequestException("timeout"))
    def test_sends_notification_on_error(self, mock_post, mock_notify):
        rewrite("text", "instruction")
        mock_notify.assert_called_once()
        title, msg = mock_notify.call_args[0]
        self.assertIn("Error", title)

    @patch("writing_tool.requests.post")
    def test_non_streaming_request(self, mock_post):
        mock_post.return_value = _mock_ollama_response("ok")
        rewrite("text", "instruction")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertFalse(payload["stream"])


# ──────────────────────────────────────────────────────────────
# rewrite_multiple() — variant generation and parsing
# ──────────────────────────────────────────────────────────────

class TestRewriteMultiple(unittest.TestCase):

    @patch("writing_tool.requests.post")
    def test_returns_three_parsed_variants(self, mock_post):
        raw = "1. First option\n\n2. Second option\n\n3. Third option"
        mock_post.return_value = _mock_ollama_response(raw)
        results = rewrite_multiple("text", "instruction", n=3, temperature=0.3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "First option")
        self.assertEqual(results[1], "Second option")
        self.assertEqual(results[2], "Third option")

    @patch("writing_tool.requests.post")
    def test_fallback_to_raw_on_parse_failure(self, mock_post):
        mock_post.return_value = _mock_ollama_response("unparseable response")
        results = rewrite_multiple("text", "instruction", n=3, temperature=0.3)
        self.assertEqual(results, ["unparseable response"])

    @patch("writing_tool.notify")
    @patch("writing_tool.requests.post",
           side_effect=requests.RequestException("timeout"))
    def test_returns_empty_list_on_error(self, mock_post, mock_notify):
        results = rewrite_multiple("text", "instruction", n=3, temperature=0.3)
        self.assertEqual(results, [])


# ──────────────────────────────────────────────────────────────
# Full pipeline — clipboard in → API → picker → clipboard out
# ──────────────────────────────────────────────────────────────

class TestFullPipeline(unittest.TestCase):
    """Drive WritingToolApp._process() directly for each mode."""

    def _run_process(self, mode_name: str, clipboard_input: str, variants: list, pick: str):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value=clipboard_input),
            patch("writing_tool.pyperclip.copy") as mock_copy,
            patch("writing_tool.notify"),
            patch("writing_tool.rewrite_multiple", return_value=variants),
            patch("writing_tool.pick_result", return_value=pick),
        ):
            app._process(mode_name)
        return mock_copy

    # --- casual ---

    def test_casual_updates_clipboard(self):
        mock_copy = self._run_process("casual", "I wanted to follow up.", ["Just checking in."], "Just checking in.")
        mock_copy.assert_called_once_with("Just checking in.")

    def test_casual_uses_temperature_0_6(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="some text"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.pick_result", return_value="result"),
            patch("writing_tool.requests.post",
                  return_value=_mock_ollama_response("1. a\n\n2. b\n\n3. c")) as mock_post,
        ):
            app._process("casual")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.6)

    def test_casual_instruction_mentions_slack(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="text"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.pick_result", return_value="ok"),
            patch("writing_tool.requests.post",
                  return_value=_mock_ollama_response("1. a\n\n2. b\n\n3. c")) as mock_post,
        ):
            app._process("casual")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertIn("Slack", payload["prompt"])

    # --- simplify ---

    def test_simplify_updates_clipboard(self):
        long_text = "In order to proceed with the implementation it is worth noting that we need to do the thing."
        mock_copy = self._run_process("simplify", long_text, ["We need to do the thing."], "We need to do the thing.")
        mock_copy.assert_called_once_with("We need to do the thing.")

    def test_simplify_uses_temperature_0_3(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="long text here"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.pick_result", return_value="short"),
            patch("writing_tool.requests.post",
                  return_value=_mock_ollama_response("1. a\n\n2. b\n\n3. c")) as mock_post,
        ):
            app._process("simplify")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.3)

    # --- soften ---

    def test_soften_updates_clipboard(self):
        mock_copy = self._run_process("soften", "Fix this now.", ["Could you take a look at this?"], "Could you take a look at this?")
        mock_copy.assert_called_once_with("Could you take a look at this?")

    def test_soften_uses_temperature_0_6(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="blunt message"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.pick_result", return_value="softer"),
            patch("writing_tool.requests.post",
                  return_value=_mock_ollama_response("1. a\n\n2. b\n\n3. c")) as mock_post,
        ):
            app._process("soften")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.6)

    def test_soften_instruction_mentions_collaborative(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="text"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.pick_result", return_value="ok"),
            patch("writing_tool.requests.post",
                  return_value=_mock_ollama_response("1. a\n\n2. b\n\n3. c")) as mock_post,
        ):
            app._process("soften")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertIn("collaborative", payload["prompt"])

    # --- direct ---

    def test_direct_updates_clipboard(self):
        buried = "I just wanted to reach out because I was thinking that maybe we could potentially..."
        mock_copy = self._run_process("direct", buried, ["Can we schedule a call?"], "Can we schedule a call?")
        mock_copy.assert_called_once_with("Can we schedule a call?")

    def test_direct_uses_temperature_0_3(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="wordy message"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.pick_result", return_value="direct"),
            patch("writing_tool.requests.post",
                  return_value=_mock_ollama_response("1. a\n\n2. b\n\n3. c")) as mock_post,
        ):
            app._process("direct")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.3)

    def test_direct_instruction_mentions_ask(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="text"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.pick_result", return_value="ok"),
            patch("writing_tool.requests.post",
                  return_value=_mock_ollama_response("1. a\n\n2. b\n\n3. c")) as mock_post,
        ):
            app._process("direct")
        payload = mock_post.call_args_list[0][1]["json"]
        self.assertIn("ask", payload["prompt"])


# ──────────────────────────────────────────────────────────────
# Notification behaviour
# ──────────────────────────────────────────────────────────────

class TestNotifications(unittest.TestCase):

    def test_success_notification_shows_preview(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="input"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify") as mock_notify,
            patch("writing_tool.rewrite_multiple", return_value=["rewritten text"]),
            patch("writing_tool.pick_result", return_value="rewritten text"),
        ):
            app._process("casual")

        last_call = mock_notify.call_args_list[-1]
        title, preview = last_call[0]
        self.assertIn("✓", title)
        self.assertIn("rewritten text", preview)

    def test_empty_clipboard_sends_notification_and_no_api_call(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value=""),
            patch("writing_tool.pyperclip.copy") as mock_copy,
            patch("writing_tool.notify") as mock_notify,
            patch("writing_tool.requests.post") as mock_post,
        ):
            app._process("casual")

        mock_post.assert_not_called()
        mock_copy.assert_not_called()
        mock_notify.assert_called_once()
        msg = mock_notify.call_args[0][1]
        self.assertIn("empty", msg.lower())

    def test_whitespace_only_clipboard_treated_as_empty(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="   \n\t  "),
            patch("writing_tool.pyperclip.copy") as mock_copy,
            patch("writing_tool.notify"),
            patch("writing_tool.requests.post") as mock_post,
        ):
            app._process("casual")

        mock_post.assert_not_called()
        mock_copy.assert_not_called()

    def test_empty_variants_does_not_overwrite_clipboard(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="original"),
            patch("writing_tool.pyperclip.copy") as mock_copy,
            patch("writing_tool.notify"),
            patch("writing_tool.rewrite_multiple", return_value=[]),
        ):
            app._process("casual")

        mock_copy.assert_not_called()

    def test_cancelled_picker_does_not_overwrite_clipboard(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="original"),
            patch("writing_tool.pyperclip.copy") as mock_copy,
            patch("writing_tool.notify"),
            patch("writing_tool.rewrite_multiple", return_value=["a", "b", "c"]),
            patch("writing_tool.pick_result", return_value=None),
        ):
            app._process("casual")

        mock_copy.assert_not_called()


# ──────────────────────────────────────────────────────────────
# Concurrency guard
# ──────────────────────────────────────────────────────────────

class TestConcurrencyGuard(unittest.TestCase):

    def test_second_click_ignored_while_processing(self):
        """Callback must not spawn a thread while processing is True."""
        app = _make_app()
        app.processing = True

        spawned = []
        original_thread = threading.Thread

        def tracking_thread(*args, **kwargs):
            spawned.append(True)
            return original_thread(*args, **kwargs)

        with patch("writing_tool.threading.Thread", side_effect=tracking_thread), \
             patch("writing_tool.notify"):
            callback = app._make_callback("casual")
            callback(None)

        self.assertEqual(spawned, [])

    def test_processing_flag_reset_after_success(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="text"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.rewrite_multiple", return_value=["result"]),
            patch("writing_tool.pick_result", return_value="result"),
        ):
            app._process("casual")

        self.assertFalse(app.processing)

    def test_processing_flag_reset_after_error(self):
        app = _make_app()
        with (
            patch("writing_tool.pyperclip.paste", return_value="text"),
            patch("writing_tool.pyperclip.copy"),
            patch("writing_tool.notify"),
            patch("writing_tool.rewrite_multiple", side_effect=Exception("boom")),
        ):
            app._process("casual")

        self.assertFalse(app.processing)


# ──────────────────────────────────────────────────────────────
# MODES config sanity checks
# ──────────────────────────────────────────────────────────────

class TestModesConfig(unittest.TestCase):

    def test_all_four_modes_present(self):
        for mode in ("casual", "simplify", "soften", "direct"):
            self.assertIn(mode, MODES, f"Missing mode: {mode}")

    def test_all_modes_have_required_keys(self):
        for name, cfg in MODES.items():
            for key in ("instruction", "label", "temperature"):
                self.assertIn(key, cfg, f"Mode '{name}' missing key '{key}'")

    def test_temperatures_are_valid(self):
        for name, cfg in MODES.items():
            t = cfg["temperature"]
            self.assertIsInstance(t, float, f"Mode '{name}' temperature is not a float")
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 1.0)

    def test_system_prompt_bans_filler_phrases(self):
        self.assertIn("Hope this helps", SYSTEM_PROMPT)
        self.assertIn("Just wanted to", SYSTEM_PROMPT)

    def test_casual_and_soften_have_higher_temperature(self):
        self.assertGreater(MODES["casual"]["temperature"], MODES["simplify"]["temperature"])
        self.assertGreater(MODES["soften"]["temperature"], MODES["direct"]["temperature"])


# ──────────────────────────────────────────────────────────────
# generate_nuance_explanation() — unit tests
# ──────────────────────────────────────────────────────────────

class TestNuance(unittest.TestCase):

    @patch("writing_tool.requests.post")
    def test_returns_model_response(self, mock_post):
        mock_post.return_value = _mock_ollama_response(
            "- under the weather — idiom — feeling ill — Often used to excuse an absence — "
            "Example: She stayed home because she was feeling under the weather.\n"
            "- stepping away from — phrasal verb — temporarily distancing oneself from — "
            "Used when taking a break from an activity — "
            "Example: He is stepping away from social media for a while."
        )
        result = generate_nuance_explanation(
            "Feeling under the weather so stepping away from my keyboard for a bit"
        )
        self.assertIn("under the weather", result)
        self.assertIn("stepping away from", result)

    @patch("writing_tool.requests.post")
    def test_none_response_returned_as_is(self, mock_post):
        mock_post.return_value = _mock_ollama_response("NONE")
        result = generate_nuance_explanation("I went to the store.")
        self.assertEqual(result, "NONE")

    @patch("writing_tool.requests.post", side_effect=requests.RequestException("timeout"))
    def test_request_exception_returns_empty_string(self, _mock_post):
        result = generate_nuance_explanation("Some text here.")
        self.assertEqual(result, "")



# ──────────────────────────────────────────────────────────────
# _run_learn_card() — always creates card regardless of explanation
# ──────────────────────────────────────────────────────────────

class TestRunLearnCard(unittest.TestCase):

    def _anki_mock(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"result": 12345, "error": None}
        return resp

    @patch("writing_tool.notify")
    @patch("writing_tool.requests.post")
    def test_card_created_when_explanation_succeeds(self, mock_post, mock_notify):
        """Normal path: explanation → card created with explanation as back."""
        mock_post.side_effect = [
            _mock_ollama_response("- speak up — phrasal verb — to say something"),
            self._anki_mock(),  # createDeck
            self._anki_mock(),  # addNote
        ]
        _run_learn_card("how do I speak up?")
        self.assertTrue(any(
            call.args and "Vocab card added" in str(call.args)
            for call in mock_notify.call_args_list
        ))

    @patch("writing_tool.notify")
    @patch("writing_tool.requests.post")
    def test_card_created_when_explanation_times_out(self, mock_post, mock_notify):
        """Timeout path: Ollama fails → card still created with fallback back."""
        mock_post.side_effect = [
            requests.RequestException("Read timed out"),  # nuance call
            self._anki_mock(),  # createDeck
            self._anki_mock(),  # addNote
        ]
        _run_learn_card("how do I speak up or talk about my accomplishments?")
        self.assertTrue(any(
            call.args and "Vocab card added" in str(call.args)
            for call in mock_notify.call_args_list
        ))

    @patch("writing_tool.notify")
    @patch("writing_tool.requests.post")
    def test_card_created_when_explanation_returns_none(self, mock_post, mock_notify):
        """NONE path: model says no phrases → card still created with fallback back."""
        mock_post.side_effect = [
            _mock_ollama_response("NONE"),  # nuance call
            self._anki_mock(),  # createDeck
            self._anki_mock(),  # addNote
        ]
        _run_learn_card("I went to the store.")
        self.assertTrue(any(
            call.args and "Vocab card added" in str(call.args)
            for call in mock_notify.call_args_list
        ))


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
