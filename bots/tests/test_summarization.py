"""Pure unit tests for the summarization prompt/format/parse helpers (no network)."""

import json
from unittest.mock import MagicMock, patch

import requests
from django.test import SimpleTestCase

from bots.models import DEFAULT_AZURE_OPENAI_API_VERSION
from bots.summarization import azure_client, prompts


class FormatTranscriptTests(SimpleTestCase):
    def test_labels_speakers_and_defaults_blank(self):
        out = prompts.format_transcript([{"speaker_name": "Alice", "text": "Hello"}, {"speaker_name": "", "text": "Hi"}])
        self.assertIn("Alice: Hello", out)
        self.assertIn("Unknown speaker: Hi", out)

    def test_drops_empty_text(self):
        self.assertEqual(prompts.format_transcript([{"speaker_name": "A", "text": "   "}]), "")

    def test_truncate_keeps_head_and_tail(self):
        text = "A" * 100 + "B" * 100
        out = prompts.truncate_transcript(text, budget=60)
        self.assertLessEqual(len(out), 60 + len(prompts._TRUNCATION_MARKER))
        self.assertTrue(out.startswith("A"))
        self.assertTrue(out.endswith("B"))
        self.assertIn("truncated", out)

    def test_truncate_noop_when_within_budget(self):
        self.assertEqual(prompts.truncate_transcript("short", budget=100), "short")

    def test_is_long_enough(self):
        self.assertFalse(prompts.is_long_enough("tiny"))
        self.assertFalse(prompts.is_long_enough(None))
        self.assertTrue(prompts.is_long_enough("x" * prompts.MIN_TRANSCRIPT_CHARS))

    def test_context_block_includes_only_known_fields(self):
        block = prompts.build_context_block({"title": "Q3 Planning", "duration": "30m"})
        self.assertIn("Title: Q3 Planning", block)
        self.assertIn("Duration: 30m", block)
        self.assertNotIn("Platform", block)

    def test_context_block_empty_when_nothing_known(self):
        self.assertEqual(prompts.build_context_block({}), "")

    def test_context_block_joins_participant_list(self):
        block = prompts.build_context_block({"participants": ["Alice", "Bob"]})
        self.assertIn("Participants: Alice, Bob", block)
        self.assertNotIn("[", block)


class BuildMessagesTests(SimpleTestCase):
    def test_system_and_user_roles(self):
        messages = azure_client.build_messages([{"speaker_name": "A", "text": "hi"}], {"title": "T"})
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_system_prompt_requests_json_output(self):
        # The prompt asks for a JSON {title, notes} envelope so parse_response can split them.
        self.assertIn("JSON", azure_client.build_messages([], None)[0]["content"])

    def test_user_message_carries_context_and_transcript(self):
        user = azure_client.build_messages([{"speaker_name": "A", "text": "hi"}], {"title": "T"})[1]["content"]
        self.assertIn("Title: T", user)
        self.assertIn("A: hi", user)


class ParseResponseTests(SimpleTestCase):
    def test_clean_json(self):
        out = azure_client.parse_response(json.dumps({"title": "Weekly roadmap sync", "notes": "# Meeting Notes\nbody"}))
        self.assertEqual(out["title"], "Weekly roadmap sync")
        self.assertTrue(out["notes"].startswith("# Meeting Notes"))

    def test_code_fenced_json(self):
        out = azure_client.parse_response('```json\n{"title": "Title here", "notes": "body"}\n```')
        self.assertEqual(out["title"], "Title here")
        self.assertEqual(out["notes"], "body")

    def test_non_json_falls_back_to_notes_without_title(self):
        raw = "# Meeting Notes\nSome content"
        out = azure_client.parse_response(raw)
        self.assertIsNone(out["title"])
        self.assertEqual(out["notes"], raw)

    def test_title_capped_at_twenty_words(self):
        out = azure_client.parse_response(json.dumps({"title": " ".join(["word"] * 30), "notes": "n"}))
        self.assertEqual(len(out["title"].split()), azure_client.TITLE_MAX_WORDS)

    def test_missing_or_empty_title(self):
        out = azure_client.parse_response(json.dumps({"notes": "just notes"}))
        self.assertIsNone(out["title"])
        self.assertEqual(out["notes"], "just notes")

    def test_title_stripped_of_markdown_and_whitespace(self):
        out = azure_client.parse_response(json.dumps({"title": "  # Big   Title  ", "notes": "n"}))
        self.assertEqual(out["title"], "Big Title")

    def test_empty_reply(self):
        out = azure_client.parse_response("")
        self.assertIsNone(out["title"])
        self.assertEqual(out["notes"], "")

    def test_single_line_fence_preserves_body(self):
        # A one-line fence must not blank the reply — notes are kept, never lost.
        out = azure_client.parse_response('```json{"title": "T", "notes": "important body"}```')
        self.assertTrue(out["notes"])

    def test_dict_missing_notes_returns_empty_not_envelope(self):
        out = azure_client.parse_response(json.dumps({"title": "T"}))
        self.assertEqual(out["notes"], "")
        self.assertNotIn("{", out["notes"])

    def test_json_list_falls_back_to_notes(self):
        raw = json.dumps([1, 2, 3])
        out = azure_client.parse_response(raw)
        self.assertIsNone(out["title"])
        self.assertEqual(out["notes"], raw)


class SummarizeTests(SimpleTestCase):
    def _creds(self):
        return {"endpoint": "https://res.openai.azure.com", "deployment": "gpt-5.2-codex-wf", "api_key": "SECRET", "api_version": "2025-04-01-preview"}

    def _completed(self, text):
        return {"status": "completed", "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    @patch("bots.summarization.azure_client.requests.post")
    def test_summarize_calls_responses_api(self, mock_post):
        response = MagicMock()
        response.json.return_value = self._completed('{"title":"T","notes":"n"}')
        mock_post.return_value = response

        out = azure_client.summarize(self._creds(), [{"role": "system", "content": "x"}], max_output_tokens=16000)

        self.assertEqual(out, '{"title":"T","notes":"n"}')
        url = mock_post.call_args[0][0]
        _, kwargs = mock_post.call_args
        self.assertIn("/openai/responses?api-version=2025-04-01-preview", url)
        self.assertEqual(kwargs["json"]["model"], "gpt-5.2-codex-wf")
        self.assertEqual(kwargs["json"]["input"], [{"role": "system", "content": "x"}])
        self.assertEqual(kwargs["json"]["max_output_tokens"], 16000)
        self.assertNotIn("temperature", kwargs["json"])  # unsupported by this reasoning model
        self.assertNotIn("response_format", kwargs["json"])
        self.assertEqual(kwargs["headers"]["api-key"], "SECRET")
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertIn("timeout", kwargs)
        response.raise_for_status.assert_called_once()

    @patch("bots.summarization.azure_client.requests.post")
    def test_summarize_raises_on_truncation(self, mock_post):
        response = MagicMock()
        response.json.return_value = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}
        mock_post.return_value = response
        with self.assertRaises(azure_client.SummaryTruncated):
            azure_client.summarize(self._creds(), [])

    @patch("bots.summarization.azure_client.requests.post")
    def test_summarize_empty_when_no_message(self, mock_post):
        # Only a reasoning item / content-filtered -> no message text; caller treats "" as failed.
        response = MagicMock()
        response.json.return_value = {"status": "completed", "output": [{"type": "reasoning"}]}
        mock_post.return_value = response
        self.assertEqual(azure_client.summarize(self._creds(), []), "")

    @patch("bots.summarization.azure_client.requests.post")
    def test_summarize_raises_on_http_error(self, mock_post):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError("500")
        mock_post.return_value = response
        with self.assertRaises(requests.HTTPError):
            azure_client.summarize(self._creds(), [])


class EndpointUrlTests(SimpleTestCase):
    def test_builds_responses_url_and_defaults_api_version(self):
        url = azure_client._endpoint_url({"endpoint": "https://res.openai.azure.com/", "deployment": "d", "api_key": "k"})
        self.assertEqual(url, f"https://res.openai.azure.com/openai/responses?api-version={DEFAULT_AZURE_OPENAI_API_VERSION}")

    def test_honours_explicit_api_version(self):
        url = azure_client._endpoint_url({"endpoint": "https://res.openai.azure.com", "deployment": "d", "api_key": "k", "api_version": "2025-04-01-preview"})
        self.assertTrue(url.endswith("/openai/responses?api-version=2025-04-01-preview"))
