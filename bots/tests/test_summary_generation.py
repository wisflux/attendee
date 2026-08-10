"""Tests for meeting-summary generation: pure helpers, the Celery task, the reaper, the endpoint."""

import json
from datetime import timedelta
from unittest.mock import patch

import requests
from django.test import SimpleTestCase, override_settings
from django.utils import timezone

from bots.models import Bot, Credentials, Participant, Recording, RecordingManager, RecordingStates, RecordingTranscriptionStates, RecordingTypes, SessionTypes, SummaryStates, TranscriptionProviders, TranscriptionTypes, Utterance
from bots.summarization import generation
from bots.summarization.azure_client import SummaryTruncated
from bots.tasks.generate_meeting_summary_task import generate_meeting_summary, reap_stale_summary_generations
from bots.tests.meetings_api_base import MEMBER_A, MEMBER_B, MeetingsApiTestBase, member_token

AZURE = Credentials.CredentialTypes.AZURE_OPENAI
TASK = "bots.tasks.generate_meeting_summary_task"
GOOD = json.dumps({"title": "Launch planning deciding the release date and QA sign-off owner", "notes": "# Meeting Notes\n\nDetails."})


def _http_error(status_code):
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(response=response)


class GenerationHelpersTests(SimpleTestCase):
    def test_classify_failure_messages(self):
        self.assertIn("authentication", generation.classify_failure(_http_error(401)).lower())
        self.assertIn("deployment", generation.classify_failure(_http_error(404)).lower())
        self.assertIn("rate limit", generation.classify_failure(_http_error(429)).lower())
        self.assertIn("temporary", generation.classify_failure(_http_error(503)).lower())
        self.assertIn("reach", generation.classify_failure(requests.ConnectionError()).lower())
        self.assertEqual(generation.classify_failure(requests.RequestException()), "Summary generation failed.")

    def test_is_transient(self):
        self.assertTrue(generation.is_transient(_http_error(429)))
        self.assertTrue(generation.is_transient(_http_error(500)))
        self.assertTrue(generation.is_transient(requests.Timeout()))
        self.assertFalse(generation.is_transient(_http_error(401)))
        self.assertFalse(generation.is_transient(_http_error(404)))

    def test_concise_retry_appends_without_mutating(self):
        original = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        out = generation.concise_retry_messages(original)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[-1]["role"], "user")
        self.assertIn("concise", out[-1]["content"].lower())
        self.assertEqual(len(original), 2)  # original not mutated


class SummaryTaskTests(MeetingsApiTestBase):
    def setUp(self):
        super().setUp()
        self.bot = self.make_meeting(owner_user_id=MEMBER_A, session_type=SessionTypes.BOT)

    def _add_credential(self):
        Credentials.objects.create(project=self.project, credential_type=AZURE).set_credentials({"endpoint": "https://r.openai.azure.com", "deployment": "d", "api_key": "k", "api_version": "2025-04-01-preview"})

    def _seed_transcript(self, lines=12):
        recording = self.bot.recordings.filter(is_default_recording=True).first()
        speaker = Participant.objects.create(bot=self.bot, uuid="p1", full_name="Alice", is_host=True)
        for i in range(lines):
            Utterance.objects.create(recording=recording, participant=speaker, timestamp_ms=i * 1000, duration_ms=500, transcription={"transcript": "We discussed the launch timeline in detail."})

    def _reload(self):
        self.bot.refresh_from_db()
        return self.bot

    @patch(f"{TASK}.summarize", return_value=GOOD)
    def test_success_writes_summary_and_title(self, _):
        self._add_credential()
        self._seed_transcript()
        generate_meeting_summary(self.bot.id)
        b = self._reload()
        self.assertEqual(b.summary_state, SummaryStates.DONE)
        self.assertTrue(b.summary.startswith("# Meeting Notes"))
        self.assertEqual(b.name, "Launch planning deciding the release date and QA sign-off owner")
        self.assertIsNone(b.summary_error)

    def test_too_short_is_skipped(self):
        self._add_credential()  # but no transcript -> too short
        generate_meeting_summary(self.bot.id)
        self.assertEqual(self._reload().summary_state, SummaryStates.SKIPPED)

    def test_no_credential_fails_with_reason(self):
        self._seed_transcript()
        generate_meeting_summary(self.bot.id)
        b = self._reload()
        self.assertEqual(b.summary_state, SummaryStates.FAILED)
        self.assertIn("credential", b.summary_error.lower())

    @patch(f"{TASK}.summarize", side_effect=_http_error(401))
    def test_auth_error_fails_and_keeps_name(self, _):
        self._add_credential()
        self._seed_transcript()
        generate_meeting_summary(self.bot.id)
        b = self._reload()
        self.assertEqual(b.summary_state, SummaryStates.FAILED)
        self.assertIn("authentication", b.summary_error.lower())
        self.assertEqual(b.name, "Standup")  # success-only: name untouched on failure

    @patch(f"{TASK}.summarize", side_effect=[SummaryTruncated(), GOOD])
    def test_truncation_triggers_concise_retry(self, mock_sum):
        self._add_credential()
        self._seed_transcript()
        generate_meeting_summary(self.bot.id)
        self.assertEqual(self._reload().summary_state, SummaryStates.DONE)
        self.assertEqual(mock_sum.call_count, 2)

    @patch(f"{TASK}.summarize", side_effect=[SummaryTruncated(), SummaryTruncated()])
    def test_double_truncation_fails(self, _):
        self._add_credential()
        self._seed_transcript()
        generate_meeting_summary(self.bot.id)
        b = self._reload()
        self.assertEqual(b.summary_state, SummaryStates.FAILED)
        self.assertIn("too long", b.summary_error.lower())

    @patch(f"{TASK}.summarize", return_value=json.dumps({"title": "t", "notes": ""}))
    def test_empty_notes_fails(self, _):
        self._add_credential()
        self._seed_transcript()
        generate_meeting_summary(self.bot.id)
        self.assertEqual(self._reload().summary_state, SummaryStates.FAILED)

    @patch(f"{TASK}.summarize", return_value=GOOD)
    def test_atomic_claim_noops_when_not_pending(self, mock_sum):
        self._add_credential()
        self._seed_transcript()
        Bot.objects.filter(id=self.bot.id).update(summary_state=SummaryStates.DONE)
        generate_meeting_summary(self.bot.id)  # already DONE -> claim returns 0 -> no-op
        mock_sum.assert_not_called()
        self.assertEqual(self._reload().summary_state, SummaryStates.DONE)

    def test_delete_data_clears_all_summary_fields(self):
        Bot.objects.filter(id=self.bot.id).update(summary="notes", summary_error="e", summary_generated_at=timezone.now(), summary_state=SummaryStates.DONE)
        self._reload().delete_data()
        b = self._reload()
        self.assertIsNone(b.summary)
        self.assertIsNone(b.summary_error)
        self.assertIsNone(b.summary_generated_at)
        self.assertIsNone(b.summary_started_at)
        self.assertEqual(b.summary_state, SummaryStates.PENDING)

    def test_reaper_resets_stale_generating(self):
        Bot.objects.filter(id=self.bot.id).update(summary_state=SummaryStates.GENERATING, summary_started_at=timezone.now() - timedelta(hours=1))
        reap_stale_summary_generations()
        b = self._reload()
        self.assertEqual(b.summary_state, SummaryStates.PENDING)
        self.assertIsNone(b.summary_started_at)

    def test_reaper_leaves_fresh_generating(self):
        Bot.objects.filter(id=self.bot.id).update(summary_state=SummaryStates.GENERATING, summary_started_at=timezone.now())
        reap_stale_summary_generations()
        self.assertEqual(self._reload().summary_state, SummaryStates.GENERATING)

    @patch(f"{TASK}.summarize")
    def test_delete_during_generation_does_not_resurrect(self, mock_sum):
        # A concurrent delete_data (transcript wiped, state -> PENDING) must void the task's late
        # write, so a summary never outlives its transcript.
        self._add_credential()
        self._seed_transcript()

        def delete_then_return(*args, **kwargs):
            self._reload().delete_data()  # simulate the delete landing mid-generation
            return GOOD

        mock_sum.side_effect = delete_then_return
        generate_meeting_summary(self.bot.id)
        b = self._reload()
        self.assertIsNone(b.summary)  # NOT resurrected
        self.assertNotEqual(b.summary_state, SummaryStates.DONE)

    def test_tasks_are_registered_with_celery(self):
        from attendee.celery import app

        self.assertIn("bots.tasks.generate_meeting_summary_task.generate_meeting_summary", app.tasks)
        self.assertIn("bots.tasks.generate_meeting_summary_task.reap_stale_summary_generations", app.tasks)

    def _make_in_progress_recording(self, is_default):
        return Recording.objects.create(bot=self.bot, recording_type=RecordingTypes.AUDIO_AND_VIDEO, transcription_type=TranscriptionTypes.NON_REALTIME, transcription_provider=TranscriptionProviders.ELEVENLABS, is_default_recording=is_default, state=RecordingStates.COMPLETE, transcription_state=RecordingTranscriptionStates.IN_PROGRESS)

    @patch(f"{TASK}.generate_meeting_summary.delay")
    def test_transcript_complete_enqueues_after_commit(self, mock_delay):
        # make_meeting already created a default recording (COMPLETE); use a second, unfinished one.
        self.bot.recordings.update(is_default_recording=False)
        recording = self._make_in_progress_recording(is_default=True)
        with self.captureOnCommitCallbacks(execute=True):
            RecordingManager.set_recording_transcription_complete(recording)
        mock_delay.assert_called_once_with(self.bot.id)

    @patch(f"{TASK}.generate_meeting_summary.delay")
    def test_non_default_recording_does_not_enqueue(self, mock_delay):
        recording = self._make_in_progress_recording(is_default=False)
        with self.captureOnCommitCallbacks(execute=True):
            RecordingManager.set_recording_transcription_complete(recording)
        mock_delay.assert_not_called()

    def test_serializer_exposes_summary_error(self):
        from bots.meetings_serializers import MeetingSerializer

        Bot.objects.filter(id=self.bot.id).update(summary_error="boom", summary_state=SummaryStates.FAILED)
        data = MeetingSerializer(self._reload()).data
        self.assertEqual(data["summary_error"], "boom")


@override_settings(TEAM_DAY_JWT_SECRET="test-team-day-secret")
class SummarizeEndpointTests(MeetingsApiTestBase):
    def _post(self, object_id, token):
        headers = {"HTTP_AUTHORIZATION": f"Token {self.api_key_plain}", "HTTP_X_USER_TOKEN": token}
        return self.client.post(f"/api/v1/meetings/{object_id}/summarize", **headers)

    @patch(f"{TASK}.generate_meeting_summary.delay")
    def test_regenerate_from_terminal_enqueues(self, mock_delay):
        bot = self.make_meeting(owner_user_id=MEMBER_A)
        Bot.objects.filter(id=bot.id).update(summary_state=SummaryStates.FAILED, summary_error="x")
        response = self._post(bot.object_id, member_token(MEMBER_A))
        self.assertEqual(response.status_code, 200)
        mock_delay.assert_called_once_with(bot.id)
        bot.refresh_from_db()
        self.assertEqual(bot.summary_state, SummaryStates.PENDING)
        self.assertIsNone(bot.summary_error)

    @patch(f"{TASK}.generate_meeting_summary.delay")
    def test_noop_while_fresh_generating(self, mock_delay):
        bot = self.make_meeting(owner_user_id=MEMBER_A)
        Bot.objects.filter(id=bot.id).update(summary_state=SummaryStates.GENERATING, summary_started_at=timezone.now())
        response = self._post(bot.object_id, member_token(MEMBER_A))
        self.assertEqual(response.status_code, 200)
        mock_delay.assert_not_called()
        self.assertEqual(Bot.objects.get(id=bot.id).summary_state, SummaryStates.GENERATING)

    @patch(f"{TASK}.generate_meeting_summary.delay")
    def test_other_members_meeting_is_404(self, mock_delay):
        bot = self.make_meeting(owner_user_id=MEMBER_A)
        response = self._post(bot.object_id, member_token(MEMBER_B))
        self.assertEqual(response.status_code, 404)
        mock_delay.assert_not_called()

    @patch(f"{TASK}.generate_meeting_summary.delay")
    def test_stale_generating_is_restarted(self, mock_delay):
        # A crashed worker left it GENERATING; Regenerate recovers it (no beat needed).
        bot = self.make_meeting(owner_user_id=MEMBER_A)
        Bot.objects.filter(id=bot.id).update(summary_state=SummaryStates.GENERATING, summary_started_at=timezone.now() - timedelta(hours=1))
        response = self._post(bot.object_id, member_token(MEMBER_A))
        self.assertEqual(response.status_code, 200)
        mock_delay.assert_called_once_with(bot.id)
        self.assertEqual(Bot.objects.get(id=bot.id).summary_state, SummaryStates.PENDING)
