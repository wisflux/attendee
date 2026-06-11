import uuid
from unittest import mock

from django.test import Client, TestCase

from accounts.models import Organization
from bots.bots_api_utils import retry_failed_transcription
from bots.models import (
    ApiKey,
    AudioChunk,
    Bot,
    BotStates,
    Participant,
    Project,
    Recording,
    RecordingStates,
    RecordingTranscriptionStates,
    RecordingTypes,
    TranscriptionProviders,
    TranscriptionTypes,
    Utterance,
)
from bots.serializers import BotSerializer


class RetryFailedTranscriptionTestBase(TestCase):
    def setUp(self):
        self.organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=self.organization)
        self.bot = Bot.objects.create(
            project=self.project,
            meeting_url="https://meet.google.com/abc-defg-hij",
            name="Test Bot",
            state=BotStates.ENDED,
            meeting_dedup_key="meet:abc-defg-hij",
        )
        self.recording = Recording.objects.create(
            bot=self.bot,
            recording_type=RecordingTypes.AUDIO_AND_VIDEO,
            transcription_type=TranscriptionTypes.NON_REALTIME,
            transcription_provider=TranscriptionProviders.ELEVENLABS,
            is_default_recording=True,
            state=RecordingStates.COMPLETE,
            transcription_state=RecordingTranscriptionStates.FAILED,
            transcription_failure_data={"failure_reasons": ["credentials_invalid"]},
        )
        self.participant = Participant.objects.create(bot=self.bot, uuid=str(uuid.uuid4()))

    def create_utterance(self, failure_data=None, with_audio=True, transcription=None):
        audio_chunk = None
        if with_audio:
            audio_chunk = AudioChunk.objects.create(recording=self.recording, participant=self.participant, audio_blob=b"rawpcmbytes", timestamp_ms=0, duration_ms=500, sample_rate=16000)
        return Utterance.objects.create(
            recording=self.recording,
            participant=self.participant,
            audio_chunk=audio_chunk,
            timestamp_ms=0,
            duration_ms=500,
            failure_data=failure_data,
            transcription=transcription,
            transcription_attempt_count=5 if failure_data else 0,
        )


class TestRetryFailedTranscription(RetryFailedTranscriptionTestBase):
    @mock.patch("bots.tasks.process_utterance_task.process_utterance.delay")
    def test_retries_failed_utterances_and_resets_recording_state(self, mock_delay):
        failed_with_audio = self.create_utterance(failure_data={"reason": "credentials_invalid"})
        failed_without_audio = self.create_utterance(failure_data={"reason": "credentials_invalid"}, with_audio=False)
        succeeded = self.create_utterance(transcription={"transcript": "hello"})

        retried_count, error = retry_failed_transcription(self.bot)

        self.assertIsNone(error)
        self.assertEqual(retried_count, 1)

        # Only the failed utterance WITH retained audio is reset and re-enqueued
        failed_with_audio.refresh_from_db()
        self.assertIsNone(failed_with_audio.failure_data)
        self.assertEqual(failed_with_audio.transcription_attempt_count, 0)
        mock_delay.assert_called_once_with(failed_with_audio.id)

        failed_without_audio.refresh_from_db()
        self.assertIsNotNone(failed_without_audio.failure_data)

        succeeded.refresh_from_db()
        self.assertEqual(succeeded.transcription, {"transcript": "hello"})

        # The recording's transcription state moves FAILED -> IN_PROGRESS with failure data cleared
        self.recording.refresh_from_db()
        self.assertEqual(self.recording.transcription_state, RecordingTranscriptionStates.IN_PROGRESS)
        self.assertIsNone(self.recording.transcription_failure_data)

    @mock.patch("bots.tasks.process_utterance_task.process_utterance.delay")
    def test_idempotent_when_nothing_to_retry(self, mock_delay):
        self.create_utterance(transcription={"transcript": "hello"})

        retried_count, error = retry_failed_transcription(self.bot)

        self.assertIsNone(error)
        self.assertEqual(retried_count, 0)
        mock_delay.assert_not_called()
        # Recording state untouched when there is nothing to retry
        self.recording.refresh_from_db()
        self.assertEqual(self.recording.transcription_state, RecordingTranscriptionStates.FAILED)

    def test_rejects_bot_still_in_meeting(self):
        self.bot.state = BotStates.JOINED_RECORDING
        self.bot.save()

        retried_count, error = retry_failed_transcription(self.bot)

        self.assertIsNone(retried_count)
        self.assertIn("error", error)

    def test_rejects_bot_with_deleted_data(self):
        self.bot.state = BotStates.DATA_DELETED
        self.bot.save()

        retried_count, error = retry_failed_transcription(self.bot)

        self.assertIsNone(retried_count)
        self.assertIn("error", error)


class TestRetryTranscriptionEndpoint(RetryFailedTranscriptionTestBase):
    def setUp(self):
        super().setUp()
        self.api_key, self.api_key_plain = ApiKey.create(project=self.project, name="Test API Key")
        self.client = Client()

    def post_retry(self, object_id):
        return self.client.post(f"/api/v1/bots/{object_id}/retry_transcription", HTTP_AUTHORIZATION=f"Token {self.api_key_plain}")

    @mock.patch("bots.tasks.process_utterance_task.process_utterance.delay")
    def test_endpoint_retries_and_returns_count(self, mock_delay):
        self.create_utterance(failure_data={"reason": "credentials_invalid"})

        response = self.post_retry(self.bot.object_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"retried_utterances": 1})
        self.assertEqual(mock_delay.call_count, 1)

    def test_endpoint_404_for_unknown_bot(self):
        response = self.post_retry("bot_doesnotexist000")
        self.assertEqual(response.status_code, 404)

    def test_endpoint_400_for_active_bot(self):
        self.bot.state = BotStates.JOINED_RECORDING
        self.bot.save()
        response = self.post_retry(self.bot.object_id)
        self.assertEqual(response.status_code, 400)


class TestMeetingDedupKeyInBotSerializer(RetryFailedTranscriptionTestBase):
    def test_bot_serializer_exposes_meeting_dedup_key(self):
        data = BotSerializer(self.bot).data
        self.assertEqual(data["meeting_dedup_key"], "meet:abc-defg-hij")

    def test_bot_detail_endpoint_returns_meeting_dedup_key(self):
        api_key, api_key_plain = ApiKey.create(project=self.project, name="Test API Key")
        response = Client().get(f"/api/v1/bots/{self.bot.object_id}", HTTP_AUTHORIZATION=f"Token {api_key_plain}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meeting_dedup_key"], "meet:abc-defg-hij")
