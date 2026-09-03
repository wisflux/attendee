"""Transcribing a run of local utterances as one ElevenLabs request.

The provider can only send one clip per request today, which is what makes the model re-identify
the language every second or two. This is the group equivalent: join the clips, send once, and
hand the returned words back to the rows they came from.
"""

from unittest import mock

from django.test import TransactionTestCase

from bots.models import AudioChunk, Bot, Credentials, Organization, Participant, Project, Recording, RecordingStates, Utterance
from bots.transcription_providers.elevenlabs import get_transcription_via_elevenlabs_for_utterance_group

ELEVENLABS_PROVIDER = 7
SAMPLE_RATE = 16000
UTTERANCE_MS = 600


def api_response(status_code=200, payload=None):
    response = mock.Mock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    response.text = ""
    return response


def transcript_payload(words, language_probability=0.99):
    return {
        "text": " ".join(word["text"] for word in words),
        "words": words,
        "language_code": "en",
        "language_probability": language_probability,
    }


def spoken(text, start, end):
    return {"text": text, "start": start, "end": end}


class ElevenLabsUtteranceGroupTest(TransactionTestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Org")
        self.project = Project.objects.create(name="Proj", organization=self.org)
        self.bot = Bot.objects.create(
            project=self.project,
            meeting_url="local_recording",
            # Mirrors build_local_session_settings: a local recording turns event tags OFF.
            settings={"transcription_settings": {"elevenlabs": {"tag_audio_events": False}}},
        )
        self.recording = Recording.objects.create(
            bot=self.bot,
            recording_type=1,
            transcription_type=1,
            state=RecordingStates.COMPLETE,
            transcription_provider=ELEVENLABS_PROVIDER,
        )
        self.participant = Participant.objects.create(bot=self.bot, uuid="mic")
        Credentials.objects.create(project=self.project, credential_type=Credentials.CredentialTypes.ELEVENLABS)
        self.utterances = [self._utterance(0), self._utterance(UTTERANCE_MS * 2)]

    def _utterance(self, timestamp_ms):
        chunk = AudioChunk.objects.create(
            recording=self.recording,
            participant=self.participant,
            audio_blob=b"\x01\x00" * (UTTERANCE_MS * SAMPLE_RATE // 1000),
            timestamp_ms=timestamp_ms,
            duration_ms=UTTERANCE_MS,
            sample_rate=SAMPLE_RATE,
        )
        utterance = Utterance.objects.create(
            recording=self.recording,
            participant=self.participant,
            audio_chunk=chunk,
            timestamp_ms=timestamp_ms,
            duration_ms=UTTERANCE_MS,
        )
        utterance.refresh_from_db()
        return utterance

    def _patched_creds(self):
        return mock.patch.object(Credentials, "get_credentials", return_value={"api_key": "fake-key"})

    def _run(self, response, gaps_seconds=None):
        with self._patched_creds(), mock.patch("bots.transcription_providers.elevenlabs.requests.post", return_value=response) as post:
            transcriptions, failure = get_transcription_via_elevenlabs_for_utterance_group(
                self.utterances,
                gaps_seconds=gaps_seconds if gaps_seconds is not None else [0.6],
            )
        return transcriptions, failure, post

    def test_the_whole_group_goes_in_a_single_request(self):
        """The entire point: one request, so the language is identified once for all of it."""
        response = api_response(payload=transcript_payload([spoken("hello", 0.1, 0.4)]))

        _, _, post = self._run(response)

        self.assertEqual(post.call_count, 1)

    def test_each_word_comes_back_on_the_row_it_was_spoken_in(self):
        # Windows with a 0.6s gap: 0.0-0.6 and 1.2-1.8.
        words = [spoken("first", 0.1, 0.4), spoken("second", 1.3, 1.6)]
        transcriptions, failure, _ = self._run(api_response(payload=transcript_payload(words)))

        self.assertIsNone(failure)
        self.assertEqual(transcriptions[self.utterances[0].id]["transcript"], "first")
        self.assertEqual(transcriptions[self.utterances[1].id]["transcript"], "second")

    def test_the_group_request_carries_the_sessions_own_settings(self):
        """A local session turns audio-event tags off; the group request must honour that, or
        '(mouse clicking)' comes back as if somebody had said it."""
        _, _, post = self._run(api_response(payload=transcript_payload([spoken("hi", 0.1, 0.3)])))

        self.assertIs(post.call_args.kwargs["data"]["tag_audio_events"], False)

    def test_a_rejected_request_reports_a_failure_and_writes_nothing(self):
        transcriptions, failure, _ = self._run(api_response(status_code=500))

        self.assertIsNone(transcriptions)
        self.assertIsNotNone(failure)

    def test_a_whole_group_is_not_discarded_for_low_language_confidence(self):
        """The single-clip rule drops text scoring under the threshold, because a one-second clip
        is too short to identify a language from. A group is half a minute of speech -- that
        reasoning does not apply, and dropping it would lose real speech from every row."""
        words = [spoken("first", 0.1, 0.4), spoken("second", 1.3, 1.6)]
        payload = transcript_payload(words, language_probability=0.2)

        transcriptions, failure, _ = self._run(api_response(payload=payload))

        self.assertIsNone(failure)
        self.assertEqual(transcriptions[self.utterances[0].id]["transcript"], "first")
