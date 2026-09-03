"""The task that transcribes one group and writes the words back to its rows.

Nothing here may lose speech. The audio lives on the AudioChunk rows from the moment a clip is
cut, so a lost Redis key or a dead worker costs the grouping and never the audio -- and if the
group request fails for good, every clip is still transcribed the way it is today, one at a time.
"""

from unittest import mock

from django.test import TransactionTestCase

from bots.models import AudioChunk, Bot, Credentials, Organization, Participant, Project, Recording, RecordingStates, Utterance
from bots.tasks.process_local_utterance_group_task import process_local_utterance_group

ELEVENLABS_PROVIDER = 7
SAMPLE_RATE = 16000
UTTERANCE_MS = 600
GROUP_PATH = "bots.tasks.process_local_utterance_group_task.get_transcription_via_elevenlabs_for_utterance_group"
FALLBACK_PATH = "bots.tasks.process_local_utterance_group_task.process_utterance"


class ProcessLocalUtteranceGroupTest(TransactionTestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Org")
        self.project = Project.objects.create(name="Proj", organization=self.org)
        self.bot = Bot.objects.create(
            project=self.project,
            meeting_url="local_recording",
            settings={"transcription_settings": {"elevenlabs": {"tag_audio_events": False}}},
        )
        self.recording = Recording.objects.create(
            bot=self.bot,
            recording_type=1,
            transcription_type=1,
            state=RecordingStates.IN_PROGRESS,
            transcription_provider=ELEVENLABS_PROVIDER,
        )
        self.participant = Participant.objects.create(bot=self.bot, uuid="mic")
        Credentials.objects.create(project=self.project, credential_type=Credentials.CredentialTypes.ELEVENLABS)
        self.utterances = [self._utterance(0), self._utterance(UTTERANCE_MS * 2)]
        self.ids = [utterance.id for utterance in self.utterances]

    def _utterance(self, timestamp_ms):
        chunk = AudioChunk.objects.create(
            recording=self.recording,
            participant=self.participant,
            audio_blob=b"\x01\x00" * (UTTERANCE_MS * SAMPLE_RATE // 1000),
            timestamp_ms=timestamp_ms,
            duration_ms=UTTERANCE_MS,
            sample_rate=SAMPLE_RATE,
        )
        return Utterance.objects.create(
            recording=self.recording,
            participant=self.participant,
            audio_chunk=chunk,
            timestamp_ms=timestamp_ms,
            duration_ms=UTTERANCE_MS,
        )

    def _transcriptions(self):
        return {utterance.id: {"transcript": f"line {index}", "words": [], "language": "en"} for index, utterance in enumerate(self.utterances)}

    def test_every_row_in_the_group_gets_its_own_text(self):
        with mock.patch(GROUP_PATH, return_value=(self._transcriptions(), None)):
            process_local_utterance_group(self.ids, [0.6])

        for index, utterance in enumerate(self.utterances):
            utterance.refresh_from_db()
            self.assertEqual(utterance.transcription["transcript"], f"line {index}")

    def test_the_group_is_sent_as_one_request(self):
        with mock.patch(GROUP_PATH, return_value=(self._transcriptions(), None)) as send:
            process_local_utterance_group(self.ids, [0.6])

        self.assertEqual(send.call_count, 1)

    def test_a_replayed_task_does_not_transcribe_again(self):
        """Celery redelivers. A second run must be a no-op, not a second charge."""
        with mock.patch(GROUP_PATH, return_value=(self._transcriptions(), None)):
            process_local_utterance_group(self.ids, [0.6])

        with mock.patch(GROUP_PATH) as send:
            process_local_utterance_group(self.ids, [0.6])

        send.assert_not_called()

    def test_a_group_that_cannot_be_transcribed_falls_back_to_one_clip_at_a_time(self):
        """Degrade to today's behaviour rather than lose the speech."""
        failure = {"reason": "some_permanent_failure"}
        with mock.patch(GROUP_PATH, return_value=(None, failure)), mock.patch(FALLBACK_PATH) as single:
            process_local_utterance_group(self.ids, [0.6])

        self.assertEqual([call.args[0] for call in single.delay.call_args_list], self.ids)

    def test_the_audio_survives_a_failed_group_so_the_fallback_has_something_to_send(self):
        failure = {"reason": "some_permanent_failure"}
        with mock.patch(GROUP_PATH, return_value=(None, failure)), mock.patch(FALLBACK_PATH):
            process_local_utterance_group(self.ids, [0.6])

        for utterance in self.utterances:
            utterance.refresh_from_db()
            self.assertTrue(utterance.audio_chunk.audio_blob, "a retry would have nothing to transcribe")

    def test_a_group_whose_rows_have_vanished_is_dropped_quietly(self):
        """delete_data removes utterances; a queued task must not crash on them."""
        Utterance.objects.filter(id__in=self.ids).delete()

        with mock.patch(GROUP_PATH) as send:
            process_local_utterance_group(self.ids, [0.6])

        send.assert_not_called()

    def test_an_empty_group_does_nothing(self):
        with mock.patch(GROUP_PATH) as send:
            process_local_utterance_group([], [])

        send.assert_not_called()

    def test_rows_are_sent_in_the_order_the_gaps_describe(self):
        """The gaps are positional, so a reordered fetch would misplace every word."""
        with mock.patch(GROUP_PATH, return_value=(self._transcriptions(), None)) as send:
            process_local_utterance_group(list(reversed(self.ids)), [0.6])

        sent = [utterance.id for utterance in send.call_args.args[0]]
        self.assertEqual(sent, list(reversed(self.ids)))
