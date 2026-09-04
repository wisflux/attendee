"""The task that transcribes one group and writes the words back to its rows.

Nothing here may lose speech. The audio lives on the AudioChunk rows from the moment a clip is
cut, so a lost Redis key or a dead worker costs the grouping and never the audio -- and if the
group request fails for good, every clip is still transcribed the way it is today, one at a time.
"""

from unittest import mock

from django.test import TransactionTestCase

from bots.local_session_api_views import pending_utterance_count
from bots.models import AudioChunk, Bot, Credentials, Organization, Participant, Project, Recording, RecordingStates, RecordingTranscriptionStates, Utterance
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

    def _finish_recording(self):
        """The state finalize leaves behind: the recording is over, its words are not back yet.

        Refreshed first because Recording carries an optimistic-lock version: a task run in the
        same test has already saved the row, so this instance is stale.
        """
        self.recording.refresh_from_db()
        self.recording.state = RecordingStates.COMPLETE
        self.recording.save()

    def test_the_session_is_marked_transcribed_once_the_last_group_lands(self):
        """Nothing else notices this. set_recording_complete ran while these rows were still empty,
        and the group path never calls process_utterance, which is what settles the state on the
        bot path -- so the session would stay IN_PROGRESS forever and no summary would run."""
        self._finish_recording()

        with mock.patch(GROUP_PATH, return_value=(self._transcriptions(), None)):
            process_local_utterance_group(self.ids, [0.6])

        self.recording.refresh_from_db()
        self.assertEqual(self.recording.transcription_state, RecordingTranscriptionStates.COMPLETE)

    def test_a_session_still_recording_is_not_marked_transcribed(self):
        """A group that closed on "stopped talking" must not declare the session over."""
        with mock.patch(GROUP_PATH, return_value=(self._transcriptions(), None)):
            process_local_utterance_group(self.ids, [0.6])

        self.recording.refresh_from_db()
        self.assertEqual(self.recording.transcription_state, RecordingTranscriptionStates.IN_PROGRESS)

    def test_a_session_with_a_row_still_waiting_is_not_marked_transcribed(self):
        """Another group is still in flight, and its row has no text yet."""
        self._finish_recording()
        self._utterance(UTTERANCE_MS * 4)

        with mock.patch(GROUP_PATH, return_value=(self._transcriptions(), None)):
            process_local_utterance_group(self.ids, [0.6])

        self.recording.refresh_from_db()
        self.assertEqual(self.recording.transcription_state, RecordingTranscriptionStates.IN_PROGRESS)

    def test_a_replay_settles_a_session_a_crash_left_open(self):
        """The rows were written and the task died before the state was settled. The replay returns
        early because every row already has text -- it must still finish the session."""
        with mock.patch(GROUP_PATH, return_value=(self._transcriptions(), None)):
            process_local_utterance_group(self.ids, [0.6])
        self._finish_recording()

        with mock.patch(GROUP_PATH) as send:
            process_local_utterance_group(self.ids, [0.6])

        send.assert_not_called()
        self.recording.refresh_from_db()
        self.assertEqual(self.recording.transcription_state, RecordingTranscriptionStates.COMPLETE)

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


class PendingUtteranceCountTest(TransactionTestCase):
    """What the desktop needs to know that a flat line count no longer tells it.

    Rows are invisible until they have text, and a group holds several of them for up to a minute.
    Without a pending count the app sees the same number of lines poll after poll and declares the
    session finished over words that are still on their way.
    """

    def setUp(self):
        self.org = Organization.objects.create(name="Org")
        self.project = Project.objects.create(name="Proj", organization=self.org)
        self.bot = Bot.objects.create(project=self.project, meeting_url="local_recording")
        self.recording = Recording.objects.create(
            bot=self.bot,
            recording_type=1,
            transcription_type=1,
            state=RecordingStates.IN_PROGRESS,
            transcription_provider=ELEVENLABS_PROVIDER,
        )
        self.participant = Participant.objects.create(bot=self.bot, uuid="mic")

    def _utterance(self, transcription=None, failure_data=None):
        return Utterance.objects.create(
            recording=self.recording,
            participant=self.participant,
            timestamp_ms=0,
            duration_ms=UTTERANCE_MS,
            transcription=transcription,
            failure_data=failure_data,
        )

    def test_a_row_that_failed_for_good_is_not_counted(self):
        """Its words are never coming. Counted, it would hold the desktop on "Transcribing..." for
        the rest of the session -- which is what a bad key or an exhausted quota produced."""
        self._utterance(failure_data={"reason": "transcription_request_failed"})

        self.assertEqual(pending_utterance_count(self.bot), 0)

    def test_a_row_still_waiting_for_its_words_is_counted(self):
        self._utterance()

        self.assertEqual(pending_utterance_count(self.bot), 1)

    def test_a_row_that_has_its_text_is_not_counted(self):
        self._utterance(transcription={"transcript": "done", "words": []})

        self.assertEqual(pending_utterance_count(self.bot), 0)

    def test_a_session_with_nothing_transcribed_yet_reports_them_all(self):
        for _ in range(3):
            self._utterance()
        self._utterance(transcription={"transcript": "done", "words": []})

        self.assertEqual(pending_utterance_count(self.bot), 3)

    def test_another_session_is_not_counted(self):
        """The count gates one session's UI; another user's backlog must not hold it open."""
        other_bot = Bot.objects.create(project=self.project, meeting_url="local_recording")
        other_recording = Recording.objects.create(bot=other_bot, recording_type=1, transcription_type=1, state=RecordingStates.IN_PROGRESS, transcription_provider=ELEVENLABS_PROVIDER)
        Utterance.objects.create(recording=other_recording, participant=self.participant, timestamp_ms=0, duration_ms=UTTERANCE_MS)

        self.assertEqual(pending_utterance_count(self.bot), 0)
