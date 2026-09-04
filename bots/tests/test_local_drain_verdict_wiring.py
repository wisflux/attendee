"""_drain threading the verdict cache between Redis and the manager.

The manager-level mechanics (a cached verdict skips the detector, an exhausted cache falls
through to a live call) are already proven in test_local_audio_processing_verdict_reuse.py.
What is proven here is the GLUE: that a drain reads the tail's saved verdicts and hands them to
build_manager, and that whatever the manager still has buffered at the end gets saved back --
using a real Bot/Recording/Participant, exactly as production does, with build_manager and feed
replaced by fakes so no real audio or model is needed to prove the wiring is correct.
"""

from unittest import mock

from django.test import TestCase

from bots import local_session_store as store
from bots.models import Bot, Organization, Participant, Project, Recording, RecordingStates, SessionTypes
from bots.tasks.process_local_audio_segment_task import _drain

BUILD_MANAGER_PATH = "bots.tasks.process_local_audio_segment_task.build_manager"
FEED_PATH = "bots.tasks.process_local_audio_segment_task.feed"
SETTLE_GROUP_PATH = "bots.tasks.process_local_audio_segment_task._settle_group"
SOURCE = "mic"
SAMPLE_RATE = 16000
FRAME_BYTES = 320  # 10ms at 16kHz, 16-bit mono


class FakeManager:
    """Just enough surface for _drain to call, with a scriptable buffered_verdicts result."""

    def __init__(self, still_buffered_bytes=b"", verdicts_to_carry=None):
        self.group_members = []
        self.utterances = {SOURCE: still_buffered_bytes} if still_buffered_bytes else {}
        self.first_nonsilent_audio_time = {}
        self._verdicts_to_carry = verdicts_to_carry or []

    def buffered_verdicts(self, source):
        return self._verdicts_to_carry

    def export_vad_state(self):
        return {"stub": True}


class DrainVerdictWiringTest(TestCase):
    def setUp(self):
        org = Organization.objects.create(name="Org")
        project = Project.objects.create(name="Proj", organization=org)
        self.bot = Bot.objects.create(project=project, meeting_url="local_recording", session_type=SessionTypes.LOCAL)
        self.recording = Recording.objects.create(bot=self.bot, recording_type=1, transcription_type=1, state=RecordingStates.IN_PROGRESS, is_default_recording=True)
        Participant.objects.create(bot=self.bot, uuid=SOURCE)
        self.client = store.redis_client()
        store.clear_session_state(self.bot.id)
        self.addCleanup(store.clear_session_state, self.bot.id)

    def _enqueue_one_segment(self):
        store.enqueue_segment(self.bot.id, SOURCE, sequence=1, audio=b"\x00" * FRAME_BYTES, sample_rate=SAMPLE_RATE, offset_ms=0)

    def test_the_tails_saved_verdicts_reach_build_manager(self):
        store.save_tail(self.client, self.bot.id, SOURCE, b"\x00" * (FRAME_BYTES * 2), 0, 0, -1, SAMPLE_RATE, verdicts=[True, False])
        self._enqueue_one_segment()

        with mock.patch(BUILD_MANAGER_PATH, return_value=FakeManager()) as build, mock.patch(FEED_PATH, return_value=0), mock.patch(SETTLE_GROUP_PATH):
            _drain(self.client, self.bot.id, SOURCE, is_final=False)

        self.assertEqual(build.call_args.kwargs["cached_verdicts"], [True, False])

    def test_a_tail_with_no_verdicts_yet_passes_an_empty_cache(self):
        self._enqueue_one_segment()

        with mock.patch(BUILD_MANAGER_PATH, return_value=FakeManager()) as build, mock.patch(FEED_PATH, return_value=0), mock.patch(SETTLE_GROUP_PATH):
            _drain(self.client, self.bot.id, SOURCE, is_final=False)

        self.assertEqual(build.call_args.kwargs["cached_verdicts"], [])

    def test_a_verdict_cache_longer_than_its_own_audio_is_not_trusted(self):
        """Corrupted or pre-this-feature Redis data must fall back to deciding live, not crash
        and not silently mis-score real audio with verdicts that do not belong to it."""
        store.save_tail(self.client, self.bot.id, SOURCE, b"\x00" * FRAME_BYTES, 0, 0, -1, SAMPLE_RATE, verdicts=[True, True, True])
        self._enqueue_one_segment()

        with mock.patch(BUILD_MANAGER_PATH, return_value=FakeManager()) as build, mock.patch(FEED_PATH, return_value=0), mock.patch(SETTLE_GROUP_PATH):
            _drain(self.client, self.bot.id, SOURCE, is_final=False)

        self.assertEqual(build.call_args.kwargs["cached_verdicts"], [])

    def test_whatever_is_still_buffered_is_saved_for_the_next_drain(self):
        self._enqueue_one_segment()
        manager = FakeManager(still_buffered_bytes=b"\x00" * FRAME_BYTES, verdicts_to_carry=[False])

        with mock.patch(BUILD_MANAGER_PATH, return_value=manager), mock.patch(FEED_PATH, return_value=FRAME_BYTES), mock.patch(SETTLE_GROUP_PATH):
            _drain(self.client, self.bot.id, SOURCE, is_final=False)

        self.assertEqual(store.load_tail(self.client, self.bot.id, SOURCE)["verdicts"], [False])

    def test_a_final_drain_saves_an_empty_cache_alongside_its_emptied_audio(self):
        self._enqueue_one_segment()
        manager = FakeManager()

        with mock.patch(BUILD_MANAGER_PATH, return_value=manager), mock.patch(FEED_PATH, return_value=0), mock.patch(SETTLE_GROUP_PATH), mock.patch("bots.tasks.process_local_audio_segment_task.flush_remaining"):
            _drain(self.client, self.bot.id, SOURCE, is_final=True)

        self.assertEqual(store.load_tail(self.client, self.bot.id, SOURCE)["verdicts"], [])
