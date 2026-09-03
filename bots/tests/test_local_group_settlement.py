"""Gathering each drain's utterances into a group and sending it when it is ready.

This is the wiring between the segmenter and the transcriber, and it is where two guarantees the
user can see are actually enforced: a group is never sent twice, and mic and system audio never
share a request -- so "You" and "Others" can never be blended into one set of words.

Uses the real Redis the drain uses; the store round-trip is part of what is being tested.
"""

from datetime import datetime
from unittest import mock

from django.test import TestCase

from bots import local_session_store as store
from bots.local_utterance_group import STOPPED_TALKING_MS
from bots.tasks.process_local_audio_segment_task import _settle_group

DISPATCH_PATH = "bots.tasks.process_local_audio_segment_task.process_local_utterance_group"
EPOCH = datetime(2026, 1, 1)
EPOCH_MS = int(EPOCH.timestamp() * 1000)
BOT_ID = 987654


class FakeManager:
    def __init__(self, members):
        self.group_members = members


def member(utterance_id, offset_ms, duration_ms=1000, voice_ms=1000):
    return {
        "utterance_id": utterance_id,
        "voice_ms": voice_ms,
        "duration_ms": duration_ms,
        "timestamp_ms": EPOCH_MS + offset_ms,
        "flush_reason": "silence_limit",
    }


class SettleGroupTest(TestCase):
    def setUp(self):
        self.client = store.redis_client()
        store.clear_session_state(BOT_ID)
        self.addCleanup(store.clear_session_state, BOT_ID)

    def _settle(self, source, members, end_offset_ms, session_ended=False):
        with mock.patch(DISPATCH_PATH) as dispatch:
            _settle_group(self.client, BOT_ID, source, FakeManager(members), EPOCH, end_offset_ms, session_ended)
        return dispatch

    def test_a_group_that_is_not_ready_is_held_for_the_next_drain(self):
        """Still talking: keep gathering rather than send a fragment."""
        dispatch = self._settle("mic", [member(1, 0)], end_offset_ms=1200)

        dispatch.delay.assert_not_called()
        self.assertEqual(len(store.load_group(self.client, BOT_ID, "mic")), 1)

    def test_what_an_earlier_drain_held_is_carried_forward(self):
        self._settle("mic", [member(1, 0)], end_offset_ms=1200)
        self._settle("mic", [member(2, 1500)], end_offset_ms=2600)

        held = store.load_group(self.client, BOT_ID, "mic")
        self.assertEqual([entry["utterance_id"] for entry in held], [1, 2])

    def test_a_speaker_who_stops_has_the_group_sent(self):
        dispatch = self._settle("mic", [member(1, 0)], end_offset_ms=1000 + STOPPED_TALKING_MS)

        self.assertEqual(dispatch.delay.call_count, 1)
        self.assertEqual(dispatch.delay.call_args.args[0], [1])

    def test_a_sent_group_is_cleared_so_the_next_drain_cannot_send_it_again(self):
        """Double-sending would transcribe and charge for the same audio twice.

        The held group must be non-empty FIRST -- clearing an already-empty key proves nothing,
        which is exactly what an earlier version of this test did.
        """
        self._settle("mic", [member(1, 0)], end_offset_ms=1200)
        self.assertEqual(len(store.load_group(self.client, BOT_ID, "mic")), 1, "precondition: something is held")

        self._settle("mic", [], end_offset_ms=1000 + STOPPED_TALKING_MS)

        self.assertEqual(store.load_group(self.client, BOT_ID, "mic"), [])
        second = self._settle("mic", [], end_offset_ms=1000 + STOPPED_TALKING_MS)
        second.delay.assert_not_called()

    def test_the_gaps_sent_match_the_members_positions(self):
        """The task splits words by these gaps, so they must describe the same audio."""
        dispatch = self._settle(
            "mic",
            [member(1, 0), member(2, 1400)],
            end_offset_ms=2400 + STOPPED_TALKING_MS,
        )

        self.assertEqual(dispatch.delay.call_args.args[1], [400])

    def test_session_end_sends_whatever_is_held(self):
        """Nothing may be stranded when the recording stops."""
        dispatch = self._settle("mic", [member(1, 0)], end_offset_ms=1100, session_ended=True)

        self.assertEqual(dispatch.delay.call_count, 1)

    def test_session_end_with_nothing_held_sends_nothing(self):
        dispatch = self._settle("mic", [], end_offset_ms=0, session_ended=True)

        dispatch.delay.assert_not_called()

    def test_mic_and_system_never_share_a_group(self):
        """The You/Others split depends on this: one request may only ever hold one voice."""
        self._settle("mic", [member(1, 0)], end_offset_ms=1200)
        self._settle("system", [member(2, 0)], end_offset_ms=1200)

        self.assertEqual([entry["utterance_id"] for entry in store.load_group(self.client, BOT_ID, "mic")], [1])
        self.assertEqual([entry["utterance_id"] for entry in store.load_group(self.client, BOT_ID, "system")], [2])

    def test_sending_one_sources_group_leaves_the_other_untouched(self):
        self._settle("system", [member(2, 0)], end_offset_ms=1200)
        self._settle("mic", [member(1, 0)], end_offset_ms=1000 + STOPPED_TALKING_MS)

        self.assertEqual(store.load_group(self.client, BOT_ID, "mic"), [])
        self.assertEqual(len(store.load_group(self.client, BOT_ID, "system")), 1)
