"""The tail's verdict cache round-tripping through Redis.

Only the new `verdicts` field is covered here -- the rest of load_tail/save_tail's shape is
exercised indirectly by test_local_group_settlement.py.
"""

from django.test import TestCase

from bots import local_session_store as store

BOT_ID = 555111
SOURCE = "mic"


class TailVerdictsTest(TestCase):
    def setUp(self):
        self.client = store.redis_client()
        store.clear_session_state(BOT_ID)
        self.addCleanup(store.clear_session_state, BOT_ID)

    def test_saved_verdicts_round_trip(self):
        store.save_tail(self.client, BOT_ID, SOURCE, b"\x00" * 320, 0, 10, 1, 16000, verdicts=[True, False])

        tail = store.load_tail(self.client, BOT_ID, SOURCE)

        self.assertEqual(tail["verdicts"], [True, False])

    def test_a_tail_with_no_verdicts_yet_loads_as_empty(self):
        store.save_tail(self.client, BOT_ID, SOURCE, b"", None, None, 1, 16000)

        tail = store.load_tail(self.client, BOT_ID, SOURCE)

        self.assertEqual(tail["verdicts"], [])

    def test_no_tail_at_all_has_no_verdicts(self):
        tail = store.load_tail(self.client, BOT_ID, SOURCE)

        self.assertEqual(tail["verdicts"], [])

    def test_a_tail_saved_before_this_field_existed_loads_as_empty(self):
        """Backward compatibility for a tail written by a worker running the old code."""
        import json

        old_shape = {"audio": "", "started_offset_ms": None, "end_offset_ms": None, "last_sequence": 1, "sample_rate": 16000, "vad_state": None}
        self.client.set(store.tail_key(BOT_ID, SOURCE), json.dumps(old_shape))

        tail = store.load_tail(self.client, BOT_ID, SOURCE)

        self.assertEqual(tail["verdicts"], [])
