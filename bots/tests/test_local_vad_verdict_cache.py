"""Reusing an already-decided speech/silence verdict instead of asking the model again.

Every drain glues the still-open utterance's bytes onto the front of whatever's new and replays
the whole thing through the VAD -- meaning the SAME seconds get judged again on every drain an
utterance stays open. Measured on a realistic 30s sentence built up one second at a time: the
model runs 465 times for 30 seconds of audio, and the first second is re-judged roughly 30 times
by the time the sentence closes. That is wasted work, and worse, it tells a RECURRENT model it
just heard the same audio a second time, which is not the input it was designed for.

The verdict for a given frame, once decided, is a fixed fact -- nothing about it changes on a
later drain. So it only needs deciding once.
"""

from django.test import TestCase

from bots.local_vad_verdict_cache import VerdictCache, safe_cached_verdicts, verdicts_for_buffered


class VerdictCacheTest(TestCase):
    """Hands back cached verdicts in order, then defers to a live call once exhausted."""

    def test_a_cached_verdict_is_returned_without_calling_the_live_check(self):
        cache = VerdictCache([True, False])
        live = []

        result = cache.next(lambda: live.append("called") or True)

        self.assertTrue(result)
        self.assertEqual(live, [], "the live check must not run while a cached verdict exists")

    def test_cached_verdicts_are_consumed_in_order(self):
        cache = VerdictCache([True, False, True])

        self.assertEqual([cache.next(lambda: None) for _ in range(3)], [True, False, True])

    def test_once_exhausted_every_further_call_is_live(self):
        cache = VerdictCache([True])
        live_calls = []

        cache.next(lambda: True)  # consumes the one cached entry
        result = cache.next(lambda: live_calls.append("called") or False)

        self.assertFalse(result)
        self.assertEqual(live_calls, ["called"])

    def test_an_empty_cache_is_always_live(self):
        cache = VerdictCache([])
        live_calls = []

        cache.next(lambda: live_calls.append("called") or True)

        self.assertEqual(live_calls, ["called"])

    def test_no_cache_at_all_behaves_like_an_empty_one(self):
        cache = VerdictCache(None)
        live_calls = []

        cache.next(lambda: live_calls.append("called") or True)

        self.assertEqual(live_calls, ["called"])


class VerdictsForBufferedTest(TestCase):
    """Which recorded verdicts belong to whatever is still buffered, unflushed."""

    def test_the_trailing_slice_matching_the_buffer_is_returned(self):
        recorded = [False, False, True, True, True]  # 2 dropped before the buffer opened, then 3 buffered

        self.assertEqual(verdicts_for_buffered(recorded, buffered_frame_count=3), [True, True, True])

    def test_nothing_buffered_means_nothing_carried(self):
        self.assertEqual(verdicts_for_buffered([True, True], buffered_frame_count=0), [])

    def test_an_empty_recording_carries_nothing(self):
        self.assertEqual(verdicts_for_buffered([], buffered_frame_count=0), [])


class SafeCachedVerdictsTest(TestCase):
    """A cache is only trusted when it cannot possibly outrun the audio it is meant to cover.

    A cache longer than the audio's own frame count would bleed cached (and therefore wrong)
    verdicts onto genuinely new audio -- corrupted Redis data, or an older tail saved before this
    existed, must degrade to "no cache" rather than silently mis-score real frames.
    """

    FRAME_BYTES = 320  # 10ms at 16kHz, 16-bit mono

    def test_a_cache_no_longer_than_the_audio_is_trusted(self):
        audio = b"\x00" * (self.FRAME_BYTES * 5)

        self.assertEqual(safe_cached_verdicts([True] * 5, audio, self.FRAME_BYTES), [True] * 5)

    def test_a_cache_longer_than_the_audio_is_discarded(self):
        audio = b"\x00" * (self.FRAME_BYTES * 3)

        with self.assertLogs("bots.local_vad_verdict_cache", level="WARNING"):
            result = safe_cached_verdicts([True] * 5, audio, self.FRAME_BYTES)

        self.assertEqual(result, [])

    def test_no_cache_is_simply_no_cache(self):
        audio = b"\x00" * (self.FRAME_BYTES * 3)

        self.assertEqual(safe_cached_verdicts([], audio, self.FRAME_BYTES), [])
