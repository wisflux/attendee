"""Choosing where to cut a long utterance."""

import numpy as np
from django.test import SimpleTestCase

from bots.audio_split import BYTES_PER_SAMPLE, quietest_split_point

SAMPLE_RATE = 16000


def pcm(*sections):
    return np.concatenate(sections).astype(np.int16).tobytes()


def loud(seconds):
    return np.full(int(SAMPLE_RATE * seconds), 8000.0)


def quiet(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds))


def seconds_at(byte_offset):
    return byte_offset / BYTES_PER_SAMPLE / SAMPLE_RATE


class QuietestSplitPointTest(SimpleTestCase):
    def test_splits_inside_the_gap_rather_than_at_the_very_end(self):
        """28s of speech, a 400ms gap, then a last second: the cut belongs in the gap."""
        audio = pcm(loud(28), quiet(0.4), loud(1))

        at = seconds_at(quietest_split_point(audio, SAMPLE_RATE))

        self.assertGreater(at, 28.0)
        self.assertLess(at, 28.5)

    def test_a_gap_older_than_the_search_window_is_not_a_candidate(self):
        """Only the tail is searched. A gap further back is not reachable, and the caller
        falls back to cutting at the cap rather than discarding seconds of buffered speech."""
        audio = pcm(loud(28), quiet(0.4), loud(3))

        self.assertEqual(quietest_split_point(audio, SAMPLE_RATE, search_ms=3000), len(audio))

    def test_returns_the_end_when_the_speaker_talks_straight_through(self):
        """No gap to find, so behave exactly as the caller did before: cut at the cap."""
        audio = pcm(loud(31))

        self.assertEqual(quietest_split_point(audio, SAMPLE_RATE), len(audio))

    def test_the_offset_is_always_a_whole_sample(self):
        """An odd byte count is not decodable PCM16."""
        audio = pcm(loud(28), quiet(0.4), loud(3))

        self.assertEqual(quietest_split_point(audio, SAMPLE_RATE) % BYTES_PER_SAMPLE, 0)

    def test_the_offset_is_never_zero(self):
        """Zero would emit an empty utterance and leave the buffer unchanged, forever."""
        audio = pcm(quiet(31))

        self.assertGreater(quietest_split_point(audio, SAMPLE_RATE), 0)

    def test_audio_shorter_than_the_search_window_is_returned_whole(self):
        audio = pcm(loud(0.05))

        self.assertEqual(quietest_split_point(audio, SAMPLE_RATE), len(audio))

    def test_empty_audio_is_returned_whole(self):
        self.assertEqual(quietest_split_point(b"", SAMPLE_RATE), 0)

    def test_only_the_tail_is_searched(self):
        """A quiet stretch early on is not a candidate -- cutting there would throw away
        several seconds of speech that had already been buffered."""
        audio = pcm(loud(2), quiet(3), loud(25))

        at = seconds_at(quietest_split_point(audio, SAMPLE_RATE, search_ms=3000))

        self.assertGreater(at, 5.0)
