"""Putting one request's words back onto the rows they came from.

A group is transcribed as a single audio file, so every word comes back timed against that file
rather than against the utterance it belongs to. This is where each word is handed to its row and
its timings rebased.

NO WORD MAY BE LOST. The rule is that a word belongs to the utterance whose window its START
falls in, and every instant between one window's start and the next belongs to somebody -- so
there is no position a word can occupy and be discarded. A boundary-straddling word used to be
written to the log and thrown away, which was survivable only because the fixed 3s spacer meant
no word ever spanned a join; with real gaps it would happen constantly.
"""

from django.test import TestCase

from bots.transcription_utils import get_mp3_for_utterance_group, split_transcription_by_utterance


class FakeUtterance:
    """Only what the splitter reads: an id and how long the audio is."""

    def __init__(self, utterance_id, duration_ms):
        self.id = utterance_id
        self.duration_ms = duration_ms


def word(text, start, end):
    return {"word": text, "start": start, "end": end}


def result(*words):
    return {"language": "en", "words": list(words)}


class SplitTranscriptionByUtteranceTest(TestCase):
    # Two 1s utterances joined by a 0.5s gap: windows are 0.0-1.0 and 1.5-2.5.
    GAPS = [0.5]

    def _two_utterances(self):
        return [FakeUtterance(1, 1000), FakeUtterance(2, 1000)]

    def test_each_word_lands_on_the_utterance_it_was_spoken_in(self):
        utterances = self._two_utterances()
        transcription = result(word("hello", 0.1, 0.4), word("world", 1.6, 1.9))

        split = split_transcription_by_utterance(transcription, utterances, gaps_seconds=self.GAPS)

        self.assertEqual(split[1]["transcript"], "hello")
        self.assertEqual(split[2]["transcript"], "world")

    def test_word_times_are_rebased_onto_their_own_utterance(self):
        """A row's words must read from the start of that row, not of the joined file."""
        utterances = self._two_utterances()
        transcription = result(word("world", 1.6, 1.9))

        split = split_transcription_by_utterance(transcription, utterances, gaps_seconds=self.GAPS)

        self.assertAlmostEqual(split[2]["words"][0]["start"], 0.1)

    def test_a_word_straddling_a_boundary_is_kept(self):
        """The bug this fixes: it was logged and dropped, so the word vanished from both rows."""
        utterances = self._two_utterances()
        transcription = result(word("straddle", 0.9, 1.7))

        split = split_transcription_by_utterance(transcription, utterances, gaps_seconds=self.GAPS)

        self.assertEqual(split[1]["transcript"] + split[2]["transcript"], "straddle")

    def test_a_straddling_word_is_not_written_to_both_rows(self):
        utterances = self._two_utterances()
        transcription = result(word("straddle", 0.9, 1.7))

        split = split_transcription_by_utterance(transcription, utterances, gaps_seconds=self.GAPS)

        appearances = [row for row in split.values() if "straddle" in row["transcript"]]
        self.assertEqual(len(appearances), 1)

    def test_a_word_falling_inside_the_gap_goes_to_the_utterance_before_it(self):
        """Timings drift by a few ms; a word must never land in the silence and be lost."""
        utterances = self._two_utterances()
        transcription = result(word("edge", 1.1, 1.3))

        split = split_transcription_by_utterance(transcription, utterances, gaps_seconds=self.GAPS)

        self.assertEqual(split[1]["transcript"], "edge")

    def test_no_word_is_ever_dropped(self):
        """The property that matters most: every word the model returned reaches some row."""
        utterances = self._two_utterances()
        spoken = [word("a", 0.1, 0.3), word("b", 0.9, 1.7), word("c", 1.2, 1.4), word("d", 2.0, 2.4)]

        split = split_transcription_by_utterance(result(*spoken), utterances, gaps_seconds=self.GAPS)

        landed = sum(len(row["words"]) for row in split.values())
        self.assertEqual(landed, len(spoken))

    def test_the_gaps_that_built_the_audio_are_the_gaps_used_to_read_it(self):
        """With a wider gap the second window moves later, and the same word belongs elsewhere."""
        utterances = self._two_utterances()
        transcription = result(word("late", 1.6, 1.9))

        split = split_transcription_by_utterance(transcription, utterances, gaps_seconds=[2.0])

        self.assertEqual(split[1]["transcript"], "late", "with a 2s gap, 1.6s is still the first utterance's tail")
        self.assertEqual(split[2]["transcript"], "")

    def test_the_old_fixed_spacer_still_applies_when_no_gaps_are_given(self):
        """The async transcription path passes no gaps and must behave exactly as before."""
        utterances = self._two_utterances()
        transcription = result(word("hello", 0.1, 0.4), word("world", 4.1, 4.4))

        split = split_transcription_by_utterance(transcription, utterances)

        self.assertEqual(split[1]["transcript"], "hello")
        self.assertEqual(split[2]["transcript"], "world")


class ConcatenateWithRealGapsTest(TestCase):
    """The audio side of the same contract: the gaps asked for are the gaps written.

    Runs the real ffmpeg the production path uses -- a mocked encoder would prove nothing about
    whether the silence actually lands in the file.
    """

    SAMPLE_RATE = 16000

    class Audible(FakeUtterance):
        def __init__(self, utterance_id, duration_ms, sample_rate):
            super().__init__(utterance_id, duration_ms)
            self._sample_rate = sample_rate

        def get_sample_rate(self):
            return self._sample_rate

        def get_audio_blob(self):
            return b"\x01\x00" * (self.duration_ms * self._sample_rate // 1000)

    def _pair(self):
        return [self.Audible(1, 500, self.SAMPLE_RATE), self.Audible(2, 500, self.SAMPLE_RATE)]

    def test_a_wider_gap_produces_a_longer_file(self):
        short = get_mp3_for_utterance_group(self._pair(), sample_rate=self.SAMPLE_RATE, gaps_seconds=[0.1])
        wide = get_mp3_for_utterance_group(self._pair(), sample_rate=self.SAMPLE_RATE, gaps_seconds=[2.0])

        self.assertGreater(len(wide), len(short))

    def test_each_gap_is_written_independently(self):
        """Three utterances, two different gaps -- not one spacer applied twice."""
        three = self._pair() + [self.Audible(3, 500, self.SAMPLE_RATE)]

        uneven = get_mp3_for_utterance_group(three, sample_rate=self.SAMPLE_RATE, gaps_seconds=[0.1, 2.0])
        even = get_mp3_for_utterance_group(three, sample_rate=self.SAMPLE_RATE, gaps_seconds=[0.1, 0.1])

        self.assertGreater(len(uneven), len(even))

    def test_the_fixed_spacer_is_still_used_when_no_gaps_are_given(self):
        """The async transcription path passes no gaps and must be byte-identical."""
        with_default = get_mp3_for_utterance_group(self._pair(), sample_rate=self.SAMPLE_RATE)
        with_explicit = get_mp3_for_utterance_group(self._pair(), sample_rate=self.SAMPLE_RATE, gaps_seconds=[3.0])

        self.assertEqual(len(with_default), len(with_explicit))
