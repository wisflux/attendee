"""Dropping the silence that closed an utterance, before it is sent for transcription.

Every local utterance ends with the pause that triggered its flush -- that is how the manager
knows a sentence finished. Measured on this pipeline that dead air is roughly 30% of a typical
clip and up to 80% of a short one, and a transcription model handed mostly-silence fills it
with invented text rather than returning nothing.

Helpers are imported from test_local_audio_processing rather than repeated, so both suites
drive the manager exactly the same way.
"""

from django.test import TestCase

from bots.local_audio_processing import (
    TRAILING_SILENCE_PAD_MS,
    frames_without_trailing_silence,
)
from bots.tests.test_local_audio_processing import (
    FRAME_MS,
    SAMPLE_RATE,
    build_manager,
    feed,
    silent_frame,
    speech_frame,
    timeline,
    verdicts_for,
)

PAD_FRAMES = TRAILING_SILENCE_PAD_MS // FRAME_MS


def bytes_for_ms(milliseconds):
    return milliseconds * SAMPLE_RATE // 1000 * 2


class TrailingSilenceTrimTest(TestCase):
    """The pure decision: how many frames of an utterance are worth sending."""

    def test_the_silence_that_closed_the_utterance_is_dropped(self):
        verdicts = [True] * 20 + [False] * 150

        self.assertEqual(frames_without_trailing_silence(verdicts), 20 + PAD_FRAMES)

    def test_a_pause_inside_the_utterance_is_not_touched(self):
        """Only the run at the very end goes. An internal pause is sentence structure."""
        verdicts = [True] * 20 + [False] * 30 + [True] * 10 + [False] * 150

        self.assertEqual(frames_without_trailing_silence(verdicts), 60 + PAD_FRAMES)

    def test_an_utterance_still_in_speech_is_left_whole(self):
        """A size-capped flush cuts while somebody is still talking; nothing to trim."""
        verdicts = [True] * 50

        self.assertEqual(frames_without_trailing_silence(verdicts), 50)

    def test_trailing_silence_shorter_than_the_pad_is_left_whole(self):
        """The pad is a ceiling, never an extension past the audio we actually hold."""
        verdicts = [True] * 20 + [False] * 5

        self.assertEqual(frames_without_trailing_silence(verdicts), 25)

    def test_a_clip_with_no_speech_is_left_whole(self):
        """Never return zero. An empty utterance would be emitted with no audio at all."""
        verdicts = [False] * 50

        self.assertEqual(frames_without_trailing_silence(verdicts), 50)

    def test_no_verdicts_leaves_nothing_to_decide(self):
        self.assertEqual(frames_without_trailing_silence([]), 0)


class EmittedUtteranceTest(TestCase):
    """What the manager actually hands to the transcriber."""

    def test_the_emitted_utterance_does_not_carry_the_silence_that_closed_it(self):
        """200ms of speech must not be sent as a 1.7s clip that is 88% dead air."""
        emitted = []
        sections = ((200, speech_frame), (2000, silent_frame))
        feed(build_manager(emitted, verdicts=verdicts_for(*sections)), timeline(*sections))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0]["audio_data"]), bytes_for_ms(200 + TRAILING_SILENCE_PAD_MS))

    def test_the_final_word_keeps_its_decay(self):
        """Cutting at the last speech frame would clip the word's own tail."""
        emitted = []
        sections = ((200, speech_frame), (2000, silent_frame))
        feed(build_manager(emitted, verdicts=verdicts_for(*sections)), timeline(*sections))

        self.assertGreater(len(emitted[0]["audio_data"]), bytes_for_ms(200))


class VoiceMeasurementTest(TestCase):
    """The group needs to know how much real speech a clip carries, not just how long it is."""

    def test_the_emitted_message_reports_how_much_of_it_was_speech(self):
        emitted = []
        sections = ((200, speech_frame), (2000, silent_frame))
        feed(build_manager(emitted, verdicts=verdicts_for(*sections)), timeline(*sections))

        self.assertEqual(emitted[0]["voice_ms"], 200)

    def test_a_clip_broken_by_a_short_pause_counts_only_the_speech(self):
        """Otherwise a group of pauses would look like a group full of conversation."""
        emitted = []
        sections = ((200, speech_frame), (1000, silent_frame), (300, speech_frame), (2000, silent_frame))
        feed(build_manager(emitted, verdicts=verdicts_for(*sections)), timeline(*sections))

        self.assertEqual(emitted[0]["voice_ms"], 500)
