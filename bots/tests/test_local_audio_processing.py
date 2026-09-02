"""Segmentation behaviour for local recording sessions.

These drive the real LocalAudioInputManager rather than a stand-in, so what is pinned is how
production actually cuts utterances. Frames are synthetic: the manager decides silence on RMS
before it consults the VAD, so a loud constant reads as speech and digital zeros read as
silence, which is all these tests need.
"""

from datetime import datetime, timedelta

from django.test import TestCase

from bots.local_audio_processing import (
    LOCAL_SILENCE_DURATION_LIMIT_SECONDS,
    LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS,
    LocalAudioInputManager,
)

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_BYTES = SAMPLE_RATE // 100 * 2
SPEAKER = "mic"
EPOCH = datetime(2026, 1, 1)


def speech_frame():
    """Loud enough to clear the manager's RMS gate, so it counts as speech."""
    return b"\x00\x40" * (FRAME_BYTES // 2)


def silent_frame():
    return b"\x00" * FRAME_BYTES


def frames_for(milliseconds, frame):
    return [frame() for _ in range(milliseconds // FRAME_MS)]


def build_manager(emitted, utterance_size_limit=None):
    return LocalAudioInputManager(
        save_audio_chunk_callback=emitted.append,
        get_participant_callback=lambda _: {"participant_uuid": SPEAKER, "participant_full_name": "You"},
        sample_rate=SAMPLE_RATE,
        utterance_size_limit=utterance_size_limit or LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS * SAMPLE_RATE * 2,
        silence_duration_limit=LOCAL_SILENCE_DURATION_LIMIT_SECONDS,
        should_print_diagnostic_info=False,
    )


def feed(manager, frames):
    for index, frame in enumerate(frames):
        manager.process_chunk(SPEAKER, EPOCH + timedelta(milliseconds=index * FRAME_MS), frame)


class SilenceLimitTest(TestCase):
    """How long a pause has to run before it ends an utterance."""

    def test_a_one_second_thinking_pause_does_not_end_the_utterance(self):
        """A pause of a second is a breath, not a sentence boundary.

        Cutting there splits one thought into fragments that are then transcribed with no
        knowledge of each other, which is what produces half-finished lines.
        """
        emitted = []
        feed(build_manager(emitted), frames_for(200, speech_frame) + frames_for(1000, silent_frame) + frames_for(200, speech_frame))

        self.assertEqual(emitted, [])

    def test_a_long_pause_still_ends_the_utterance(self):
        """Past the limit an utterance must close, or nothing is ever transcribed."""
        emitted = []
        feed(build_manager(emitted), frames_for(200, speech_frame) + frames_for(2000, silent_frame))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["flush_reason"], "silence_limit")

    def test_speech_either_side_of_a_short_pause_lands_in_one_utterance(self):
        """The point of the wider limit: one continuous thought, one request."""
        emitted = []
        manager = build_manager(emitted)
        feed(manager, frames_for(200, speech_frame) + frames_for(1000, silent_frame) + frames_for(200, speech_frame) + frames_for(2000, silent_frame))

        self.assertEqual(len(emitted), 1)
        # 200ms + 1000ms pause + 200ms, plus the silence that triggered the flush.
        self.assertGreater(len(emitted[0]["audio_data"]), 1400 * SAMPLE_RATE // 1000 * 2)
