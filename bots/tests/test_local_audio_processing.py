"""Segmentation behaviour for local recording sessions.

These drive the real LocalAudioInputManager rather than a stand-in, so what is pinned is how
production actually cuts utterances -- including its RMS gate and its VAD.

The speech fixture is a phase-continuous 200 Hz tone, chosen by measurement rather than
assumption: webrtcvad classifies it as speech in 200 of 200 frames, whereas a constant
full-scale signal is classified as speech in only 10 of 200. The detector carries hangover
state and settles to "not speech" on a DC signal, which would quietly turn every test built
on one into a test of silence handling instead.
"""

from datetime import datetime, timedelta

import numpy as np
from django.test import TestCase

from bots.local_audio_processing import (
    LOCAL_SILENCE_DURATION_LIMIT_SECONDS,
    LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS,
    LocalAudioInputManager,
    split_local_utterance,
)

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE // 100
FRAME_BYTES = FRAME_SAMPLES * 2
SPEECH_HZ = 200
SPEECH_AMPLITUDE = 8000
SPEAKER = "mic"
EPOCH = datetime(2026, 1, 1)


def speech_frame(index):
    """One 10ms frame of a tone that runs continuously across frames."""
    samples = np.arange(index * FRAME_SAMPLES, (index + 1) * FRAME_SAMPLES)
    return (np.sin(2 * np.pi * SPEECH_HZ * samples / SAMPLE_RATE) * SPEECH_AMPLITUDE).astype(np.int16).tobytes()


def silent_frame(index):
    return b"\x00" * FRAME_BYTES


def timeline(*sections):
    """Build a frame list from (milliseconds, frame_kind) pairs, keeping the tone's phase."""
    frames = []
    for milliseconds, kind in sections:
        for _ in range(milliseconds // FRAME_MS):
            frames.append(kind(len(frames)))
    return frames


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
        feed(build_manager(emitted), timeline((200, speech_frame), (1000, silent_frame), (200, speech_frame)))

        self.assertEqual(emitted, [])

    def test_a_long_pause_still_ends_the_utterance(self):
        """Past the limit an utterance must close, or nothing is ever transcribed."""
        emitted = []
        feed(build_manager(emitted), timeline((200, speech_frame), (2000, silent_frame)))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["flush_reason"], "silence_limit")

    def test_speech_either_side_of_a_short_pause_lands_in_one_utterance(self):
        """The point of the wider limit: one continuous thought, one request."""
        emitted = []
        feed(build_manager(emitted), timeline((200, speech_frame), (1000, silent_frame), (200, speech_frame), (2000, silent_frame)))

        self.assertEqual(len(emitted), 1)
        self.assertGreater(len(emitted[0]["audio_data"]), 1400 * SAMPLE_RATE // 1000 * 2)


class SizeCapSplitTest(TestCase):
    """What happens when somebody talks straight past the size cap."""

    CAP_BYTES = 2 * SAMPLE_RATE * 2  # two seconds

    def _manager_splitting_in_half(self, emitted):
        manager = build_manager(emitted, utterance_size_limit=self.CAP_BYTES)
        manager.split_at_size_limit = lambda audio, rate: len(audio) // 2
        return manager

    def test_the_tail_after_the_split_starts_the_next_utterance(self):
        """The remainder must be carried forward, or the seam is still at the cap.

        Splitting only decides where the FIRST piece ends. If the rest were dropped, the next
        utterance would still begin at the exact cap offset -- mid-word -- which is the whole
        problem this is meant to fix.
        """
        emitted = []
        manager = self._manager_splitting_in_half(emitted)

        feed(manager, timeline((2000, speech_frame)))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["flush_reason"], "buffer_full")
        self.assertEqual(len(emitted[0]["audio_data"]), self.CAP_BYTES // 2)
        self.assertEqual(len(manager.utterances[SPEAKER]), self.CAP_BYTES // 2)

    def test_the_carried_tail_keeps_the_timeline_continuous(self):
        """The next utterance starts where the first one ended, not where the first began."""
        emitted = []
        manager = self._manager_splitting_in_half(emitted)

        feed(manager, timeline((2000, speech_frame)))

        self.assertEqual(manager.first_nonsilent_audio_time[SPEAKER], EPOCH + timedelta(seconds=1))

    def test_the_bot_path_is_untouched_when_no_hook_is_set(self):
        """Default None must behave exactly as before: whole buffer out, nothing carried."""
        emitted = []
        manager = build_manager(emitted, utterance_size_limit=self.CAP_BYTES)
        self.assertIsNone(manager.split_at_size_limit)

        feed(manager, timeline((2000, speech_frame)))

        self.assertEqual(len(emitted[0]["audio_data"]), self.CAP_BYTES)
        self.assertEqual(len(manager.utterances[SPEAKER]), 0)
        self.assertNotIn(SPEAKER, manager.first_nonsilent_audio_time)


class LocalSplitPolicyTest(TestCase):
    """The local session's own rule for where a size-capped utterance may be cut."""

    def test_a_tail_that_would_hold_only_silence_is_not_split_off(self):
        """Carrying dead air forward would open the next utterance on silence.

        The silence timer measures from the last real speech, which sits behind the split, so
        that utterance would flush almost at once as a clip containing nothing but a pause --
        exactly the near-silent input that makes the transcriber invent a line.
        """
        audio = b"".join(timeline((2000, speech_frame), (1000, silent_frame)))

        self.assertEqual(split_local_utterance(audio, SAMPLE_RATE), len(audio))

    def test_a_tail_holding_speech_is_split_off_normally(self):
        audio = b"".join(timeline((2000, speech_frame), (200, silent_frame), (1000, speech_frame)))

        self.assertLess(split_local_utterance(audio, SAMPLE_RATE), len(audio))
