"""Segmentation behaviour for local recording sessions.

These drive the real LocalAudioInputManager rather than a stand-in, so what is pinned is how
production actually cuts utterances -- including its RMS gate and its VAD.

Speech is SCRIPTED, not synthesised. Silero rejects every artificial signal -- pure tones,
white noise, formant stacks all score 0 of 100 -- which is correct behaviour and exactly why
it beats an amplitude gate, but it makes synthetic audio useless for driving the manager.
So these tests inject a detector with scripted verdicts and check the manager's own logic:
when an utterance opens, when it closes, what gets carried forward. Silero's accuracy against
real speech is measured separately by bots/e2e_tests/vad_report.py.
"""

from datetime import datetime, timedelta

import numpy as np
from django.test import TestCase

from bots.local_audio_processing import (
    LOCAL_SILENCE_DURATION_LIMIT_SECONDS,
    LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS,
    TRIM_SILENCE_OVER_MS,
    LocalAudioInputManager,
    shorten_long_silences,
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
    """Frames from (milliseconds, kind) pairs."""
    frames = []
    for milliseconds, kind in sections:
        for _ in range(milliseconds // FRAME_MS):
            frames.append(kind(len(frames)))
    return frames


def verdicts_for(*sections):
    """The speech/silence script matching the same (milliseconds, kind) pairs."""
    out = []
    for milliseconds, kind in sections:
        out.extend([kind is speech_frame] * (milliseconds // FRAME_MS))
    return out


class ScriptedDetector:
    """Says speech for frames whose index is marked True; state export is a stub."""

    def __init__(self, verdicts):
        self._verdicts = verdicts
        self._index = 0

    def is_speech(self, chunk_bytes):
        verdict = self._verdicts[min(self._index, len(self._verdicts) - 1)]
        self._index += 1
        return verdict

    def export_state(self):
        return {"scripted": True}


def build_manager(emitted, utterance_size_limit=None, verdicts=None, cached_verdicts=None):
    return LocalAudioInputManager(
        detector=ScriptedDetector(verdicts if verdicts is not None else [True] * 100000),
        cached_verdicts=cached_verdicts,
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
        sections = ((200, speech_frame), (1000, silent_frame), (200, speech_frame))
        feed(build_manager(emitted, verdicts=verdicts_for(*sections)), timeline(*sections))

        self.assertEqual(emitted, [])

    def test_a_long_pause_still_ends_the_utterance(self):
        """Past the limit an utterance must close, or nothing is ever transcribed."""
        emitted = []
        sections = ((200, speech_frame), (2000, silent_frame))
        feed(build_manager(emitted, verdicts=verdicts_for(*sections)), timeline(*sections))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["flush_reason"], "silence_limit")

    def test_speech_either_side_of_a_short_pause_lands_in_one_utterance(self):
        """The point of the wider limit: one continuous thought, one request."""
        emitted = []
        sections = ((200, speech_frame), (1000, silent_frame), (200, speech_frame), (2000, silent_frame))
        feed(build_manager(emitted, verdicts=verdicts_for(*sections)), timeline(*sections))

        self.assertEqual(len(emitted), 1)
        self.assertGreater(len(emitted[0]["audio_data"]), 1400 * SAMPLE_RATE // 1000 * 2)


class SizeCapSplitTest(TestCase):
    """What happens when somebody talks straight past the size cap."""

    CAP_BYTES = 2 * SAMPLE_RATE * 2  # two seconds

    def _manager_splitting_in_half(self, emitted):
        manager = build_manager(emitted, utterance_size_limit=self.CAP_BYTES, verdicts=[True] * 100000)
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
        manager = build_manager(emitted, utterance_size_limit=self.CAP_BYTES, verdicts=[True] * 100000)
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


class SileroStateTest(TestCase):
    """Silero is recurrent: the drain task must hand its memory to the next drain."""

    def test_state_survives_a_round_trip(self):
        """A resumed detector continues mid-stream instead of restarting cold.

        The drain runs as a Celery task and keeps nothing between runs, so without this the
        model is rebuilt roughly once a second and never leaves its warm-up regime -- worth
        about 13 seconds of false silence on a 90 second recording.
        """
        from bots.local_silero_vad import LocalSileroVad

        frames = timeline((600, speech_frame))
        continuous = LocalSileroVad(SAMPLE_RATE)
        for frame in frames:
            continuous.is_speech(frame)

        first = LocalSileroVad(SAMPLE_RATE)
        for frame in frames[:30]:
            first.is_speech(frame)
        resumed = LocalSileroVad(SAMPLE_RATE, state=first.export_state())
        for frame in frames[30:]:
            verdict = resumed.is_speech(frame)

        self.assertEqual(verdict, continuous.is_speech(frames[-1]))

    def test_unusable_state_starts_cold_rather_than_failing(self):
        """A stale or corrupt blob must not take a session down with it."""
        from bots.local_silero_vad import LocalSileroVad

        detector = LocalSileroVad(SAMPLE_RATE, state={"state": "not-hex"})

        self.assertFalse(detector.is_speech(silent_frame(0)))

    def test_the_manager_exports_state_the_store_can_carry(self):
        from bots.local_silero_vad import STATE_SHAPE

        # A real Silero detector, not the scripted one -- this asserts the shape the Redis
        # tail has to carry.
        manager = LocalAudioInputManager(
            save_audio_chunk_callback=lambda _: None,
            get_participant_callback=lambda _: {"participant_uuid": SPEAKER, "participant_full_name": "You"},
            sample_rate=SAMPLE_RATE,
            utterance_size_limit=LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS * SAMPLE_RATE * 2,
            silence_duration_limit=LOCAL_SILENCE_DURATION_LIMIT_SECONDS,
            should_print_diagnostic_info=False,
        )
        feed(manager, timeline((200, speech_frame)))
        exported = manager.export_vad_state()

        self.assertEqual(set(exported), {"state", "context", "buffer", "remainder", "speaking"})
        self.assertEqual(len(bytes.fromhex(exported["state"])), 4 * STATE_SHAPE[0] * STATE_SHAPE[1] * STATE_SHAPE[2])


class ShortenLongSilencesTest(TestCase):
    """Long pauses are shortened, never removed, and only in the middle."""

    def frames_of(self, *sections):
        return b"".join(timeline(*sections)), verdicts_for(*sections)

    def test_a_long_silence_is_shortened_not_removed(self):
        """The transcriber reads a pause as sentence structure. Deleting one costs content."""
        audio, verdicts = self.frames_of((1000, speech_frame), (4000, silent_frame), (1000, speech_frame))

        out = shorten_long_silences(audio, verdicts, SAMPLE_RATE)

        self.assertLess(len(out), len(audio))
        self.assertGreater(len(out), 2000 * SAMPLE_RATE // 1000 * 2)  # both speech runs survive

    def test_a_short_pause_is_left_exactly_alone(self):
        """Short gaps sit inside and between words; touching them corrupts the speech."""
        audio, verdicts = self.frames_of((1000, speech_frame), (800, silent_frame), (1000, speech_frame))

        self.assertEqual(shorten_long_silences(audio, verdicts, SAMPLE_RATE), audio)

    def test_the_edges_of_a_long_silence_are_kept(self):
        """Only the middle goes, so the splice is silence-to-silence and clips no word."""
        audio, verdicts = self.frames_of((1000, speech_frame), (4000, silent_frame), (1000, speech_frame))

        out = shorten_long_silences(audio, verdicts, SAMPLE_RATE)

        remaining_silence = len(out) - 2000 * SAMPLE_RATE // 1000 * 2
        self.assertGreater(remaining_silence, 0)

    def test_audio_with_no_verdicts_is_returned_untouched(self):
        audio = b"".join(timeline((500, speech_frame)))

        self.assertEqual(shorten_long_silences(audio, [], SAMPLE_RATE), audio)

    def test_nothing_is_shortened_at_the_current_silence_limit(self):
        """Groundwork, deliberately inert for now -- and this pins why.

        A silence long enough to be worth shortening (3s) is far past the limit that CLOSES an
        utterance (1.5s), so it can only ever fall between two utterances, never inside one.
        The transform starts doing work when utterances are accumulated into longer blocks;
        until then it must leave everything exactly as it is, and this asserts that it does.
        """
        emitted = []
        sections = ((500, speech_frame), (4000, silent_frame), (500, speech_frame), (2000, silent_frame))
        feed(build_manager(emitted, verdicts=verdicts_for(*sections)), timeline(*sections))

        self.assertEqual(len(emitted), 2, "a 4s silence closes the utterance rather than sitting inside it")
        for message in emitted:
            self.assertLess(len(message["audio_data"]), TRIM_SILENCE_OVER_MS * SAMPLE_RATE // 1000 * 2)
