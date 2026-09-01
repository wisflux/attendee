"""Tests for the local session's audio manager.

These build the manager directly rather than through ``build_manager``, which needs a
Recording and a Participant. The behaviour under test is segmentation, which touches neither.
"""

from datetime import datetime, timedelta

import numpy as np
from django.test import SimpleTestCase

from bots.local_audio_processing import (
    BYTES_PER_SAMPLE,
    LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS,
    VAD_FRAME_MS,
    LocalAudioInputManager,
    duration_ms,
    feed,
    flush_remaining,
)
from bots.local_vad_params import LocalVadParams

SAMPLE_RATE = 16000
EPOCH = datetime(2026, 1, 1)


def build(params=None, **overrides):
    """A manager plus the list its utterances land in."""
    emitted = []
    settings = {
        "params": params or LocalVadParams(),
        "save_audio_chunk_callback": emitted.append,
        "get_participant_callback": lambda speaker_id: {"participant_uuid": "u", "participant_full_name": "You"},
        "sample_rate": SAMPLE_RATE,
        "utterance_size_limit": LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE,
        "silence_duration_limit": (params or LocalVadParams()).min_silence_seconds,
        "should_print_diagnostic_info": False,
    }
    settings.update(overrides)
    return LocalAudioInputManager(**settings), emitted


def silence(ms):
    return b"\x00\x00" * (SAMPLE_RATE * ms // 1000)


def frames_of(audio):
    """Split PCM into the 10 ms frames the pipeline feeds."""
    step = SAMPLE_RATE * VAD_FRAME_MS // 1000 * BYTES_PER_SAMPLE
    return [audio[i : i + step] for i in range(0, len(audio), step)]


class ParameterWiringTests(SimpleTestCase):
    def test_the_silence_limit_comes_from_the_parameters(self):
        manager, _ = build(LocalVadParams(min_silence_ms=2000))
        self.assertEqual(manager.SILENCE_DURATION_LIMIT, 2.0)

    def test_the_detector_is_built_with_the_configured_threshold(self):
        params = LocalVadParams(threshold=0.7, hysteresis_offset=0.15)
        manager, _ = build(params)
        manager.vad.is_speech("s", silence(VAD_FRAME_MS), SAMPLE_RATE)
        stream = manager.vad._streams["s"]
        self.assertEqual(stream._threshold, 0.7)
        self.assertAlmostEqual(stream._exit_threshold, 0.55)

    def test_the_bot_path_detector_keeps_its_own_defaults(self):
        """Local tuning must never change how a meeting bot is segmented."""
        from bots.bot_controller.silero_vad import _HYSTERESIS_OFFSET, _SPEECH_THRESHOLD

        self.assertEqual(_SPEECH_THRESHOLD, 0.5)
        self.assertEqual(_HYSTERESIS_OFFSET, 0.0, "hysteresis must be off by default or bot decisions change")


class MinimumSpeechTests(SimpleTestCase):
    def test_zero_keeps_every_utterance(self):
        manager, emitted = build(LocalVadParams(min_speech_ms=0))
        manager._voiced_ms = 30.0
        manager.save_audio_chunk_callback({"audio_data": b"\x00\x00", "timestamp_ms": 0})
        self.assertEqual(len(emitted), 1)

    def test_an_utterance_below_the_minimum_is_dropped(self):
        manager, emitted = build(LocalVadParams(min_speech_ms=300))
        manager._voiced_ms = 120.0
        manager.save_audio_chunk_callback({"audio_data": b"\x00\x00", "timestamp_ms": 0})
        self.assertEqual(emitted, [])

    def test_an_utterance_at_the_minimum_is_kept(self):
        manager, emitted = build(LocalVadParams(min_speech_ms=300))
        manager._voiced_ms = 300.0
        manager.save_audio_chunk_callback({"audio_data": b"\x00\x00", "timestamp_ms": 0})
        self.assertEqual(len(emitted), 1)

    def test_the_tally_resets_between_utterances(self):
        """Otherwise voiced audio from an earlier utterance would qualify a later short one."""
        manager, emitted = build(LocalVadParams(min_speech_ms=300))
        manager._voiced_ms = 500.0
        manager.save_audio_chunk_callback({"audio_data": b"\x00\x00", "timestamp_ms": 0})
        manager._voiced_ms += 50.0
        manager.save_audio_chunk_callback({"audio_data": b"\x00\x00", "timestamp_ms": 1})
        self.assertEqual(len(emitted), 1)

    def test_voiced_audio_is_tallied_from_the_silence_decision(self):
        manager, _ = build(LocalVadParams())
        frame = silence(VAD_FRAME_MS)
        manager.silence_detected("s", frame)
        self.assertEqual(manager._voiced_ms, 0.0, "digital silence must not count as voiced")


class SilenceSegmentationTests(SimpleTestCase):
    def test_pure_silence_produces_no_utterance(self):
        manager, emitted = build(LocalVadParams(min_silence_ms=1000))
        feed(manager, "mic", silence(4000), EPOCH, 0, SAMPLE_RATE)
        self.assertEqual(emitted, [])

    def test_flush_remaining_is_a_no_op_when_nothing_is_buffered(self):
        manager, emitted = build()
        flush_remaining(manager, "mic", EPOCH, 5000)
        self.assertEqual(emitted, [])

    def test_flush_remaining_trips_the_flush_at_any_configured_limit(self):
        """The probe is placed relative to the manager's limit, not a hard-coded one."""
        for min_silence_ms in (1000, 2000, 5000):
            with self.subTest(min_silence_ms=min_silence_ms):
                manager, emitted = build(LocalVadParams(min_silence_ms=min_silence_ms))
                manager.utterances["mic"] = bytearray(silence(500))
                manager.first_nonsilent_audio_time["mic"] = EPOCH
                manager.last_nonsilent_audio_time["mic"] = EPOCH
                flush_remaining(manager, "mic", EPOCH, 500)
                self.assertEqual(len(emitted), 1, "the final utterance was not emitted")


class FeedTests(SimpleTestCase):
    def test_only_whole_frames_are_consumed(self):
        manager, _ = build()
        frame_bytes = SAMPLE_RATE * VAD_FRAME_MS // 1000 * BYTES_PER_SAMPLE
        audio = silence(100) + b"\x00" * 7  # a ragged tail
        consumed = feed(manager, "mic", audio, EPOCH, 0, SAMPLE_RATE)
        self.assertEqual(consumed % frame_bytes, 0)
        self.assertLessEqual(consumed, len(audio))

    def test_frame_timestamps_follow_the_session_timeline_not_the_wall_clock(self):
        manager, emitted = build(LocalVadParams(min_silence_ms=1000))
        seen = []
        manager.process_chunk = lambda source, at, chunk: seen.append(at)
        feed(manager, "mic", silence(50), EPOCH, 1000, SAMPLE_RATE)
        self.assertEqual(seen[0], EPOCH + timedelta(milliseconds=1000))
        self.assertEqual(seen[1], EPOCH + timedelta(milliseconds=1010))


class DurationTests(SimpleTestCase):
    def test_duration_of_a_known_buffer(self):
        self.assertEqual(duration_ms(silence(250), SAMPLE_RATE), 250)

    def test_duration_of_an_empty_buffer(self):
        self.assertEqual(duration_ms(b"", SAMPLE_RATE), 0)

    def test_frames_helper_matches_the_feed_step(self):
        self.assertEqual(len(frames_of(silence(100))), 10)
        self.assertEqual(len(np.frombuffer(silence(10), dtype=np.int16)), SAMPLE_RATE * VAD_FRAME_MS // 1000)
