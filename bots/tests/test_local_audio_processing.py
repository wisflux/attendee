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


class TrailingSilenceTrimTests(SimpleTestCase):
    """An utterance always ends with the full silence limit, because that is the flush
    condition. Measured at ~30% of a typical clip and up to 80% of a short one -- and dead air
    is what a transcription model fills with invented text."""

    def emit(self, params, audio, voiced_end_bytes):
        manager, emitted = build(params)
        manager._voiced_ms = 1000.0
        manager._voiced_end_bytes = voiced_end_bytes
        manager.save_audio_chunk_callback({"audio_data": audio, "timestamp_ms": 0})
        return emitted[0]["audio_data"] if emitted else None

    def test_trailing_silence_is_cut_back_to_the_keep_margin(self):
        speech_bytes = len(silence(500))
        audio = silence(500) + silence(2000)
        kept = self.emit(LocalVadParams(trailing_keep_ms=200), audio, speech_bytes)
        self.assertEqual(duration_ms(kept, SAMPLE_RATE), 700)

    def test_a_zero_margin_cuts_to_the_last_voiced_sample(self):
        speech_bytes = len(silence(500))
        kept = self.emit(LocalVadParams(trailing_keep_ms=0), silence(500) + silence(2000), speech_bytes)
        self.assertEqual(duration_ms(kept, SAMPLE_RATE), 500)

    def test_silence_inside_the_utterance_is_preserved(self):
        """Internal pauses carry timing the transcript depends on."""
        audio = silence(400) + silence(600) + silence(400) + silence(2000)
        voiced_end = len(silence(1400))  # speech ended after the internal pause
        kept = self.emit(LocalVadParams(trailing_keep_ms=200), audio, voiced_end)
        self.assertEqual(duration_ms(kept, SAMPLE_RATE), 1600)

    def test_an_utterance_with_no_voiced_audio_is_left_alone(self):
        audio = silence(800)
        self.assertEqual(self.emit(LocalVadParams(min_speech_ms=0), audio, 0), audio)

    def test_a_margin_longer_than_the_tail_is_a_no_op(self):
        audio = silence(500) + silence(100)
        kept = self.emit(LocalVadParams(trailing_keep_ms=5000), audio, len(silence(500)))
        self.assertEqual(kept, audio)

    def test_the_result_is_always_a_whole_number_of_samples(self):
        """An odd byte count is not decodable PCM16 and the upload endpoint rejects it."""
        for keep_ms in (0, 1, 7, 33, 200):
            with self.subTest(keep_ms=keep_ms):
                kept = self.emit(LocalVadParams(trailing_keep_ms=keep_ms), silence(500) + silence(900), len(silence(500)) - 1)
                self.assertEqual(len(kept) % BYTES_PER_SAMPLE, 0)

    def test_trimming_never_lengthens_the_audio(self):
        audio = silence(300) + silence(1500)
        kept = self.emit(LocalVadParams(trailing_keep_ms=200), audio, len(silence(300)))
        self.assertLessEqual(len(kept), len(audio))

    def test_the_boundary_resets_between_utterances(self):
        """A stale offset from the previous utterance would truncate the next one wrongly."""
        manager, emitted = build(LocalVadParams(trailing_keep_ms=0))
        manager._voiced_ms, manager._voiced_end_bytes = 1000.0, len(silence(100))
        manager.save_audio_chunk_callback({"audio_data": silence(900), "timestamp_ms": 0})
        manager._voiced_ms = 1000.0  # second utterance, no voiced boundary recorded
        manager.save_audio_chunk_callback({"audio_data": silence(900), "timestamp_ms": 1})
        self.assertEqual(duration_ms(emitted[0]["audio_data"], SAMPLE_RATE), 100)
        self.assertEqual(duration_ms(emitted[1]["audio_data"], SAMPLE_RATE), 900, "the second utterance reused a stale boundary")
