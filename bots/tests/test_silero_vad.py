"""Tests for the streaming Silero VAD wrapper.

The accumulator tests are the point of this file. Silero accepts exactly 512 samples at
16 kHz and raises on anything else, while the pipeline feeds 10 ms frames (160 samples at
16 kHz). An implementation without an accumulator therefore never invokes the model at all --
it silently falls back to whatever its error path does -- and no aggregate metric reveals it.
That is a real failure mode: it shipped in another fork of this codebase.
"""

from unittest.mock import patch

import numpy as np
from django.test import SimpleTestCase

from bots.bot_controller import silero_vad
from bots.bot_controller.silero_vad import (
    _CONTEXT_SAMPLES,
    _WINDOW_SAMPLES,
    SileroModelUnavailable,
    SileroVoiceActivityDetector,
    _SpeakerStream,
)

FRAME_MS = 10
SPEECH_DBFS = -20
MODEL_INPUT_WIDTH = _WINDOW_SAMPLES + _CONTEXT_SAMPLES


def frame_bytes(sample_rate, dbfs=SPEECH_DBFS, ms=FRAME_MS, voiced=True):
    """One PCM16 frame: a two-harmonic tone if voiced, otherwise digital silence."""
    n = sample_rate * ms // 1000
    if not voiced:
        return b"\x00\x00" * n
    amplitude = (10 ** (dbfs / 20)) * 32767
    t = np.arange(n) / sample_rate
    wave = amplitude * (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)) / 1.5
    return wave.astype(np.int16).tobytes()


class AccumulatorTests(SimpleTestCase):
    def test_ten_millisecond_frames_reach_the_model(self):
        """REGRESSION: without an accumulator the model is never invoked at all."""
        for sample_rate, frames_per_window in ((16000, 3.2), (32000, 3.2), (48000, 3.2)):
            with self.subTest(sample_rate=sample_rate):
                calls = []
                real_session = silero_vad._get_session()

                def spy(_outputs, feeds, _calls=calls, _real=real_session):
                    _calls.append(feeds["input"].shape)
                    return _real.run(None, feeds)

                stream = _SpeakerStream(sample_rate)
                with patch.object(silero_vad, "_get_session") as get_session:
                    get_session.return_value.run.side_effect = spy
                    for _ in range(10):
                        stream.is_speech(frame_bytes(sample_rate))

                self.assertGreater(len(calls), 0, "the model was never invoked")
                expected = int(10 / frames_per_window)
                self.assertAlmostEqual(len(calls), expected, delta=1)

    def test_the_model_is_only_ever_given_one_window_width(self):
        """Silero raises on any other width, so a wrong size must be impossible by construction."""
        calls = []
        real_session = silero_vad._get_session()

        def spy(_outputs, feeds, _calls=calls, _real=real_session):
            _calls.append(feeds["input"].shape)
            return _real.run(None, feeds)

        stream = _SpeakerStream(16000)
        with patch.object(silero_vad, "_get_session") as get_session:
            get_session.return_value.run.side_effect = spy
            # Deliberately ragged frame lengths, including a partial frame.
            for ms in (10, 10, 7, 10, 23, 10, 1, 10, 10):
                stream.is_speech(frame_bytes(16000, ms=ms))

        self.assertGreater(len(calls), 0)
        for shape in calls:
            self.assertEqual(shape, (1, MODEL_INPUT_WIDTH))

    def test_a_partial_frame_holds_the_previous_decision(self):
        stream = _SpeakerStream(16000)
        for _ in range(4):
            stream.is_speech(frame_bytes(16000))
        settled = stream.is_speech(frame_bytes(16000))
        self.assertIs(stream.is_speech(b"\x00\x00"), settled)


class DiscriminationTests(SimpleTestCase):
    """What synthetic audio can and cannot prove about Silero.

    It can prove rejection: silence and tones must not be classified as speech, and that is
    the property this VAD exists for. It CANNOT prove recall -- Silero correctly rejects
    synthetic tones as non-speech (unlike webrtcvad, which is energy-based and accepts them),
    so a "speech is speech" assertion built from a sine sum would only pass vacuously, via the
    fail-open decision returned before the first 512-sample window completes. Recall against
    real speech belongs in the offline comparison harness, which has real audio fixtures.
    """

    def test_digital_silence_is_not_speech(self):
        vad = SileroVoiceActivityDetector()
        decisions = [vad.is_speech("s", frame_bytes(16000, voiced=False), 16000) for _ in range(20)]
        self.assertFalse(any(decisions[4:]), "silence was classified as speech")

    def test_a_pure_tone_is_not_speech(self):
        """webrtcvad called a 1 kHz tone 100% speech above roughly -22 dBFS; Silero must not."""
        vad = SileroVoiceActivityDetector()
        n = 16000 * FRAME_MS // 1000
        amplitude = (10 ** (SPEECH_DBFS / 20)) * 32767
        decisions = []
        for i in range(20):
            t = (np.arange(n) + i * n) / 16000
            tone = (amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.int16).tobytes()
            decisions.append(vad.is_speech("s", tone, 16000))
        self.assertFalse(any(decisions[4:]), "a 1 kHz tone was classified as speech")


class StateIsolationTests(SimpleTestCase):
    def test_two_speakers_do_not_share_state(self):
        """Interleaving two speakers must give each the same answers it gets alone."""
        alone = SileroVoiceActivityDetector()
        solo = [alone.is_speech("a", frame_bytes(16000), 16000) for _ in range(20)]

        shared = SileroVoiceActivityDetector()
        interleaved = []
        for _ in range(20):
            interleaved.append(shared.is_speech("a", frame_bytes(16000), 16000))
            shared.is_speech("b", frame_bytes(16000, voiced=False), 16000)

        self.assertEqual(solo, interleaved)

    def test_reset_returns_a_speaker_to_a_fresh_stream(self):
        vad = SileroVoiceActivityDetector()
        fresh = [vad.is_speech("a", frame_bytes(16000), 16000) for _ in range(8)]
        vad.reset("a")
        after_reset = [vad.is_speech("a", frame_bytes(16000), 16000) for _ in range(8)]
        self.assertEqual(fresh, after_reset)

    def test_reset_of_an_unknown_speaker_is_harmless(self):
        SileroVoiceActivityDetector().reset("never-seen")


class DeterminismTests(SimpleTestCase):
    def test_the_same_audio_gives_the_same_answers(self):
        """Celery retries re-run the VAD over the same audio; it must not drift."""
        frames = [frame_bytes(16000) for _ in range(15)]
        first = [SileroVoiceActivityDetector().is_speech("a", f, 16000) for f in frames]
        second = [SileroVoiceActivityDetector().is_speech("a", f, 16000) for f in frames]
        self.assertEqual(first, second)


class UnsupportedInputTests(SimpleTestCase):
    def test_a_rate_that_cannot_be_decimated_raises(self):
        """EC2/EC3: swallowing this would return 'speech' forever and disable the VAD."""
        for sample_rate in (8000, 44100, 22050):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    _SpeakerStream(sample_rate)

    def test_an_unsupported_rate_is_not_swallowed_by_the_fail_open_path(self):
        with self.assertRaises(ValueError):
            SileroVoiceActivityDetector().is_speech("a", frame_bytes(16000), 44100)

    def test_supported_rates_are_accepted(self):
        for sample_rate in (16000, 32000, 48000):
            with self.subTest(sample_rate=sample_rate):
                _SpeakerStream(sample_rate)

    def test_an_unusable_model_raises_rather_than_failing_open(self):
        """EC5: a detector that answers 'speech' to everything is worse than a crash."""
        vad = SileroVoiceActivityDetector()
        with patch.object(silero_vad, "_get_session", side_effect=SileroModelUnavailable("boom")):
            with self.assertRaises(SileroModelUnavailable):
                vad.is_speech("a", frame_bytes(16000), 16000)

    def test_a_transient_per_chunk_error_fails_open(self):
        vad = SileroVoiceActivityDetector()
        vad.is_speech("a", frame_bytes(16000), 16000)
        with patch.object(silero_vad, "_get_session", side_effect=RuntimeError("one bad chunk")):
            self.assertTrue(vad.is_speech("a", frame_bytes(16000), 16000))


class InitialDecisionTests(SimpleTestCase):
    """A fresh stream must not claim speech before it has evaluated a window.

    The stream is reset at every utterance flush, so an optimistic initial decision re-arms on
    every non-speech stretch and emits a short clip each time -- the offline harness measured
    three spurious utterances from a single steady tone.
    """

    def test_a_fresh_stream_reports_silence_before_its_first_window(self):
        stream = _SpeakerStream(16000)
        self.assertFalse(stream.is_speech(b""), "claimed speech with no audio at all")

    def test_a_tone_produces_no_speech_even_across_resets(self):
        vad = SileroVoiceActivityDetector()
        n = 16000 * FRAME_MS // 1000
        amplitude = (10 ** (SPEECH_DBFS / 20)) * 32767
        decisions = []
        for i in range(60):
            if i % 20 == 0:
                vad.reset("s")  # what the manager does when an utterance flushes
            t = (np.arange(n) + i * n) / 16000
            tone = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes()
            decisions.append(vad.is_speech("s", tone, 16000))
        self.assertFalse(any(decisions), "a reset re-armed an optimistic guess on non-speech")
