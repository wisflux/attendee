"""Tests for the shared PCM loudness meter.

``test_full_scale_sine_reads_near_full_scale`` and
``test_louder_audio_always_reads_louder`` are the regression tests for the int16 overflow:
both fail against the previous implementation, which squared an int16 array in int16 and so
reported *smaller* values for *louder* audio.
"""

import numpy as np
from django.test import SimpleTestCase

from bots.audio_utils import INT16_FULL_SCALE, calculate_normalized_rms

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# A sine's RMS is its amplitude / sqrt(2), so a full-scale sine reads ~0.707.
FULL_SCALE_SINE_RMS = 1.0 / np.sqrt(2)
# Generous tolerance: these assertions are about orders of magnitude, not precision.
RELATIVE_TOLERANCE = 0.01


def sine_frame(dbfs, samples=FRAME_SAMPLES):
    """One frame of a 440 Hz sine at the requested level, as PCM16 bytes."""
    amplitude = (10 ** (dbfs / 20)) * (INT16_FULL_SCALE - 1)
    t = np.arange(samples) / SAMPLE_RATE
    wave = amplitude * np.sin(2 * np.pi * 440 * t)
    return wave.astype(np.int16).tobytes()


class CalculateNormalizedRmsTests(SimpleTestCase):
    def test_digital_silence_reads_zero(self):
        self.assertEqual(calculate_normalized_rms(b"\x00\x00" * FRAME_SAMPLES), 0.0)

    def test_empty_buffer_reads_zero(self):
        # The previous implementation returned nan here, which compares False against every
        # threshold and so read as "not silent".
        self.assertEqual(calculate_normalized_rms(b""), 0.0)

    def test_none_reads_zero(self):
        self.assertEqual(calculate_normalized_rms(None), 0.0)

    def test_odd_byte_count_reads_zero(self):
        # Not a whole number of 16-bit samples; must not raise.
        self.assertEqual(calculate_normalized_rms(b"\x00\x00\x01"), 0.0)

    def test_full_scale_sine_reads_near_full_scale(self):
        """REGRESSION: the int16 overflow reported ~0.0008 for this frame."""
        rms = calculate_normalized_rms(sine_frame(0))
        self.assertGreater(rms, 0.5)
        self.assertAlmostEqual(rms, FULL_SCALE_SINE_RMS, delta=RELATIVE_TOLERANCE)

    def test_normal_speech_level_is_not_silent(self):
        """REGRESSION: -20 dBFS is ordinary talking level and was classified as silence."""
        rms = calculate_normalized_rms(sine_frame(-20))
        self.assertGreater(rms, 0.01)

    def test_quiet_speech_level_matches_the_analytic_value(self):
        rms = calculate_normalized_rms(sine_frame(-40))
        expected = (10 ** (-40 / 20)) * FULL_SCALE_SINE_RMS
        self.assertAlmostEqual(rms, expected, delta=expected * 0.05)

    def test_louder_audio_always_reads_louder(self):
        """REGRESSION: the overflow inverted this ordering."""
        levels = [-60, -50, -40, -30, -20, -10, 0]
        readings = [calculate_normalized_rms(sine_frame(dbfs)) for dbfs in levels]
        self.assertEqual(readings, sorted(readings))

    def test_every_sample_above_the_old_overflow_point_is_measured_correctly(self):
        """182 is where squaring in int16 began to wrap; 181 was the last safe value."""
        for amplitude in (181, 182, 3000, 20000):
            frame = np.full(FRAME_SAMPLES, amplitude, dtype=np.int16).tobytes()
            self.assertAlmostEqual(
                calculate_normalized_rms(frame),
                amplitude / INT16_FULL_SCALE,
                delta=RELATIVE_TOLERANCE,
                msg=f"amplitude {amplitude} misread",
            )
