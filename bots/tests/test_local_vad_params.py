"""Tests for local VAD parameter parsing.

The point of the validation is that a mistyped override must not silently fall back to the
default: a sweep run with VAD_THRESHOLD=0.g that quietly used 0.7 would produce results
attributed to the wrong setting.
"""

from unittest.mock import patch

from django.test import SimpleTestCase

from bots.local_vad_params import (
    DEFAULT_HYSTERESIS_OFFSET,
    DEFAULT_MIN_SILENCE_MS,
    DEFAULT_MIN_SPEECH_MS,
    DEFAULT_THRESHOLD,
    InvalidVadParameter,
    LocalVadParams,
)


def with_env(**overrides):
    return patch.dict("os.environ", overrides, clear=False)


class DefaultsTests(SimpleTestCase):
    def test_an_unset_environment_gives_the_documented_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            params = LocalVadParams.from_env()
        self.assertEqual(params.threshold, DEFAULT_THRESHOLD)
        self.assertEqual(params.hysteresis_offset, DEFAULT_HYSTERESIS_OFFSET)
        self.assertEqual(params.min_speech_ms, DEFAULT_MIN_SPEECH_MS)
        self.assertEqual(params.min_silence_ms, DEFAULT_MIN_SILENCE_MS)

    def test_min_speech_defaults_to_zero_so_short_replies_survive(self):
        """'yeah' and 'mhm' are real turns; any positive default would delete them."""
        self.assertEqual(DEFAULT_MIN_SPEECH_MS, 0)

    def test_the_exit_threshold_stays_clear_of_the_measured_music_peak(self):
        """Music with attack transients measured 0.452 on this model."""
        self.assertGreater(DEFAULT_THRESHOLD - DEFAULT_HYSTERESIS_OFFSET, 0.452)

    def test_an_empty_string_is_treated_as_unset(self):
        with with_env(VAD_THRESHOLD=""):
            self.assertEqual(LocalVadParams.from_env().threshold, DEFAULT_THRESHOLD)


class OverrideTests(SimpleTestCase):
    def test_the_requested_first_run_values_are_accepted(self):
        with with_env(VAD_THRESHOLD="0.7", VAD_MIN_SPEECH_MS="0", VAD_MIN_SILENCE_MS="2000", VAD_HYSTERESIS_OFFSET="0.15"):
            params = LocalVadParams.from_env()
        self.assertEqual(params.threshold, 0.7)
        self.assertEqual(params.min_speech_ms, 0)
        self.assertEqual(params.min_silence_ms, 2000)
        self.assertEqual(params.hysteresis_offset, 0.15)

    def test_min_silence_is_exposed_in_the_units_the_manager_wants(self):
        with with_env(VAD_MIN_SILENCE_MS="2500"):
            self.assertEqual(LocalVadParams.from_env().min_silence_seconds, 2.5)

    def test_params_are_immutable(self):
        with self.assertRaises(Exception):
            LocalVadParams().threshold = 0.9


class ValidationTests(SimpleTestCase):
    def test_an_unparseable_value_is_rejected_rather_than_defaulted(self):
        for name, value in (("VAD_THRESHOLD", "0.g"), ("VAD_MIN_SPEECH_MS", "lots"), ("VAD_MIN_SILENCE_MS", "2.5")):
            with self.subTest(name=name):
                with with_env(**{name: value}):
                    with self.assertRaises(InvalidVadParameter):
                        LocalVadParams.from_env()

    def test_out_of_range_values_are_rejected(self):
        cases = (
            ("VAD_THRESHOLD", "0.0"),
            ("VAD_THRESHOLD", "1.0"),
            ("VAD_THRESHOLD", "-0.5"),
            ("VAD_MIN_SPEECH_MS", "-1"),
            ("VAD_MIN_SPEECH_MS", "999999"),
            ("VAD_MIN_SILENCE_MS", "50"),
            ("VAD_MIN_SILENCE_MS", "600000"),
            ("VAD_HYSTERESIS_OFFSET", "0.9"),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with with_env(**{name: value}):
                    with self.assertRaises(InvalidVadParameter):
                        LocalVadParams.from_env()

    def test_hysteresis_at_or_above_the_threshold_is_rejected(self):
        """The exit threshold would be <= 0, so speech could never end."""
        with with_env(VAD_THRESHOLD="0.5", VAD_HYSTERESIS_OFFSET="0.5"):
            with self.assertRaises(InvalidVadParameter):
                LocalVadParams.from_env()

    def test_the_error_names_the_variable_and_the_value(self):
        with with_env(VAD_THRESHOLD="2.0"):
            with self.assertRaises(InvalidVadParameter) as caught:
                LocalVadParams.from_env()
        self.assertIn("VAD_THRESHOLD", str(caught.exception))
        self.assertIn("2.0", str(caught.exception))
