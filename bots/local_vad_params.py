"""Tunable VAD parameters for local recording sessions.

Local-only on purpose: the meeting-bot path keeps the detector's own defaults, so tuning a
local session can never change how a bot recording is segmented.

Every value is a named default that an environment variable may override, so the parameters
can be swept during the Silero/webrtcvad comparison without a rebuild. Once the values are
settled the defaults here become the answer and the variables stay as an escape hatch.

An override that cannot be parsed, or that falls outside the documented range, is rejected
loudly rather than silently falling back -- a mistyped threshold that quietly reverts to the
default is the kind of thing that invalidates a whole comparison run.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Silero reports a speech probability per 32 ms window. Measured on this model: real speech
# averages ~0.90, while music with attack transients peaks at ~0.45 -- so 0.7 clears both with
# room either side.
DEFAULT_THRESHOLD = 0.7

# Speech ends at threshold - offset, so a probability wobbling around the threshold cannot chop
# an utterance into fragments. The exit (0.55 by default) must still clear that music peak.
DEFAULT_HYSTERESIS_OFFSET = 0.15

# Total voiced audio an utterance must contain to be worth transcribing. Zero keeps everything,
# which is deliberate: "yeah" and "mhm" are real turns, and Silero rejects non-speech well
# enough that this filter is not needed to suppress noise.
DEFAULT_MIN_SPEECH_MS = 0

# How long silence must persist before an utterance is closed. Drives the manager's own limit;
# there is deliberately no second timer inside the detector.
DEFAULT_MIN_SILENCE_MS = 1000

# How much trailing silence to keep on a finished utterance. Every clip ends with exactly
# min_silence_ms of silence, because that is the flush condition -- measured at roughly 30% of
# a typical clip, and up to 80% of a short one. That dead air is what a transcription model
# fills in when it invents text, so it is trimmed back to a natural-sounding tail. Zero would
# clip the final consonant.
DEFAULT_TRAILING_KEEP_MS = 200

# Loudness below which a frame is discarded WITHOUT asking the VAD. The bot path uses 0.01
# (-40 dBFS), which predates Silero and was doing the noise rejection webrtcvad could not:
# measured, it deletes 910ms of speech at -38 dBFS and everything at -44 dBFS. The desktop
# captures the microphone raw, with no automatic gain control, so ordinary speech frequently
# sits below that. Silero scores real speech at -40 dBFS at 0.905, so this gate is now only
# a "definitely nothing here" short-circuit and the speech decision belongs to the model.
DEFAULT_SILENCE_RMS = 0.0005  # -66 dBFS, below any real microphone noise floor

MIN_THRESHOLD, MAX_THRESHOLD = 0.05, 0.95
MAX_HYSTERESIS_OFFSET = 0.5
MAX_MIN_SPEECH_MS = 5_000
MAX_TRAILING_KEEP_MS = 5_000
MIN_SILENCE_RMS, MAX_SILENCE_RMS = 0.0, 0.05
MIN_MIN_SILENCE_MS, MAX_MIN_SILENCE_MS = 100, 30_000


class InvalidVadParameter(ValueError):
    """An override was unparseable or out of range."""


def _env_float(name, default, low, high):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise InvalidVadParameter(f"{name}={raw!r} is not a number") from error
    if not low <= value <= high:
        raise InvalidVadParameter(f"{name}={value} is outside [{low}, {high}]")
    return value


def _env_int(name, default, low, high):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise InvalidVadParameter(f"{name}={raw!r} is not a whole number") from error
    if not low <= value <= high:
        raise InvalidVadParameter(f"{name}={value} is outside [{low}, {high}]")
    return value


@dataclass(frozen=True)
class LocalVadParams:
    threshold: float = DEFAULT_THRESHOLD
    hysteresis_offset: float = DEFAULT_HYSTERESIS_OFFSET
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS
    trailing_keep_ms: int = DEFAULT_TRAILING_KEEP_MS
    silence_rms: float = DEFAULT_SILENCE_RMS

    @property
    def min_silence_seconds(self):
        """What the audio manager's flush rule wants; the same knob, in its units."""
        return self.min_silence_ms / 1000.0

    @classmethod
    def from_env(cls):
        params = cls(
            threshold=_env_float("VAD_THRESHOLD", DEFAULT_THRESHOLD, MIN_THRESHOLD, MAX_THRESHOLD),
            hysteresis_offset=_env_float("VAD_HYSTERESIS_OFFSET", DEFAULT_HYSTERESIS_OFFSET, 0.0, MAX_HYSTERESIS_OFFSET),
            min_speech_ms=_env_int("VAD_MIN_SPEECH_MS", DEFAULT_MIN_SPEECH_MS, 0, MAX_MIN_SPEECH_MS),
            min_silence_ms=_env_int("VAD_MIN_SILENCE_MS", DEFAULT_MIN_SILENCE_MS, MIN_MIN_SILENCE_MS, MAX_MIN_SILENCE_MS),
            trailing_keep_ms=_env_int("VAD_TRAILING_KEEP_MS", DEFAULT_TRAILING_KEEP_MS, 0, MAX_TRAILING_KEEP_MS),
            silence_rms=_env_float("VAD_SILENCE_RMS", DEFAULT_SILENCE_RMS, MIN_SILENCE_RMS, MAX_SILENCE_RMS),
        )
        if params.hysteresis_offset >= params.threshold:
            raise InvalidVadParameter(f"VAD_HYSTERESIS_OFFSET ({params.hysteresis_offset}) must be below VAD_THRESHOLD ({params.threshold}); otherwise speech could never end")
        return params
