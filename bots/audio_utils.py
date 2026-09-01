"""Pure audio helpers, with no Django or codec dependencies.

Deliberately separate from ``bots/utils.py``: that module imports ``cv2``, ``pydub`` and
``bots.models``, and the audio input managers must stay importable without pulling in the ORM.
Everything here is a pure function over PCM bytes.
"""

import numpy as np

# 16-bit PCM is the only sample format this pipeline carries, so full scale is 2**15.
INT16_FULL_SCALE = 32768.0


def calculate_normalized_rms(audio_bytes):
    """Loudness of a PCM16 frame as 0.0-1.0, where 1.0 is full scale.

    Widening to float64 *before* squaring is load-bearing, not style: ``np.square()`` on an
    int16 array returns int16, which wraps for any ``|sample| > 181`` -- i.e. for essentially
    all real speech. The wrapped result is uncorrelated with loudness (it reports *smaller*
    values for *louder* audio) and goes negative often enough that ``np.sqrt`` returns ``nan``,
    which then reads as "not silent" wherever the result is compared against a threshold.

    A frame that cannot be interpreted as PCM16 at all is reported as silence, which is the
    safe direction: it can never open an utterance on malformed input.
    """
    if not audio_bytes:
        return 0.0

    try:
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
    except (ValueError, TypeError, BufferError):
        # Odd byte counts and non-buffer inputs land here.
        return 0.0

    if samples.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(samples))) / INT16_FULL_SCALE)
