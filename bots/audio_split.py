"""Finding a place to cut a long utterance that is not in the middle of a word.

The size cap has to fire somewhere -- a speaker who never pauses would otherwise buffer
without bound. Firing it at an exact byte offset cuts mid-syllable, and the transcriber then
invents an ending for one half and a beginning for the other, which is what turns a long
answer into two half-sentences that do not join up.

Cutting at the quietest moment nearby costs a fraction of a second of boundary accuracy and
keeps the words on either side intact. When there is no quiet moment to find -- somebody
genuinely talking straight through -- the end is returned and the caller cuts at the cap
exactly as it did before.
"""

import numpy as np

BYTES_PER_SAMPLE = 2
# Above this RMS a window is the softest part of continuous speech rather than a real gap.
# Matches the audio manager's own silence gate, so both agree on what counts as quiet.
SILENCE_RMS_THRESHOLD = 0.01
INT16_FULL_SCALE = 32768.0
# How far back from the end to look, and how much audio each candidate window covers.
DEFAULT_SEARCH_MS = 3000
DEFAULT_WINDOW_MS = 100
# Windows overlap: stepping a tenth of a window keeps a short gap from falling between two
# candidates without measuring every sample.
STEPS_PER_WINDOW = 10


def _rms(samples):
    return float(np.sqrt(np.mean(np.square(samples))) / INT16_FULL_SCALE)


def _quietest_window(samples, window, search_from):
    """Centre sample of the lowest-RMS window at or after `search_from`, and its RMS."""
    best_rms, best_centre = None, None
    step = max(1, window // STEPS_PER_WINDOW)
    for start in range(search_from, samples.size - window + 1, step):
        rms = _rms(samples[start : start + window])
        if best_rms is None or rms < best_rms:
            best_rms, best_centre = rms, start + window // 2
    return best_rms, best_centre


def quietest_split_point(audio, sample_rate, search_ms=DEFAULT_SEARCH_MS, window_ms=DEFAULT_WINDOW_MS):
    """Byte offset of the quietest moment near the end of `audio`.

    Only the last `search_ms` is considered: a quiet stretch earlier in the utterance is not a
    candidate, because cutting there would throw away speech already buffered behind it.
    """
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64)
    window = int(sample_rate * window_ms / 1000)
    search = int(sample_rate * search_ms / 1000)
    if window <= 0 or samples.size < window * 2 or search < window:
        return len(audio)

    best_rms, best_centre = _quietest_window(samples, window, max(0, samples.size - search))
    if best_rms is None or best_rms > SILENCE_RMS_THRESHOLD:
        return len(audio)
    return best_centre * BYTES_PER_SAMPLE


def contains_speech(audio):
    """Whether a buffer holds anything above the silence threshold.

    Used to decide if a tail is worth carrying into the next utterance. RMS over the whole
    buffer is enough: a tail is at most a few seconds, so even a single word in it lifts the
    measurement well clear of the threshold.
    """
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return False
    return _rms(samples) > SILENCE_RMS_THRESHOLD
