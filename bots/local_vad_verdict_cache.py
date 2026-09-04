"""Reusing an already-decided speech/silence verdict instead of asking the model again.

Every drain glues the still-open utterance's bytes onto the front of whatever's new and replays
the whole thing through the VAD -- meaning the SAME seconds get judged again on every drain an
utterance stays open. Measured on a realistic 30s sentence built up one second at a time: the
model runs 465 times for 30 seconds of audio, and the first second is re-judged roughly 30 times
by the time the sentence closes. That is wasted work, and worse, it tells a RECURRENT model it
just heard the same audio a second time, which is not the input it was designed for.

The verdict for a given frame, once decided, is a fixed fact -- nothing about it changes on a
later drain. So it only needs deciding once: this module hands back what was already decided,
and only asks live for whatever is genuinely new.
"""

import logging

logger = logging.getLogger(__name__)


class VerdictCache:
    """Hands back cached verdicts in order, then defers to a live call once exhausted."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts) if verdicts else []
        self._index = 0

    def next(self, live_check):
        """The next verdict: from cache if any remain, else from calling `live_check()`."""
        if self._index < len(self._verdicts):
            verdict = self._verdicts[self._index]
            self._index += 1
            return verdict
        return live_check()


def verdicts_for_buffered(verdicts, buffered_frame_count):
    """The trailing slice of `verdicts` that belongs to whatever is still buffered, unflushed.

    Mirrors the slicing the emit path already does for a flushed utterance: the base class drops
    leading silence before a buffer opens, so verdicts recorded before the currently-open
    utterance began are not part of it, and only the last `buffered_frame_count` are.
    """
    if buffered_frame_count <= 0:
        return []
    return verdicts[-buffered_frame_count:]


def safe_cached_verdicts(verdicts, audio, frame_bytes):
    """`verdicts`, or nothing at all if it could not possibly belong to `audio`.

    A cache longer than the audio's own frame count would bleed cached -- and therefore wrong --
    verdicts onto genuinely new audio. That can only happen from corrupted Redis data or a tail
    saved before this cache existed, and either way the safe answer is to fall back to deciding
    every frame live, exactly as before this existed.
    """
    if not verdicts:
        return []
    if len(verdicts) > len(audio) // frame_bytes:
        logger.warning(f"Discarding a verdict cache longer than its audio ({len(verdicts)} verdicts for {len(audio)} bytes); scoring live instead")
        return []
    return verdicts
