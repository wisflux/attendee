"""Deciding when a run of local utterances is worth sending as one transcription request.

The live pipeline sends every utterance on its own, so the model receives 1-3 seconds at a time
and re-identifies the language on each one -- which is where wrong-language lines come from, and
why a sentence spoken across a breath arrives as two halves that do not join up. Measured on a
93.8s recording, this path made 6 requests where the offline reference made 2.

WHAT GROUPING DOES NOT CHANGE. Utterances are still cut exactly as they are today, and each
still gets its own row, its own timestamp and its own speaker. Only the transcription REQUEST is
shared: the group's audio is concatenated, sent once, and the returned words are put back on the
rows they came from. Ordering and the You/Others split therefore carry no new risk.

A GROUP IS ALWAYS ONE SOURCE. The caller keys groups by source (mic / system), so a group can
never mix the two speakers into a single request -- concatenating them would blend two voices
and there would be no honest way to attribute the words afterwards.

Timing is measured on the session's own timeline (``timestamp_ms``), never a wall clock, so a
slow drain or a retried task cannot change where a boundary falls.
"""

import logging

logger = logging.getLogger(__name__)

# Enough speech that the model can identify the language once and hold it for the whole request.
TARGET_VOICE_MS = 30000
# Once the target is met, wait for a real boundary. Closing at the first micro-gap was measured
# splitting a sentence between "the best" and "person in the world" across a 50ms pause.
MIN_BOUNDARY_PAUSE_MS = 700
# A speaker who has stopped should not wait for the target -- otherwise a short answer is held
# until the session ends.
STOPPED_TALKING_MS = 5000
# Somebody reading aloud never offers a boundary pause, so a group needs a hard ceiling. It also
# bounds the audio held for one request, which is what keeps a long meeting's memory flat.
MAX_BLOCK_MS = 90000
# Below this a request is mostly silence, and a model handed mostly-silence invents a line.
# Such a group keeps gathering instead; the next real speech carries it.
MIN_BLOCK_VOICE_MS = 300

CLOSE_ENOUGH_CONTEXT = "enough context"
CLOSE_STOPPED_TALKING = "stopped talking"
CLOSE_CEILING = "ceiling"
CLOSE_SESSION_ENDED = "session ended"

# An utterance cut by the size cap ends mid-syllable, so a request must not end there.
SIZE_CAP_FLUSH_REASON = "buffer_full"

# A pause is sentence structure, not padding. The offline blocks that produced the best measured
# transcripts were 16% and 42% silence, and deleting silence was measured losing a speaker's
# surname, losing a company name and dropping an entire Hindi section. So inside a request the
# REAL pause is reproduced -- up to the point where a silence stops reading as a breath and starts
# reading as a full stop, which undoes the context the group exists to buy.
TRIM_SILENCE_OVER_MS = 3000
# Past that limit it is shortened to this share of itself, never removed.
SILENCE_KEEP_FRACTION = 0.30
# A safety bound, not a second rhythm rule. A group holding less than MIN_BLOCK_VOICE_MS never
# closes -- the voice floor blocks it, and the ceiling cannot fire because the span only grows when
# a member is added -- so a cough can meet its next member minutes later. A fraction of a pause
# that long is still a request that is mostly silence, which is the input that makes the model
# invent a line. Set above every pause the measured reference contained: its best block ran 44.35s
# carrying 37.1s of voice, so no single silence inside it exceeded 7.25s.
MAX_GAP_MS = 10000


def shortened_silence_ms(silence_ms):
    """A pause kept whole below the limit, shortened above it, never deleted."""
    if silence_ms <= TRIM_SILENCE_OVER_MS:
        return silence_ms
    return min(MAX_GAP_MS, int(silence_ms * SILENCE_KEEP_FRACTION))


def gaps_ms(members):
    """Silence to write between each consecutive pair, in timeline order.

    One entry shorter than `members`. This same list builds the audio AND locates the words that
    come back, so the two cannot drift apart -- recomputing it separately in the splitter is what
    would let a few milliseconds per utterance accumulate and land words on the wrong row.

    Each utterance has already had the silence that closed it trimmed back to a short pad, so the
    gap measured here plus that pad reconstructs the pause as it was actually spoken.

    Never negative: trimming and rounding can leave one row ending after the next begins.
    """
    gaps = []
    for previous, current in zip(members, members[1:]):
        elapsed = current["timestamp_ms"] - (previous["timestamp_ms"] + previous["duration_ms"])
        gaps.append(shortened_silence_ms(max(0, elapsed)))
    return gaps


def span_ms(members):
    """How much of the session's timeline the group covers, gaps included."""
    if not members:
        return 0
    last = members[-1]
    return last["timestamp_ms"] + last["duration_ms"] - members[0]["timestamp_ms"]


def close_reason(members, silence_after_ms, session_ended=False):
    """Why this group should be sent now, or None to keep gathering.

    `members` are one source's utterances in timeline order, each carrying `voice_ms`,
    `duration_ms`, `timestamp_ms` and the `flush_reason` that ended it. `silence_after_ms` is how
    long the speaker has been quiet since the last member ended.
    """
    if not members:
        return None
    if session_ended:
        return CLOSE_SESSION_ENDED

    if span_ms(members) >= MAX_BLOCK_MS:
        if members[-1]["flush_reason"] == SIZE_CAP_FLUSH_REASON:
            logger.info("Local utterance group hit the ceiling on a size-capped cut; boundary may fall mid-word")
        return CLOSE_CEILING

    voice_ms = sum(member["voice_ms"] for member in members)
    if voice_ms < MIN_BLOCK_VOICE_MS:
        return None
    # Ending a request mid-word gives the model half a word to guess at -- the defect grouping
    # exists to remove. Only the ceiling above may override this.
    if members[-1]["flush_reason"] == SIZE_CAP_FLUSH_REASON:
        return None

    if silence_after_ms >= STOPPED_TALKING_MS:
        return CLOSE_STOPPED_TALKING
    if voice_ms >= TARGET_VOICE_MS and silence_after_ms >= MIN_BOUNDARY_PAUSE_MS:
        return CLOSE_ENOUGH_CONTEXT
    return None


def silence_since_last_member_ms(members, epoch_ms, end_offset_ms):
    """How long the speaker has been quiet since the last member ended.

    Measured against the audio actually processed so far, never a wall clock -- a slow drain or a
    retried task must not look like a pause and close a group early.
    """
    if not members:
        return 0
    last = members[-1]
    last_end_ms = last["timestamp_ms"] - epoch_ms + last["duration_ms"]
    return max(0, end_offset_ms - last_end_ms)
