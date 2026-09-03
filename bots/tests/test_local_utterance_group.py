"""When a run of local utterances is worth sending to the transcriber as one request.

The live pipeline sends each utterance on its own, so the model sees 1-3 seconds at a time and
re-identifies the language on every fragment -- which is where wrong-language lines come from,
and why one sentence spoken across a breath arrives as two half-sentences that do not join up.

Grouping does not change how utterances are CUT. Each one keeps its own row, timestamp and
speaker; only the transcription request is shared. These tests pin the decision alone: given
what has been gathered and how long the speaker has been quiet, is the group ready?
"""

from django.test import TestCase

from bots.local_utterance_group import (
    CLOSE_CEILING,
    CLOSE_ENOUGH_CONTEXT,
    CLOSE_SESSION_ENDED,
    CLOSE_STOPPED_TALKING,
    GAP_CAP_MS,
    MAX_BLOCK_MS,
    MIN_BLOCK_VOICE_MS,
    MIN_BOUNDARY_PAUSE_MS,
    STOPPED_TALKING_MS,
    TARGET_VOICE_MS,
    close_reason,
    gaps_ms,
)

SILENCE_LIMIT_FLUSH = "silence_limit"
SIZE_CAP_FLUSH = "buffer_full"


def members(*durations, flush_reason=SILENCE_LIMIT_FLUSH, start_ms=0, gap_ms=0):
    """A group whose members run back to back, each carrying `duration` of pure speech."""
    out, cursor = [], start_ms
    for duration in durations:
        out.append({"voice_ms": duration, "duration_ms": duration, "timestamp_ms": cursor, "flush_reason": flush_reason})
        cursor += duration + gap_ms
    return out


class GroupCloseReasonTest(TestCase):
    def test_an_empty_group_is_never_ready(self):
        self.assertIsNone(close_reason([], silence_after_ms=STOPPED_TALKING_MS * 2))

    def test_a_group_below_the_voice_target_keeps_gathering(self):
        """The whole point is a long request; closing early recreates the fragment problem."""
        self.assertIsNone(close_reason(members(5000), silence_after_ms=MIN_BOUNDARY_PAUSE_MS))

    def test_enough_voice_and_a_real_pause_closes_the_group(self):
        group = members(TARGET_VOICE_MS)

        self.assertEqual(close_reason(group, silence_after_ms=MIN_BOUNDARY_PAUSE_MS), CLOSE_ENOUGH_CONTEXT)

    def test_enough_voice_without_a_real_pause_keeps_gathering(self):
        """Closing at the first micro-gap once the target is hit splits a sentence mid-phrase."""
        group = members(TARGET_VOICE_MS)

        self.assertIsNone(close_reason(group, silence_after_ms=MIN_BOUNDARY_PAUSE_MS - 1))

    def test_a_speaker_who_stops_does_not_wait_for_the_target(self):
        """Otherwise a short answer is held until the session ends."""
        group = members(2000)

        self.assertEqual(close_reason(group, silence_after_ms=STOPPED_TALKING_MS), CLOSE_STOPPED_TALKING)

    def test_a_group_holding_almost_no_speech_is_not_sent(self):
        """A near-silent request is exactly what makes the model invent a line."""
        group = members(MIN_BLOCK_VOICE_MS - 1)

        self.assertIsNone(close_reason(group, silence_after_ms=STOPPED_TALKING_MS))

    def test_the_ceiling_closes_a_group_that_never_pauses(self):
        """Somebody reading aloud never gives a boundary pause; the group cannot grow forever."""
        group = members(MAX_BLOCK_MS)

        self.assertEqual(close_reason(group, silence_after_ms=0), CLOSE_CEILING)

    def test_a_group_does_not_close_on_a_cut_made_mid_word(self):
        """An utterance ended by the size cap stops mid-syllable. Ending a request there gives
        the model half a word to guess at, which is the defect grouping exists to remove."""
        group = members(TARGET_VOICE_MS, flush_reason=SIZE_CAP_FLUSH)

        self.assertIsNone(close_reason(group, silence_after_ms=STOPPED_TALKING_MS))

    def test_the_ceiling_still_wins_over_a_mid_word_cut(self):
        """Better a bad boundary than a request that grows without bound."""
        group = members(MAX_BLOCK_MS, flush_reason=SIZE_CAP_FLUSH)

        self.assertEqual(close_reason(group, silence_after_ms=0), CLOSE_CEILING)

    def test_session_end_sends_whatever_is_held(self):
        """Nothing may be stranded when the recording stops."""
        group = members(1000)

        self.assertEqual(close_reason(group, silence_after_ms=0, session_ended=True), CLOSE_SESSION_ENDED)

    def test_session_end_on_an_empty_group_sends_nothing(self):
        self.assertIsNone(close_reason([], silence_after_ms=0, session_ended=True))

    def test_the_span_counts_silence_between_members_towards_the_ceiling(self):
        """The ceiling bounds the audio in one request, which includes the gaps inside it."""
        group = members(1000, 1000, gap_ms=MAX_BLOCK_MS)

        self.assertEqual(close_reason(group, silence_after_ms=0), CLOSE_CEILING)


class GapsBetweenMembersTest(TestCase):
    """How much silence is written between two utterances in one request.

    Real gaps, not a fixed spacer: a 3s spacer reads to the model as a full stop and undoes the
    context the group exists to buy. Zero would splice the end of one word onto the start of the
    next. So the real pause is used, capped.
    """

    def test_the_gap_is_the_real_silence_between_them(self):
        group = members(1000, 1000, gap_ms=400)

        self.assertEqual(gaps_ms(group), [400])

    def test_a_long_pause_is_capped(self):
        """Past the cap the model hears a full stop, which is what grouping is avoiding."""
        group = members(1000, 1000, gap_ms=9000)

        self.assertEqual(gaps_ms(group), [GAP_CAP_MS])

    def test_utterances_that_overlap_never_produce_negative_silence(self):
        """Trimming and rounding can leave one row ending after the next begins."""
        group = members(1000, 1000)
        group[1]["timestamp_ms"] = group[0]["timestamp_ms"] + 500

        self.assertEqual(gaps_ms(group), [0])

    def test_a_single_utterance_has_no_gaps(self):
        self.assertEqual(gaps_ms(members(1000)), [])

    def test_an_empty_group_has_no_gaps(self):
        self.assertEqual(gaps_ms([]), [])

    def test_there_is_one_gap_fewer_than_members(self):
        group = members(500, 500, 500, 500, gap_ms=200)

        self.assertEqual(len(gaps_ms(group)), len(group) - 1)
