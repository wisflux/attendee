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
    MAX_BLOCK_MS,
    MAX_GAP_MS,
    MIN_BLOCK_VOICE_MS,
    MIN_BOUNDARY_PAUSE_MS,
    SILENCE_KEEP_FRACTION,
    STOPPED_TALKING_MS,
    TARGET_VOICE_MS,
    TRIM_SILENCE_OVER_MS,
    close_reason,
    gaps_ms,
    gaps_seconds,
    silence_since_last_member_ms,
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

    def test_a_pause_short_enough_to_be_structure_is_kept_whole(self):
        """The reference blocks that produced the best transcripts were 16-42% silence. A pause
        IS the sentence structure -- deleting one was measured losing a surname and a company
        name -- so anything under the limit is reproduced exactly."""
        group = members(1000, 1000, gap_ms=TRIM_SILENCE_OVER_MS)

        self.assertEqual(gaps_ms(group), [TRIM_SILENCE_OVER_MS])

    def test_a_pause_long_enough_to_read_as_an_ending_is_shortened_not_removed(self):
        """Past the limit it stops being a breath and starts being a full stop, so it is cut to
        the same share the offline reference kept -- shortened, never deleted."""
        group = members(1000, 1000, gap_ms=10000)

        self.assertEqual(gaps_ms(group), [int(10000 * SILENCE_KEEP_FRACTION)])

    def test_a_pause_over_the_limit_but_under_the_bound_is_still_a_fraction(self):
        """The bound is a safety net for the pathological case, not a second rhythm rule: between
        the two thresholds the measured fraction still decides."""
        group = members(1000, 1000, gap_ms=20000)

        self.assertEqual(gaps_ms(group), [int(20000 * SILENCE_KEEP_FRACTION)])

    def test_a_pause_no_one_would_speak_across_is_bounded(self):
        """A group held open by the voice floor can meet its next member minutes later.

        A cough carries less than MIN_BLOCK_VOICE_MS, so its group never closes -- and the span
        ceiling cannot rescue it either, because the span only grows when a member is added. The
        next real speech joins whenever it comes, and a fraction of a pause that long is still a
        request that is mostly silence, which is what makes the model invent a line.
        """
        five_minutes_ms = 5 * 60 * 1000
        group = members(250, 4250, gap_ms=five_minutes_ms)

        self.assertEqual(gaps_ms(group), [MAX_GAP_MS])

    def test_utterances_that_overlap_never_produce_negative_silence(self):
        """Trimming and rounding can leave one row ending after the next begins."""
        group = members(1000, 1000)
        group[1]["timestamp_ms"] = group[0]["timestamp_ms"] + 500

        self.assertEqual(gaps_ms(group), [0])

    def test_a_single_utterance_has_no_gaps(self):
        self.assertEqual(gaps_ms(members(1000)), [])

    def test_an_empty_group_has_no_gaps(self):
        self.assertEqual(gaps_ms([]), [])

    def test_the_gap_is_offered_in_the_unit_its_consumer_reads(self):
        """get_mp3_for_utterance_group multiplies this by bytes-per-SECOND, so milliseconds here
        would write 1250 seconds of silence for a 1.25 second pause."""
        group = members(1000, 1000, gap_ms=1250)

        self.assertEqual(gaps_seconds(group), [1.25])

    def test_seconds_and_milliseconds_describe_the_same_pauses(self):
        """One list builds the audio and locates the words. Two units of it must not disagree."""
        group = members(1000, 2000, 1500, gap_ms=800)

        self.assertEqual(gaps_seconds(group), [gap / 1000.0 for gap in gaps_ms(group)])

    def test_there_is_one_gap_fewer_than_members(self):
        group = members(500, 500, 500, 500, gap_ms=200)

        self.assertEqual(len(gaps_ms(group)), len(group) - 1)


class SilenceAfterGroupTest(TestCase):
    """How long the speaker has been quiet since the last member ended.

    This is what closes a group when somebody stops talking, so it is measured against the audio
    actually processed so far rather than a wall clock -- a slow drain must not look like a pause.
    """

    EPOCH_MS = 1_700_000_000_000

    def _members(self, *durations, gap_ms=0):
        return members(*durations, gap_ms=gap_ms, start_ms=self.EPOCH_MS)

    def test_silence_is_measured_from_the_end_of_the_last_member(self):
        group = self._members(1000)

        self.assertEqual(silence_since_last_member_ms(group, self.EPOCH_MS, end_offset_ms=4000), 3000)

    def test_no_silence_while_audio_is_still_arriving_inside_the_last_member(self):
        """A drain can process up to the middle of a clip; that is not a pause."""
        group = self._members(1000)

        self.assertEqual(silence_since_last_member_ms(group, self.EPOCH_MS, end_offset_ms=500), 0)

    def test_an_empty_group_has_no_silence_to_report(self):
        self.assertEqual(silence_since_last_member_ms([], self.EPOCH_MS, end_offset_ms=9000), 0)

    def test_only_the_last_member_matters(self):
        group = self._members(1000, 1000, gap_ms=500)

        self.assertEqual(silence_since_last_member_ms(group, self.EPOCH_MS, end_offset_ms=4000), 1500)
