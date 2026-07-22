"""A small, user-facing status for a meeting, derived from data we already store.

Bot.state has 19 values that describe pod plumbing (JOINING, WAITING_ROOM, POST_PROCESSING,
...). None of that is meaningful to someone looking at their own recordings, so this maps the
raw state onto the handful of things a person actually cares about.

Nothing here writes: it is a pure function of the bot, its recording and its utterance counts,
so it cannot race the lifecycle or invent a state the state machine doesn't have. "Still
transcribing" reuses the predicate the bot controller already uses -- an utterance with no
transcription *and* no failure recorded (see BotController.utterances filter).
"""

from bots.models import BotStates, RecordingStates

RECORDING = "recording"
PAUSED = "paused"
FINALIZING = "finalizing"
PROCESSING = "processing"
DONE = "done"
PARTIALLY_FAILED = "partially_failed"
FAILED = "failed"
DELETED = "deleted"


def transcription_counts(recording):
    """(pending, failed) utterance counts for a recording."""
    if recording is None:
        return 0, 0
    utterances = recording.utterances
    pending = utterances.filter(transcription__isnull=True, failure_data__isnull=True).count()
    failed = utterances.filter(failure_data__isnull=False).count()
    return pending, failed


def derive_status(bot, recording, pending=None, failed=None):
    """Map a meeting onto one user-facing status.

    Pass pending/failed when they were already fetched in bulk, to avoid a query per row.
    """
    if bot.state == BotStates.DATA_DELETED:
        return DELETED
    if bot.state == BotStates.FATAL_ERROR:
        return FAILED

    if bot.state not in BotStates.post_meeting_states():
        return _live_status(recording)

    if pending is None or failed is None:
        pending, failed = transcription_counts(recording)
    if pending:
        return PROCESSING
    if failed:
        # Some clips could not be transcribed; the rest of the transcript is still usable, so
        # this is deliberately not a hard failure.
        return PARTIALLY_FAILED
    if recording is not None and recording.state == RecordingStates.FAILED:
        return FAILED
    return DONE


def _live_status(recording):
    """A session that hasn't reached a post-meeting state yet.

    Finalization marks the recording COMPLETE *before* the session ends (so a file-less local
    recording isn't terminated as FAILED), and that gap is exactly "catching the last words".
    """
    if recording is None:
        return RECORDING
    if recording.state == RecordingStates.PAUSED:
        return PAUSED
    if recording.state == RecordingStates.COMPLETE:
        return FINALIZING
    return RECORDING
