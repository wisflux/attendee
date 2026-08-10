"""Assemble a recording's transcript into the ``{speaker_name, text}`` rows the prompt expects.

Mirrors the query that ``TranscriptView`` serves (default recording, real utterances, time order)
so the summary reads the same transcript a member sees. See docs/meeting-summaries.
"""

from bots.models import Utterance


def collect_utterances(recording):
    """Ordered ``[{speaker_name, is_host, text}]`` for a recording, skipping empty transcriptions.

    Returns [] when there is no recording. ``speaker_name`` may be blank (single-mic local
    recordings) — the prompt handles that.
    """
    if recording is None:
        return []
    rows = []
    utterances = Utterance.objects.select_related("participant").filter(recording=recording, transcription__isnull=False, async_transcription=None).order_by("timestamp_ms")
    for utterance in utterances:
        text = (utterance.transcription or {}).get("transcript", "")
        if not text:
            continue
        participant = utterance.participant
        rows.append({"speaker_name": participant.full_name if participant else "", "is_host": bool(participant and participant.is_host), "text": text})
    return rows
