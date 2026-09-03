"""Transcribe one group of local utterances in a single request and write the words back.

WHY NOTHING CAN BE LOST HERE. The audio lives on the AudioChunk rows from the moment a clip is
cut, and the group is only ever a list of ids. So a dropped Redis key or a dead worker costs the
grouping, never the speech; a replayed task is a no-op because the rows already hold their text;
and a group that cannot be transcribed falls back to one request per utterance, which is exactly
what the pipeline did before grouping existed.
"""

import logging

from celery import shared_task

from bots.models import RecordingManager, Utterance
from bots.tasks.process_utterance_task import process_utterance
from bots.transcription_providers.elevenlabs import get_transcription_via_elevenlabs_for_utterance_group
from bots.transcription_utils import is_retryable_failure

logger = logging.getLogger(__name__)

MAX_GROUP_ATTEMPTS = 3


@shared_task(
    bind=True,
    soft_time_limit=600,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=MAX_GROUP_ATTEMPTS,
)
def process_local_utterance_group(self, utterance_ids, gaps_seconds):
    if not utterance_ids:
        return

    by_id = Utterance.objects.select_related("recording").in_bulk(utterance_ids)
    # Order is positional: `gaps_seconds` describes the silence between consecutive members, so a
    # set-ordered fetch would misplace every word. Rows deleted by delete_data simply drop out.
    utterances = [by_id[utterance_id] for utterance_id in utterance_ids if utterance_id in by_id]
    if not utterances:
        logger.info(f"Local utterance group {utterance_ids}: rows are gone, dropping")
        return

    if all(utterance.transcription is not None for utterance in utterances):
        logger.info(f"Local utterance group {utterance_ids}: already transcribed, skipping")
        return

    transcriptions, failure_data = get_transcription_via_elevenlabs_for_utterance_group(utterances, gaps_seconds=gaps_seconds)

    if failure_data:
        if is_retryable_failure(failure_data) and self.request.retries < MAX_GROUP_ATTEMPTS:
            raise Exception(f"Retryable failure transcribing local utterance group {utterance_ids}: {failure_data}")
        # Degrade rather than lose the speech: the audio is untouched, so the per-utterance path
        # can still transcribe every clip the way it did before grouping.
        logger.warning(f"Local utterance group {utterance_ids} failed ({failure_data}); falling back to one request per utterance")
        for utterance in utterances:
            process_utterance.delay(utterance.id)
        return

    for utterance in utterances:
        transcription = transcriptions.get(utterance.id)
        if transcription is None:
            continue
        utterance.transcription = transcription
        utterance.save()
        # Only now: clearing before the group lands would leave a retry with nothing to send.
        if utterance.audio_chunk:
            utterance.audio_chunk.clear_audio_data()

    RecordingManager.set_recording_transcription_in_progress(utterances[0].recording)
    logger.info(f"Local utterance group {utterance_ids}: transcription complete")
