"""Turning a local session's glued audio into utterances, reusing the bots' VAD.

This is the same webrtcVAD + utterance segmentation meeting bots use, with one local-only
adjustment: the VAD is driven on the session's own timeline (offsets) rather than the wall
clock. The loudness workaround that used to live here is gone -- the shared
calculate_normalized_rms() is correct now, so the base class already does the right thing.
"""

import logging
from datetime import timedelta

from bots.bot_controller.per_participant_non_streaming_audio_input_manager import (
    PerParticipantNonStreamingAudioInputManager,
)
from bots.models import AudioChunk, RecordingManager, Utterance

logger = logging.getLogger(__name__)

# A local recording is one person's device, so utterances are cut more aggressively than a
# meeting bot's: the desktop wants lines quickly, and short clips transcribe faster.
LOCAL_SILENCE_DURATION_LIMIT_SECONDS = 1
LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS = 30

# webrtcvad only accepts 10/20/30ms frames, and the manager's "too large for VAD" guard
# compares a BYTE length against a SAMPLE count -- so anything over ~15ms of audio silently
# skips the VAD and is reported as speech. 10ms frames stay under that guard at every
# supported rate, which is also what the bot adapters happen to emit.
VAD_FRAME_MS = 10
BYTES_PER_SAMPLE = 2


class LocalAudioInputManager(PerParticipantNonStreamingAudioInputManager):
    """The bots' VAD manager, used as-is.

    This subclass used to override ``silence_detected`` purely to swap in a loudness meter
    that did not overflow. That fix now lives in ``bots.audio_utils`` and the base class uses
    it, so the override is gone. The class is kept as the named seam for local-only VAD
    behaviour, which the segmentation work will use.
    """


def duration_ms(audio, sample_rate):
    return int(len(audio) / ((sample_rate / 1000) * BYTES_PER_SAMPLE))


def create_utterance(recording, participant, message):
    """Mirror of BotController.process_individual_audio_chunk for a local session.

    Keyed by a deterministic source_uuid (a globally-unique column) so a task retry cannot
    duplicate an utterance it already wrote before failing.
    """
    from bots.tasks.process_utterance_task import process_utterance

    audio_data = message["audio_data"]
    sample_rate = message["sample_rate"]
    source_uuid = f"local:{recording.id}:{participant.uuid}:{message['timestamp_ms']}"

    if Utterance.objects.filter(source_uuid=source_uuid).exists():
        logger.info(f"Local session {recording.bot.object_id}: utterance {source_uuid} already exists, skipping")
        return

    audio_chunk = AudioChunk.objects.create(
        recording=recording,
        audio_format=AudioChunk.AudioFormat.PCM,
        timestamp_ms=message["timestamp_ms"],
        duration_ms=duration_ms(audio_data, sample_rate),
        sample_rate=sample_rate,
        source=AudioChunk.Sources.PER_PARTICIPANT_AUDIO,
        participant=participant,
        is_blob_stored_remotely=False,
        audio_blob=audio_data,
    )
    utterance = Utterance.objects.create(
        source=Utterance.Sources.PER_PARTICIPANT_AUDIO,
        async_transcription=None,
        recording=recording,
        participant=participant,
        audio_chunk=audio_chunk,
        timestamp_ms=audio_chunk.timestamp_ms,
        duration_ms=audio_chunk.duration_ms,
        source_uuid=source_uuid,
    )

    RecordingManager.set_recording_transcription_in_progress(recording)
    process_utterance.delay(utterance.id)
    logger.info(f"Local session {recording.bot.object_id}: queued utterance {utterance.id} ({audio_chunk.duration_ms}ms)")


def build_manager(recording, participant, sample_rate):
    def save_audio_chunk_callback(message):
        create_utterance(recording, participant, message)

    def get_participant_callback(speaker_id):
        # Fixed for a local session; the manager only needs this to be non-None.
        return {"participant_uuid": participant.uuid, "participant_full_name": participant.full_name}

    return LocalAudioInputManager(
        save_audio_chunk_callback=save_audio_chunk_callback,
        get_participant_callback=get_participant_callback,
        sample_rate=sample_rate,
        utterance_size_limit=LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS * sample_rate * BYTES_PER_SAMPLE,
        silence_duration_limit=LOCAL_SILENCE_DURATION_LIMIT_SECONDS,
        should_print_diagnostic_info=False,
    )


def feed(manager, source, audio, epoch, start_offset_ms, sample_rate):
    """Drive the VAD frame by frame on the session's own timeline.

    process_chunk() is called directly rather than add_chunk()/process_chunks(), because
    process_chunks() probes for silence against datetime.utcnow() -- wall clock -- which
    would force-flush this timeline instead of following it. Returns the byte count consumed
    (whole frames); a trailing partial frame rolls into the next drain.
    """
    frame_bytes = int(sample_rate * VAD_FRAME_MS / 1000) * BYTES_PER_SAMPLE
    for offset in range(0, len(audio), frame_bytes):
        frame = audio[offset : offset + frame_bytes]
        if len(frame) < frame_bytes:
            break
        frame_at = epoch + timedelta(milliseconds=start_offset_ms + (offset // frame_bytes) * VAD_FRAME_MS)
        manager.process_chunk(source, frame_at, frame)
    return len(audio) - (len(audio) % frame_bytes)


def offset_of_buffered(manager, source, epoch):
    """Where the still-buffered utterance began, as ms since the session started."""
    started_at = manager.first_nonsilent_audio_time.get(source)
    if started_at is None:
        return None
    return int((started_at - epoch).total_seconds() * 1000)


def flush_remaining(manager, source, epoch, end_offset_ms):
    """Emit whatever is still buffered, on our timeline rather than the wall clock.

    flush_utterances() probes with datetime.utcnow(); against a session-relative timeline
    that can compute negative silence and silently drop the final utterance, so the probe is
    placed just past the end of the audio we actually have.
    """
    if not manager.utterances.get(source):
        return
    probe_at = epoch + timedelta(milliseconds=end_offset_ms, seconds=LOCAL_SILENCE_DURATION_LIMIT_SECONDS + 1)
    manager.process_chunk(source, probe_at, None)
