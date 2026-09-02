"""Turning a local session's glued audio into utterances, reusing the bots' VAD.

This is the same webrtcVAD + utterance segmentation meeting bots use, with two local-only
adjustments: a loudness meter that does not overflow, and driving the VAD on the session's
own timeline (offsets) rather than the wall clock.
"""

import logging
from datetime import timedelta

import numpy as np

from bots.audio_split import BYTES_PER_SAMPLE, contains_speech, quietest_split_point
from bots.bot_controller.per_participant_non_streaming_audio_input_manager import (
    PerParticipantNonStreamingAudioInputManager,
)
from bots.models import AudioChunk, RecordingManager, Utterance

logger = logging.getLogger(__name__)

# A local recording is one person's device, so utterances are cut more aggressively than a
# meeting bot's: the desktop wants lines quickly, and short clips transcribe faster.
#
# 1.5s rather than 1.0s. An ordinary mid-sentence pause -- a breath, a moment's thinking --
# routinely runs past a second, and cutting there splits one thought into fragments that are
# each transcribed with no knowledge of the others: half-finished lines, punctuation that
# stops mid-clause, and a language re-guessed from a fragment too short to identify. The cost
# is 500ms of extra latency after a speaker stops, which is the whole of it -- an utterance
# still closes on silence, never on waiting for more speech.
LOCAL_SILENCE_DURATION_LIMIT_SECONDS = 1.5
LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS = 30

# webrtcvad only accepts 10/20/30ms frames, and the manager's "too large for VAD" guard
# compares a BYTE length against a SAMPLE count -- so anything over ~15ms of audio silently
# skips the VAD and is reported as speech. 10ms frames stay under that guard at every
# supported rate, which is also what the bot adapters happen to emit.
VAD_FRAME_MS = 10
INT16_FULL_SCALE = 32768.0


def normalized_rms(audio_bytes):
    """Loudness of a PCM frame, 0..1.

    The shared calculate_normalized_rms() squares an int16 array *in int16*, which wraps for
    any |sample| > 181 -- i.e. for all real speech -- so loud audio measures as silence.
    Widening to float64 first is what the streaming manager already does.
    """
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))) / INT16_FULL_SCALE)


class LocalAudioInputManager(PerParticipantNonStreamingAudioInputManager):
    """The bots' VAD manager, but with a loudness meter that does not overflow.

    Only ``silence_detected`` is overridden, and only to swap in the corrected RMS. The
    shared function is deliberately left alone so the bot path stays byte-identical; fixing
    it there is worth doing, but as its own change with its own testing.
    """

    def silence_detected(self, chunk_bytes):
        rms_value = normalized_rms(chunk_bytes)
        if rms_value == 0:
            self.diagnostic_info["total_chunks_marked_as_silent_due_to_rms_being_zero"] += 1
            return True
        if rms_value < 0.01:
            self.diagnostic_info["total_chunks_marked_as_silent_due_to_rms_being_small"] += 1
            return True
        if not self.is_speech(chunk_bytes):
            self.diagnostic_info["total_chunks_marked_as_silent_due_to_vad"] += 1
            return True
        return False


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


def split_local_utterance(audio, sample_rate):
    """Where to cut a local utterance that hit the size cap.

    Falls back to the end when the tail after the split would hold no speech. Carrying dead
    air forward opens the next utterance on silence, and because the silence timer measures
    from the last real speech -- which is behind the split -- that utterance flushes almost
    at once as a clip containing nothing but a pause. A near-silent clip is exactly the input
    that makes the transcriber invent a line, so this must not manufacture them.
    """
    point = quietest_split_point(audio, sample_rate)
    if point < len(audio) and not contains_speech(audio[point:]):
        return len(audio)
    return point


def build_manager(recording, participant, sample_rate):
    def save_audio_chunk_callback(message):
        create_utterance(recording, participant, message)

    def get_participant_callback(speaker_id):
        # Fixed for a local session; the manager only needs this to be non-None.
        return {"participant_uuid": participant.uuid, "participant_full_name": participant.full_name}

    manager = LocalAudioInputManager(
        save_audio_chunk_callback=save_audio_chunk_callback,
        get_participant_callback=get_participant_callback,
        sample_rate=sample_rate,
        utterance_size_limit=LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS * sample_rate * BYTES_PER_SAMPLE,
        silence_duration_limit=LOCAL_SILENCE_DURATION_LIMIT_SECONDS,
        should_print_diagnostic_info=False,
    )
    manager.split_at_size_limit = split_local_utterance
    return manager


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
