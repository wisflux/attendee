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
from bots.bot_controller.silero_vad import SileroVoiceActivityDetector
from bots.local_vad_params import LocalVadParams
from bots.models import AudioChunk, RecordingManager, Utterance

logger = logging.getLogger(__name__)

# How long an utterance may run before it is cut regardless of silence.
LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS = 30

# The VAD is driven a frame at a time. Silero itself needs 32 ms windows, but it accumulates
# them internally, so this only sets how finely the session timeline is walked.
VAD_FRAME_MS = 10
BYTES_PER_SAMPLE = 2
MS_PER_SECOND = 1000


class LocalAudioInputManager(PerParticipantNonStreamingAudioInputManager):
    """The bots' VAD manager with the local session's own tuning.

    Two local-only behaviours live here so the meeting-bot path keeps the detector's defaults:
    a configured speech threshold with hysteresis, and a minimum amount of voiced audio before
    an utterance is worth transcribing.

    One manager instance serves exactly one audio source -- ``build_manager`` is called per
    source per drain -- so the voiced-audio tally does not need to be keyed by speaker.
    """

    def __init__(self, *, params, **kwargs):
        super().__init__(**kwargs)
        self._params = params
        self.vad = SileroVoiceActivityDetector(
            threshold=params.threshold,
            hysteresis_offset=params.hysteresis_offset,
        )
        self._voiced_ms = 0.0
        self._voiced_end_bytes = 0
        # Wrap rather than override process_chunk: the base decides when to flush, and this
        # only decides whether the flushed result is worth keeping.
        self._emit_utterance = self.save_audio_chunk_callback
        self.save_audio_chunk_callback = self._emit_if_enough_speech

    def silence_detected(self, speaker_id, chunk_bytes):
        is_silent = super().silence_detected(speaker_id, chunk_bytes)
        if not is_silent and chunk_bytes:
            self._voiced_ms += duration_ms(chunk_bytes, self.sample_rate)
            # The base appends this frame straight after asking us, so the buffer's current
            # length is exactly where the frame will start.
            buffered = len(self.utterances.get(speaker_id, b""))
            self._voiced_end_bytes = buffered + len(chunk_bytes)
        return is_silent

    def _emit_if_enough_speech(self, message):
        voiced_ms, self._voiced_ms = self._voiced_ms, 0.0
        voiced_end_bytes, self._voiced_end_bytes = self._voiced_end_bytes, 0
        if voiced_ms < self._params.min_speech_ms:
            logger.info(f"Dropping utterance with only {voiced_ms:.0f}ms of speech (minimum {self._params.min_speech_ms}ms)")
            return
        self._emit_utterance({**message, "audio_data": self._trim_trailing_silence(message["audio_data"], voiced_end_bytes)})

    def _trim_trailing_silence(self, audio, voiced_end_bytes):
        """Cut the dead air off the end, keeping a short natural tail.

        An utterance always ends with the full silence limit, because that is what triggers
        the flush. Sending it costs transcription time and, more importantly, gives the model
        a stretch of nothing to fill with invented text.

        Only trailing silence is removed. Pauses inside the utterance are left alone -- they
        carry timing the transcript depends on, and removing them would make the word
        timestamps disagree with the audio.
        """
        if not voiced_end_bytes:
            return audio
        keep = self._params.trailing_keep_ms * self.sample_rate // MS_PER_SECOND * BYTES_PER_SAMPLE
        # Round to a whole sample: an odd byte count is not decodable PCM16.
        end = min(len(audio), (voiced_end_bytes + keep) // BYTES_PER_SAMPLE * BYTES_PER_SAMPLE)
        if end >= len(audio):
            return audio
        logger.info(f"Trimmed {duration_ms(audio[end:], self.sample_rate)}ms of trailing silence")
        return audio[:end]


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


def build_manager(recording, participant, sample_rate, params=None):
    params = params or LocalVadParams.from_env()

    def save_audio_chunk_callback(message):
        create_utterance(recording, participant, message)

    def get_participant_callback(speaker_id):
        # Fixed for a local session; the manager only needs this to be non-None.
        return {"participant_uuid": participant.uuid, "participant_full_name": participant.full_name}

    return LocalAudioInputManager(
        params=params,
        save_audio_chunk_callback=save_audio_chunk_callback,
        get_participant_callback=get_participant_callback,
        sample_rate=sample_rate,
        utterance_size_limit=LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS * sample_rate * BYTES_PER_SAMPLE,
        silence_duration_limit=params.min_silence_seconds,
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
    # One second past the manager's own limit, so the probe always trips the flush no matter
    # how the silence limit is configured.
    probe_at = epoch + timedelta(milliseconds=end_offset_ms, seconds=manager.SILENCE_DURATION_LIMIT + 1)
    manager.process_chunk(source, probe_at, None)
