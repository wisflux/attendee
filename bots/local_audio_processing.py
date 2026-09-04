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
from bots.local_silero_vad import LocalSileroVad
from bots.local_vad_verdict_cache import VerdictCache, verdicts_for_buffered
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
    """The bots' VAD manager with Silero in place of the amplitude gate.

    The shared ``silence_detected`` decides silence from loudness alone, below -40 dBFS,
    before any speech model is consulted -- and measured against the transcriber's own word
    timings that discards 22-52% of real speech, clipping the ends of words. Silero misses
    ~7%. The amplitude test is deliberately not kept in front of it: with the gate ahead,
    flagged silence goes back from 25s to 43s on a 94s recording, which is the gate's answer
    rather than Silero's, and the change buys nothing.

    Only the local path is affected. Meeting bots keep the shared implementation.
    """

    def __init__(self, *, vad_state=None, detector=None, cached_verdicts=None, **kwargs):
        super().__init__(**kwargs)
        # `detector` exists for tests. Silero rejects every synthetic signal -- tones, noise,
        # formant stacks -- which is correct behaviour and makes it useless for driving the
        # manager's own logic, so those tests script the verdicts instead. Silero's accuracy
        # is measured on real audio by bots/e2e_tests/vad_report.py.
        self.vad = detector or LocalSileroVad(kwargs["sample_rate"], state=vad_state)
        # A verdict already decided on an earlier drain is a fixed fact, not something to ask
        # the model again -- so the still-open buffer's known verdicts are consulted first, and
        # only genuinely new frames reach the detector. See local_vad_verdict_cache.
        self._verdict_cache = VerdictCache(cached_verdicts)
        # One verdict per frame fed, so the emitted audio can be trimmed without asking the
        # detector twice. Reconciled against the audio's own length at emit time, because the
        # base class decides which frames actually enter the buffer.
        self._verdicts = []
        self._emit_utterance = self.save_audio_chunk_callback
        self.save_audio_chunk_callback = self._shorten_then_emit

    def silence_detected(self, chunk_bytes):
        if not chunk_bytes:
            return True
        speaking = self._verdict_cache.next(lambda: self.vad.is_speech(chunk_bytes))
        self._verdicts.append(speaking)
        if speaking:
            return False
        self.diagnostic_info["total_chunks_marked_as_silent_due_to_vad"] += 1
        return True

    def buffered_verdicts(self, source):
        """Verdicts for whatever `source` still has open, unflushed -- for the next drain's
        cache. See `local_vad_verdict_cache.verdicts_for_buffered`."""
        frame_bytes = self.sample_rate // 100 * BYTES_PER_SAMPLE
        buffered_frames = len(self.utterances.get(source, b"")) // frame_bytes
        return verdicts_for_buffered(self._verdicts, buffered_frames)

    def _shorten_then_emit(self, message):
        """Drop the silence that closed the utterance, then shorten any long ones inside it.

        A pause is not padding -- the transcriber reads it as sentence structure, and removing
        one costs real content. Measured three ways on a real recording, deleting silence lost
        a surname, lost a company name, dropped an entire Hindi section and invented an ending.
        So an INTERNAL pause stays; only its MIDDLE is removed, and only when it runs past the
        limit. Cutting the middle means the splice happens silence-to-silence, so no word's
        onset or decay is ever clipped.

        The silence at the END is different: it is not structure, it is the flush trigger
        (see `frames_without_trailing_silence`), so it goes apart from a short pad.
        """
        audio = message["audio_data"]
        frame_bytes = self.sample_rate // 100 * BYTES_PER_SAMPLE
        frames = len(audio) // frame_bytes
        # The verdicts are the tail of everything fed: the base drops leading silence before a
        # buffer opens, so the last `frames` of them are this utterance's.
        verdicts, self._verdicts = self._verdicts[-frames:], []
        keep = frames_without_trailing_silence(verdicts)
        if keep < frames:
            audio, verdicts = audio[: keep * frame_bytes], verdicts[:keep]
        # How much of the clip is real speech, so a group can tell a run of conversation from a
        # run of pauses. Taken from the verdicts already recorded rather than measured again.
        voice_ms = sum(verdicts) * VAD_FRAME_MS
        self._emit_utterance({**message, "audio_data": shorten_long_silences(audio, verdicts, self.sample_rate), "voice_ms": voice_ms})

    def export_vad_state(self):
        """Handed to the Redis tail so the next drain resumes mid-stream, not cold."""
        return self.vad.export_state()

    def is_speech(self, chunk_bytes):
        """Kept so the shared manager's contract still holds; the local path uses Silero."""
        return self.vad.is_speech(chunk_bytes)


# How much of the closing silence to keep, so the final word's own decay is never clipped.
TRAILING_SILENCE_PAD_MS = 250


def frames_without_trailing_silence(verdicts):
    """How many leading frames of an utterance are worth sending.

    Every utterance ends with the silence that triggered its flush -- that pause is how the
    manager knows a sentence finished, so it is always in the buffer. Measured at roughly 30%
    of a typical clip and up to 80% of a short one, and a model handed mostly-silence fills it
    with invented text rather than returning nothing.

    Everything is kept when there is no speech at all: an utterance trimmed to nothing would
    be emitted carrying no audio, which is worse than sending the pause.
    """
    pad_frames = TRAILING_SILENCE_PAD_MS // VAD_FRAME_MS
    for index in range(len(verdicts) - 1, -1, -1):
        if verdicts[index]:
            return min(len(verdicts), index + 1 + pad_frames)
    return len(verdicts)


# A silence longer than this is shortened; anything shorter is left exactly as it is, because
# short gaps sit inside and between words and removing them corrupts the speech.
TRIM_SILENCE_OVER_MS = 3000
# ...down to this share of its length, split evenly between the two edges.
SILENCE_KEEP_FRACTION = 0.30


def shorten_long_silences(audio, verdicts, sample_rate):
    """Remove the middle of every silence past the limit, keeping both of its edges."""
    frame_bytes = sample_rate // 100 * BYTES_PER_SAMPLE
    frames = min(len(verdicts), len(audio) // frame_bytes)
    if frames == 0:
        return audio

    keep = [True] * frames
    run = 0
    for index in range(frames + 1):
        silent = (not verdicts[index]) if index < frames else False
        if silent:
            run += 1
            continue
        if run * VAD_FRAME_MS > TRIM_SILENCE_OVER_MS:
            edge = max(1, int(run * SILENCE_KEEP_FRACTION / 2))
            for position in range(index - run + edge, index - edge):
                keep[position] = False
        run = 0

    if all(keep):
        return audio
    kept = b"".join(audio[i * frame_bytes : (i + 1) * frame_bytes] for i in range(frames) if keep[i])
    return kept + audio[frames * frame_bytes :]


def duration_ms(audio, sample_rate):
    return int(len(audio) / ((sample_rate / 1000) * BYTES_PER_SAMPLE))


def create_utterance(recording, participant, message):
    """Mirror of BotController.process_individual_audio_chunk for a local session.

    Keyed by a deterministic source_uuid (a globally-unique column) so a task retry cannot
    duplicate an utterance it already wrote before failing.

    Returns the row, or None when it already existed. Transcription is NOT dispatched here any
    more: the drain gathers rows into a group and sends them as one request, so dispatching per
    utterance would both transcribe it twice and clear the audio the group still needs.
    """
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
    logger.info(f"Local session {recording.bot.object_id}: cut utterance {utterance.id} ({audio_chunk.duration_ms}ms)")
    return utterance


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


def build_manager(recording, participant, sample_rate, vad_state=None, cached_verdicts=None):
    """The manager, with `group_members` collecting what it emitted during this drain.

    Each member carries what the group decision needs -- how much of the clip was speech, how long
    it is, where it sits, and why it ended -- so the drain never has to re-read the audio.
    """
    group_members = []

    def save_audio_chunk_callback(message):
        utterance = create_utterance(recording, participant, message)
        if utterance is None:
            return
        group_members.append(
            {
                "utterance_id": utterance.id,
                "voice_ms": message["voice_ms"],
                "duration_ms": utterance.duration_ms,
                "timestamp_ms": utterance.timestamp_ms,
                "flush_reason": message["flush_reason"],
            }
        )

    def get_participant_callback(speaker_id):
        # Fixed for a local session; the manager only needs this to be non-None.
        return {"participant_uuid": participant.uuid, "participant_full_name": participant.full_name}

    manager = LocalAudioInputManager(
        vad_state=vad_state,
        cached_verdicts=cached_verdicts,
        save_audio_chunk_callback=save_audio_chunk_callback,
        get_participant_callback=get_participant_callback,
        sample_rate=sample_rate,
        utterance_size_limit=LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS * sample_rate * BYTES_PER_SAMPLE,
        silence_duration_limit=LOCAL_SILENCE_DURATION_LIMIT_SECONDS,
        should_print_diagnostic_info=False,
    )
    manager.split_at_size_limit = split_local_utterance
    manager.group_members = group_members
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
