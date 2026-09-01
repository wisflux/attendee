"""Silero VAD -- a neural replacement for ``webrtcvad`` in the audio managers.

Silero gives much better speech/silence discrimination in noisy audio than
webrtcvad. Meeting audio is delivered chunk-by-chunk and interleaved across
speakers, and Silero needs *streaming continuity* to be accurate, so this
detector keeps a small buffer plus model state per speaker::

    vad = SileroVoiceActivityDetector()
    speaking = vad.is_speech(speaker_id, pcm_bytes, sample_rate)  # bool
    vad.reset(speaker_id)   # when the speaker's utterance ends

Details hidden from the caller so the surrounding pipeline stays unchanged:

* Silero (v5) only accepts 16 kHz audio in fixed 512-sample windows, each with
  the previous 64 samples prepended. Meeting audio arrives at 16/32/48 kHz, so
  each chunk is decimated to 16 kHz (integer factor 1/2/3) and sliced into
  windows internally. Only this throwaway copy is downsampled -- the audio sent
  to transcription is untouched.
* State and context are carried across a speaker's chunks (the streaming usage
  Silero expects); ``reset`` releases them when the utterance ends.
* On any error the detector fails OPEN (reports speech) so audio is never
  dropped -- matching the previous webrtcvad error handling.
"""

import logging
import os
import threading

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "vad_models", "silero_vad.onnx")
_TARGET_SAMPLE_RATE = 16000
# The vendored model is silero-vad 6.2.1 (see vad_models/README.md). This window/context
# contract has been fixed since v5.0: any other window size raises inside the model.
_WINDOW_SAMPLES = 512
_CONTEXT_SAMPLES = 64
_STATE_SHAPE = (2, 1, 128)

# The detector's own defaults, used by the meeting-bot path. Local sessions pass their own --
# see bots/local_vad_params.py -- so tuning a local recording cannot change bot segmentation.
_SPEECH_THRESHOLD = 0.5
# 0.0 disables hysteresis, which keeps the bot path's decisions byte-identical to before.
_HYSTERESIS_OFFSET = 0.0
_INT16_FULL_SCALE = 32768.0

_session = None
_session_lock = threading.Lock()


class SileroModelUnavailable(RuntimeError):
    """The model itself cannot be used, so every chunk would fail the same way.

    Kept distinct from a per-chunk error because the fail-open path must not swallow it:
    a detector that answers "speech" to everything is indistinguishable from no VAD at all,
    and would look healthy while silently undoing the whole point of this module.
    """


def _get_session():
    """Load the ONNX model once and share it (InferenceSession.run is thread-safe)."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                try:
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    _session = ort.InferenceSession(_MODEL_PATH, sess_options=options, providers=["CPUExecutionProvider"])
                except Exception as error:
                    raise SileroModelUnavailable(f"could not load {_MODEL_PATH}: {error}") from error
    return _session


class _SpeakerStream:
    """Per-speaker buffer + Silero state for one continuous audio stream."""

    def __init__(self, sample_rate, threshold=_SPEECH_THRESHOLD, hysteresis_offset=_HYSTERESIS_OFFSET):
        if sample_rate % _TARGET_SAMPLE_RATE != 0:
            raise ValueError(f"Unsupported sample rate for Silero VAD: {sample_rate}")
        self._threshold = threshold
        self._exit_threshold = threshold - hysteresis_offset
        self._factor = sample_rate // _TARGET_SAMPLE_RATE  # 16k->1, 32k->2, 48k->3
        self._raw_remainder = np.empty(0, dtype=np.float32)
        self._buffer = np.empty(0, dtype=np.float32)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        # Fail open until 512 samples have accumulated: a false 'speech' costs one padded
        # frame, a false 'silence' costs the first word of an utterance.
        self._last_is_speech = True

    def is_speech(self, chunk_bytes):
        self._buffer = np.concatenate((self._buffer, self._to_target_rate(chunk_bytes)))

        session = _get_session()
        sample_rate_input = np.array(_TARGET_SAMPLE_RATE, dtype=np.int64)
        while self._buffer.size >= _WINDOW_SAMPLES:
            window = self._buffer[:_WINDOW_SAMPLES]
            self._buffer = self._buffer[_WINDOW_SAMPLES:]
            model_input = np.concatenate((self._context, window)).reshape(1, -1)
            probability, self._state = session.run(None, {"input": model_input, "state": self._state, "sr": sample_rate_input})
            self._context = window[-_CONTEXT_SAMPLES:]
            self._apply(float(probability.reshape(-1)[0]))

        # A chunk too short to complete a window leaves the last decision standing.
        return self._last_is_speech

    def _apply(self, probability):
        """Update the speech decision for one window, with hysteresis.

        Speech starts above the threshold and ends below the exit threshold. With a single
        threshold, a probability hovering near it flips every window and chops one utterance
        into fragments. The state is updated per window rather than per chunk, because one
        chunk can complete several windows and each must see the decision the previous one
        left behind. Equal thresholds disable the hysteresis, which is how the bot path keeps
        its previous behaviour.
        """
        if self._last_is_speech:
            self._last_is_speech = probability >= self._exit_threshold
        else:
            self._last_is_speech = probability >= self._threshold

    def _to_target_rate(self, chunk_bytes):
        samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / _INT16_FULL_SCALE
        if self._factor == 1:
            return samples
        samples = np.concatenate((self._raw_remainder, samples))
        usable = samples.size - (samples.size % self._factor)
        self._raw_remainder = samples[usable:]
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        return samples[:usable].reshape(-1, self._factor).mean(axis=1)


class SileroVoiceActivityDetector:
    """Streaming Silero VAD holding one state per speaker."""

    def __init__(self, threshold=_SPEECH_THRESHOLD, hysteresis_offset=_HYSTERESIS_OFFSET):
        self._threshold = threshold
        self._hysteresis_offset = hysteresis_offset
        self._streams = {}

    def is_speech(self, speaker_id, chunk_bytes, sample_rate):
        """Whether this chunk contains speech. Fails open, except when the model is unusable.

        Failing open on a transient error keeps audio rather than dropping it. But a model
        that cannot load, or a sample rate the detector cannot handle, would fail open on
        *every* chunk -- silently reverting the pipeline to no VAD at all while looking
        healthy. Those two are raised so the failure is visible.
        """
        stream = self._streams.get(speaker_id)
        if stream is None:
            # Raises on a rate that is not a multiple of 16 kHz. Deliberately not caught:
            # every subsequent chunk would fail the same way.
            stream = _SpeakerStream(sample_rate, self._threshold, self._hysteresis_offset)
            self._streams[speaker_id] = stream
        try:
            return stream.is_speech(chunk_bytes)
        except SileroModelUnavailable:
            raise
        except Exception as error:
            logger.exception("Silero VAD failed on one chunk, treating it as speech: %s", error)
            return True

    def reset(self, speaker_id):
        self._streams.pop(speaker_id, None)
