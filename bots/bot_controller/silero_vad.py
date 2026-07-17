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
_WINDOW_SAMPLES = 512  # Silero v5 window size at 16 kHz
_CONTEXT_SAMPLES = 64  # Silero v5 prepends the previous 64 samples to each window
_STATE_SHAPE = (2, 1, 128)
_SPEECH_THRESHOLD = 0.5
_INT16_FULL_SCALE = 32768.0

_session = None
_session_lock = threading.Lock()


def _get_session():
    """Load the ONNX model once and share it (InferenceSession.run is thread-safe)."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                _session = ort.InferenceSession(_MODEL_PATH, sess_options=options, providers=["CPUExecutionProvider"])
    return _session


class _SpeakerStream:
    """Per-speaker buffer + Silero state for one continuous audio stream."""

    def __init__(self, sample_rate):
        if sample_rate % _TARGET_SAMPLE_RATE != 0:
            raise ValueError(f"Unsupported sample rate for Silero VAD: {sample_rate}")
        self._factor = sample_rate // _TARGET_SAMPLE_RATE  # 16k->1, 32k->2, 48k->3
        self._raw_remainder = np.empty(0, dtype=np.float32)
        self._buffer = np.empty(0, dtype=np.float32)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        self._last_is_speech = True  # fail open until a full window can be evaluated

    def is_speech(self, chunk_bytes):
        self._buffer = np.concatenate((self._buffer, self._to_target_rate(chunk_bytes)))

        session = _get_session()
        sample_rate_input = np.array(_TARGET_SAMPLE_RATE, dtype=np.int64)
        window_decision = None
        while self._buffer.size >= _WINDOW_SAMPLES:
            window = self._buffer[:_WINDOW_SAMPLES]
            self._buffer = self._buffer[_WINDOW_SAMPLES:]
            model_input = np.concatenate((self._context, window)).reshape(1, -1)
            probability, self._state = session.run(None, {"input": model_input, "state": self._state, "sr": sample_rate_input})
            self._context = window[-_CONTEXT_SAMPLES:]
            window_decision = float(probability.reshape(-1)[0]) >= _SPEECH_THRESHOLD

        if window_decision is not None:
            self._last_is_speech = window_decision
        return self._last_is_speech

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

    def __init__(self):
        self._streams = {}

    def is_speech(self, speaker_id, chunk_bytes, sample_rate):
        try:
            stream = self._streams.get(speaker_id)
            if stream is None:
                stream = _SpeakerStream(sample_rate)
                self._streams[speaker_id] = stream
            return stream.is_speech(chunk_bytes)
        except Exception as error:
            logger.exception("Silero VAD failed, treating chunk as speech: %s", error)
            return True

    def reset(self, speaker_id):
        self._streams.pop(speaker_id, None)
