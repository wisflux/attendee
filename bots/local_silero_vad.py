"""Silero voice-activity detection for local sessions, with state that survives a drain.

Replaces the amplitude gate the local path used, which decided a frame was silence purely by
being quieter than -40 dBFS, before any speech model saw it. Scored against ElevenLabs' own
word timings on real recordings that gate throws away 22% of speech on a close-mic recording
and 52% on a quieter one, clipping the ends of words -- including a speaker's own company
name, which came back wrong in every transcript the old path produced. Silero misses ~7%.

The amplitude gate is deliberately gone rather than kept in front. Keeping it makes the whole
change pointless: measured, an RMS gate ahead of Silero drags the flagged silence from 25s
back to 43s on a 94s recording, which is the gate's answer, not Silero's.

STATE IS THE WHOLE DIFFICULTY. Silero is a recurrent model -- it tells a breath from a word by
remembering what came before. The drain task is a Celery task and holds nothing between runs,
so a detector built per drain restarts cold roughly once a second and never leaves its warm-up
regime. Measured, that costs ~13 seconds of false silence on a 90 second recording. So the
state rides along in the Redis tail beside the audio it belongs to: ~1.3 KB, serialised here
and restored on the next drain.
"""

import logging
import os
import threading

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "bot_controller", "vad_models", "silero_vad.onnx")

# Silero's contract, unchanged since v5.0: 512-sample windows at 16 kHz, each carrying the
# previous 64 samples as context, threaded through a (2, 1, 128) recurrent state.
TARGET_SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)

# Speech starts above ENTER and only ends below EXIT. The gap is hysteresis: without it a
# probability hovering around one number flips every window and shreds an utterance. Measured
# across 0.1-0.8 the miss rate moves by ~3 points, so neither value is a knife edge.
ENTER_THRESHOLD = 0.5
EXIT_THRESHOLD = 0.35

_session = None
_session_lock = threading.Lock()


def _model():
    """Load the graph once per process. InferenceSession.run is thread-safe."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                _session = ort.InferenceSession(MODEL_PATH, sess_options=options, providers=["CPUExecutionProvider"])
    return _session


class LocalSileroVad:
    """Frame-by-frame speech detection for one source, resumable across drains."""

    def __init__(self, sample_rate, state=None):
        if sample_rate % TARGET_SAMPLE_RATE != 0:
            raise ValueError(f"sample rate must be a multiple of {TARGET_SAMPLE_RATE}, got {sample_rate}")
        self._decimation = sample_rate // TARGET_SAMPLE_RATE
        self._buffer = np.empty(0, dtype=np.float32)
        self._remainder = np.empty(0, dtype=np.float32)
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        # Silence until a window has actually been evaluated. The opposite -- assuming speech
        # -- would open an utterance on the first frame of every drain.
        self._speaking = False
        if state:
            self._restore(state)

    def is_speech(self, chunk_bytes):
        """Whether this frame is speech. Frames too short to fill a window keep the last verdict."""
        self._buffer = np.concatenate((self._buffer, self._to_target_rate(chunk_bytes)))
        session = _model()
        rate = np.array(TARGET_SAMPLE_RATE, dtype=np.int64)
        while self._buffer.size >= WINDOW_SAMPLES:
            window = self._buffer[:WINDOW_SAMPLES]
            self._buffer = self._buffer[WINDOW_SAMPLES:]
            probability, self._state = session.run(
                None,
                {"input": np.concatenate((self._context, window)).reshape(1, -1), "state": self._state, "sr": rate},
            )
            self._context = window[-CONTEXT_SAMPLES:]
            value = float(probability.reshape(-1)[0])
            self._speaking = value >= EXIT_THRESHOLD if self._speaking else value >= ENTER_THRESHOLD
        return self._speaking

    def _to_target_rate(self, chunk_bytes):
        """Decimate to 16 kHz. Only this throwaway copy is downsampled; the audio sent for
        transcription is untouched."""
        samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if self._decimation == 1:
            return samples
        samples = np.concatenate((self._remainder, samples))
        usable = samples.size - (samples.size % self._decimation)
        self._remainder = samples[usable:]
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        return samples[:usable].reshape(-1, self._decimation).mean(axis=1)

    def export_state(self):
        """Everything needed to carry on mid-stream after the process that held it is gone."""
        return {
            "state": self._state.tobytes().hex(),
            "context": self._context.tobytes().hex(),
            "buffer": self._buffer.tobytes().hex(),
            "remainder": self._remainder.tobytes().hex(),
            "speaking": self._speaking,
        }

    def _restore(self, state):
        try:
            self._state = np.frombuffer(bytes.fromhex(state["state"]), dtype=np.float32).reshape(STATE_SHAPE).copy()
            self._context = np.frombuffer(bytes.fromhex(state["context"]), dtype=np.float32).copy()
            self._buffer = np.frombuffer(bytes.fromhex(state["buffer"]), dtype=np.float32).copy()
            self._remainder = np.frombuffer(bytes.fromhex(state["remainder"]), dtype=np.float32).copy()
            self._speaking = bool(state["speaking"])
        except (KeyError, ValueError, TypeError):
            # A malformed or stale blob is not worth failing a session over; starting cold
            # costs accuracy for a moment, and the next drain saves a good one.
            logger.warning("Discarding unusable Silero state; the detector will start cold")
            self.__init__(TARGET_SAMPLE_RATE * self._decimation)
