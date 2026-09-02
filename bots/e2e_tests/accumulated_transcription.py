"""Best-known-good local transcription pipeline, reproducible from an audio file.

This is the configuration that produced the best transcripts across four real recordings --
correct names in both scripts, no invented lines, no audio-event tags, no words cut in half.
It is kept here so the result can be reproduced and fallen back to, and so any future change
has something to be measured against.

    python bots/e2e_tests/accumulated_transcription.py recording.m4a [more.m4a ...]
    python bots/e2e_tests/accumulated_transcription.py --dry-run clip.wav   # no API spend

Deliberately self-contained: it drives the ONNX graph directly and touches no production
code, so running it cannot affect a live session.

WHAT IT DOES DIFFERENTLY FROM THE LIVE PIPELINE

1. Silero instead of the RMS gate. The live gate calls a frame silent below -40 dBFS before
   any speech model is consulted, which on real recordings mislabels 22-52% of speech as
   silence and clips the ends of words. Measured against ElevenLabs' own word timings,
   Silero misses ~7%.

2. Accumulate before sending. The live pipeline sends each utterance as it closes, so the
   model sees 1-3 seconds at a time and re-guesses the language on every one. Here audio is
   collected until there is TARGET_VOICE_MS of real speech, so each request carries a whole
   stretch of conversation.

3. Close only on a real pause. Closing at the first micro-gap after the target splits
   sentences mid-phrase -- an earlier version cut between "the best" and "person in the
   world" at a 50ms gap, and both halves were then guessed at.

4. Shorten long silences instead of removing them. Cutting silence out entirely destroys the
   transcript: tested three ways, it lost a speaker's surname, lost a company name, dropped
   an entire Hindi section and invented an ending. The pauses carry sentence structure. Only
   the MIDDLE of a long silence is removed, so no word edge is ever cut.

KNOWN REMAINING DEFECTS

* A block with little speech spread over a long span can make the model loop, repeating one
  word ("Charging. Charging"). Trailing-silence trimming does not fix it.
* Output is not deterministic between runs; the same audio can transcribe differently. Pin
  `seed` before using this to compare two changes.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import django
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendee.settings.development")
django.setup()

import onnxruntime as ort  # noqa: E402
import requests  # noqa: E402

from bots.models import Credentials  # noqa: E402
from bots.utils import pcm_to_mp3  # noqa: E402

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_BYTES = SAMPLE_RATE // 100 * 2

# Silero's contract, fixed since v5.0: 512-sample windows with the previous 64 prepended.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "bot_controller", "vad_models", "silero_vad.onnx")
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)

# Speech starts above ENTER and ends below EXIT. The gap stops a probability hovering near
# one threshold from chopping an utterance into fragments. Measured: the miss rate barely
# moves between 0.1 and 0.8, so this is not a knife edge.
ENTER_THRESHOLD = 0.5
EXIT_THRESHOLD = 0.35

# How much real speech to gather before sending. The whole point: a request carrying half a
# minute of conversation identifies its language once, from plenty of evidence, instead of
# re-guessing from a two-second fragment.
TARGET_VOICE_MS = 30000
# ...but only close on a pause this long. Any shorter and the boundary lands mid-phrase.
MIN_BOUNDARY_PAUSE_MS = 700
# They stopped talking: send what we have rather than waiting for a target we may never reach.
STOPPED_TALKING_MS = 5000
# Nothing waits forever, however long somebody talks.
MAX_BLOCK_MS = 90000

# Silences longer than this are shortened, never removed.
TRIM_SILENCE_OVER_MS = 3000
# ...to this share of their length, split evenly between the two edges, so the cut happens
# silence-to-silence and no word onset or decay is ever clipped.
SILENCE_KEEP_FRACTION = 0.30

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
MODEL_ID = "scribe_v2"


def decode_to_pcm(path):
    """Anything ffmpeg understands -> 16 kHz mono PCM16."""
    out = os.path.join(tempfile.mkdtemp(), "audio.raw")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", out],
        check=True,
        capture_output=True,
    )
    return open(out, "rb").read()


def speech_per_frame(audio, frames):
    """One voice/silence verdict per 10ms frame, from Silero run over the whole file.

    State is carried from start to finish. Rebuilding it mid-stream -- which the live drain
    task would do, once per upload -- costs about 13 seconds of false silence on a 90 second
    recording, because the model never leaves its warm-up regime.
    """
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    rate = np.array(SAMPLE_RATE, dtype=np.int64)

    probabilities = []
    for offset in range(0, len(samples) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        window = samples[offset : offset + WINDOW_SAMPLES]
        probability, state = session.run(
            None,
            {"input": np.concatenate((context, window)).reshape(1, -1), "state": state, "sr": rate},
        )
        context = window[-CONTEXT_SAMPLES:]
        probabilities.append(float(probability.reshape(-1)[0]))

    speaking, verdicts = False, []
    for index in range(frames):
        probability = probabilities[min(int(index * FRAME_MS / 32), len(probabilities) - 1)]
        speaking = probability >= EXIT_THRESHOLD if speaking else probability >= ENTER_THRESHOLD
        verdicts.append(speaking)
    return verdicts


def collect_blocks(voice, frames):
    """Group frames into blocks that each carry enough speech to be worth transcribing."""
    blocks, start, voice_ms, silence_ms = [], None, 0, 0
    for index in range(frames):
        if voice[index]:
            if start is None:
                start = index
            voice_ms += FRAME_MS
            silence_ms = 0
        elif start is not None:
            silence_ms += FRAME_MS
        if start is None:
            continue

        enough = voice_ms >= TARGET_VOICE_MS and silence_ms >= MIN_BOUNDARY_PAUSE_MS
        stopped = silence_ms >= STOPPED_TALKING_MS
        if enough or stopped or (index - start + 1) * FRAME_MS >= MAX_BLOCK_MS:
            blocks.append((start, index + 1, voice_ms, "enough context" if enough else ("stopped talking" if stopped else "ceiling")))
            start, voice_ms, silence_ms = None, 0, 0
    if start is not None:
        blocks.append((start, frames, voice_ms, "flush at end"))
    return blocks


def shorten_long_silences(audio, voice, start, end):
    """Remove the middle of any silence over the limit, keeping both edges intact."""
    keep = [True] * (end - start)
    run = 0
    for offset in range(end - start + 1):
        silent = (not voice[start + offset]) if offset < end - start else False
        if silent:
            run += 1
            continue
        if run * FRAME_MS > TRIM_SILENCE_OVER_MS:
            edge = max(1, int(run * SILENCE_KEEP_FRACTION / 2))
            for index in range(offset - run + edge, offset - edge):
                keep[index] = False
        run = 0
    kept = b"".join(audio[(start + offset) * FRAME_BYTES : (start + offset + 1) * FRAME_BYTES] for offset in range(end - start) if keep[offset])
    return kept, sum(keep) * FRAME_MS


def transcribe(pcm, api_key):
    response = requests.post(
        ELEVENLABS_URL,
        headers={"xi-api-key": api_key},
        files={"file": ("audio.mp3", pcm_to_mp3(pcm, sample_rate=SAMPLE_RATE), "audio/mpeg")},
        data={"model_id": MODEL_ID, "tag_audio_events": False},
    )
    response.raise_for_status()
    return response.json().get("text", "")


def process(path, api_key):
    audio = decode_to_pcm(path)
    frames = len(audio) // FRAME_BYTES
    voice = speech_per_frame(audio, frames)
    blocks = collect_blocks(voice, frames)

    print(f"\n{'#' * 78}\n#  {os.path.basename(path)}   {frames * FRAME_MS / 1000:.1f}s   ->  {len(blocks)} request(s)\n{'#' * 78}")
    if not blocks:
        print("\n  (no speech detected -- nothing sent)\n")
        return

    for number, (start, end, voice_ms, reason) in enumerate(blocks, 1):
        pcm, kept_ms = shorten_long_silences(audio, voice, start, end)
        raw_ms = (end - start) * FRAME_MS
        print(f"\n[{number}]  {start * FRAME_MS / 1000:.2f}s - {end * FRAME_MS / 1000:.2f}s   sent {kept_ms / 1000:.1f}s of {raw_ms / 1000:.1f}s   ({voice_ms / 1000:.1f}s voice, {reason})\n")
        if api_key is None:
            continue
        print(transcribe(pcm, api_key))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="+", help="one or more files ffmpeg can decode")
    parser.add_argument("--dry-run", action="store_true", help="segment only; do not call the API")
    args = parser.parse_args()

    api_key = None
    if not args.dry_run:
        credentials = Credentials.objects.filter(credential_type=Credentials.CredentialTypes.ELEVENLABS).first()
        if credentials is None or not credentials.get_credentials():
            sys.exit("No usable ElevenLabs credentials; rerun with --dry-run for segmentation only.")
        api_key = credentials.get_credentials()["api_key"]

    for path in args.audio:
        process(path, api_key)


if __name__ == "__main__":
    main()
