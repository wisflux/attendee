"""How well does the silence gate tell speech from silence? Measured on a real recording.

Answers the question the segmentation work keeps running into: the gate decides which audio
reaches the transcriber, but nothing tells us how often it is wrong. This does, using the
transcriber's own word timings as ground truth -- if a word is being spoken between 12.30s and
12.55s and the gate calls that stretch silence, the gate is wrong and it is wrong there.

    python bots/e2e_tests/vad_report.py recording.m4a
    python bots/e2e_tests/vad_report.py recording.wav --no-truth     # offline, no API spend

What to read:

  MISS   speech the gate throws away -- the number that matters. Clipped word endings come
         from here, and they are what the transcriber then has to guess at.
  WASTE  non-speech the gate keeps -- costs money and invites invented text, but is not
         destructive the way MISS is.
  gaps   silences long enough to close an utterance. Without these, nothing ever flushes on
         silence and every utterance runs to the size cap.

Ground truth is itself a transcription, so it is least reliable exactly where the audio is
quietest -- which means MISS is a floor, not a ceiling.
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

import requests  # noqa: E402
import webrtcvad  # noqa: E402

from bots.models import Credentials  # noqa: E402
from bots.utils import pcm_to_mp3  # noqa: E402

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_BYTES = SAMPLE_RATE // 100 * 2
# The gate the local pipeline runs today: anything quieter than this is silence, before any
# speech model is consulted.
RMS_GATE = 0.01
INT16_FULL_SCALE = 32768.0
# A silence must reach the manager's limit to close an utterance; below it nothing flushes.
FLUSH_GAPS_MS = (1000, 1500)
STRIP_SECONDS = 1


def to_pcm(path):
    """Decode anything ffmpeg understands into 16 kHz mono PCM16."""
    out = os.path.join(tempfile.mkdtemp(), "audio.raw")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", out],
        check=True,
        capture_output=True,
    )
    return open(out, "rb").read()


def frame_rms(frame):
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(np.square(samples))) / INT16_FULL_SCALE)


def production_silence(audio, frames):
    """The live gate: RMS threshold first, webrtcvad only for what survives it."""
    vad = webrtcvad.Vad()
    out = []
    for i in range(frames):
        frame = audio[i * FRAME_BYTES : (i + 1) * FRAME_BYTES]
        out.append(frame_rms(frame) < RMS_GATE or not vad.is_speech(frame, SAMPLE_RATE))
    return out


def word_timings(audio):
    """Ground truth: which frames carry a word, per the transcriber's own timestamps."""
    credentials = Credentials.objects.filter(credential_type=Credentials.CredentialTypes.ELEVENLABS).first()
    if credentials is None or not credentials.get_credentials():
        return None
    response = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": credentials.get_credentials()["api_key"]},
        files={"file": ("audio.mp3", pcm_to_mp3(audio, sample_rate=SAMPLE_RATE), "audio/mpeg")},
        data={"model_id": "scribe_v2", "tag_audio_events": False, "timestamps_granularity": "word"},
    )
    response.raise_for_status()
    return [word for word in response.json().get("words", []) if (word.get("text") or "").strip()]


# A transcriber handed near-silence often loops, emitting the same word over and over. Those
# are not speech, and counting them as ground truth blames the gate for missing words nobody
# said. Three identical words in a row is the signature.
LOOP_RUN_LENGTH = 3


def drop_looped_words(words):
    """Remove runs of the same word repeated -- hallucination, not speech."""
    kept, run = [], []
    for word in words + [None]:
        text = (word or {}).get("text", "").strip().lower() if word else None
        if run and text == run[0][1]:
            run.append((word, text))
            continue
        if len(run) < LOOP_RUN_LENGTH:
            kept.extend(item[0] for item in run)
        if word is not None:
            run = [(word, text)]
    dropped = len(words) - len(kept)
    if dropped:
        print(f"ground truth    : dropped {dropped} looped words (transcriber repeating itself on quiet audio)")
    return kept


def truth_mask(words, frames):
    mask = [None] * frames
    for word in words:
        start = int(word["start"] * 1000 / FRAME_MS)
        end = int(word["end"] * 1000 / FRAME_MS)
        for i in range(max(0, start), min(frames, end)):
            mask[i] = word["text"]
    return mask


def silence_runs(is_silent, frames):
    runs, start = [], None
    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, frames))
    return runs


def print_strip(label, keep, frames):
    """One character per second: '#' where the second is mostly voice."""
    per_second = frames // (STRIP_SECONDS * 100)
    row = ""
    for second in range(per_second):
        window = range(second * 100, min(frames, (second + 1) * 100))
        row += "#" if sum(keep(i) for i in window) > 50 else "."
    print(f"{label:<7}{row}")


def report_clipped_words(is_silent, truth, frames):
    stretches, start = [], None
    for i in range(frames):
        clipped = is_silent[i] and truth[i] is not None
        if clipped and start is None:
            start = i
        elif not clipped and start is not None:
            stretches.append((start, i))
            start = None
    if start is not None:
        stretches.append((start, frames))

    total = sum(b - a for a, b in stretches) * FRAME_MS / 1000
    print(f"\nWORDS THE GATE MARKS AS SILENCE  ({len(stretches)} stretches, {total:.1f}s)")
    print(f"{'time':>16} | {'dur':>6} | words being spoken")
    print("-" * 70)
    for a, b in sorted(stretches, key=lambda r: r[0] - r[1])[:15]:
        spoken = []
        for i in range(a, b):
            if truth[i] and (not spoken or spoken[-1] != truth[i]):
                spoken.append(truth[i])
        print(f"{a * FRAME_MS / 1000:>7.2f}-{b * FRAME_MS / 1000:>7.2f}s | {(b - a) * FRAME_MS:>4}ms | {' '.join(spoken)[:40]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="any file ffmpeg can decode")
    parser.add_argument("--no-truth", action="store_true", help="skip the transcription used as ground truth")
    args = parser.parse_args()

    audio = to_pcm(args.audio)
    frames = len(audio) // FRAME_BYTES
    print(f"{args.audio}: {frames * FRAME_MS / 1000:.1f}s, {frames} frames of {FRAME_MS}ms\n")

    is_silent = production_silence(audio, frames)
    runs = silence_runs(is_silent, frames)
    lengths = [(b - a) * FRAME_MS for a, b in runs]
    silent_seconds = sum(is_silent) * FRAME_MS / 1000

    print(f"silence flagged : {silent_seconds:.1f}s ({100 * sum(is_silent) / frames:.0f}%) in {len(runs)} runs")
    print(f"runs under 200ms: {sum(1 for x in lengths if x < 200)}  (these fall inside or between words)")
    for gap in FLUSH_GAPS_MS:
        print(f"gaps >= {gap}ms  : {sum(1 for x in lengths if x >= gap)}  (a gap this long is what closes an utterance)")
    print(f"longest silence : {max(lengths) if lengths else 0}ms")

    if args.no_truth:
        return

    words = word_timings(audio)
    if words is None:
        print("\nNo usable ElevenLabs credentials -- rerun with --no-truth for the offline numbers.")
        return
    words = drop_looped_words(words)
    truth = truth_mask(words, frames)
    speech_frames = sum(1 for t in truth if t is not None)
    missed = sum(1 for i in range(frames) if truth[i] is not None and is_silent[i])
    non_speech = frames - speech_frames
    wasted = sum(1 for i in range(frames) if truth[i] is None and not is_silent[i])

    print(f"\nground truth    : {len(words)} words covering {speech_frames * FRAME_MS / 1000:.1f}s")
    print(f"MISS            : {100 * missed / max(1, speech_frames):.1f}% of speech called silence")
    print(f"WASTE           : {100 * wasted / max(1, non_speech):.1f}% of non-speech kept")

    print("\n1 char = 1 second      '#' = voice      '.' = silence\n")
    print_strip("TRUTH", lambda i: truth[i] is not None, frames)
    print_strip("GATE", lambda i: not is_silent[i], frames)
    report_clipped_words(is_silent, truth, frames)


if __name__ == "__main__":
    main()
