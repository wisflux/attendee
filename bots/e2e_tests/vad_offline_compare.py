"""Compare webrtcvad and Silero on the same audio, through the real segmentation path.

Answers the only question that matters before switching detectors: does Silero produce a
better transcript input than what is running today? It does that offline, from audio files --
no meeting, no deployment, no API spend, and the same answer every run.

Both detectors are driven through the real ``LocalAudioInputManager``, so what is compared is
production segmentation behaviour, not raw frame decisions.

    python bots/e2e_tests/vad_offline_compare.py clip.wav [more.wav ...]
    python bots/e2e_tests/vad_offline_compare.py --expect-silent room-tone.wav

The clips worth running, and what each proves:

  clean speech        Silero must not lose words   -> utterances and audio sent stay similar
  speech + music      webrtcvad calls music speech -> Silero should send materially less audio
  quiet speech        the old RMS gate dropped it  -> Silero should send more
  room tone, nobody speaking (--expect-silent)     -> Silero must produce ZERO utterances
  a real session      sanity, in your rooms and voices

Requires webrtcvad and onnxruntime installed together, which requirements.txt keeps for
exactly this reason.
"""

import argparse
import os
import sys
import wave
from datetime import datetime

import django

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendee.settings.development")
django.setup()

import webrtcvad  # noqa: E402

from bots.audio_utils import calculate_normalized_rms  # noqa: E402
from bots.local_audio_processing import BYTES_PER_SAMPLE, LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS, VAD_FRAME_MS, LocalAudioInputManager, duration_ms, feed, flush_remaining  # noqa: E402
from bots.local_vad_params import LocalVadParams  # noqa: E402

TARGET_SAMPLE_RATE = 16000
EPOCH = datetime(2026, 1, 1)
# webrtcvad's default aggressiveness -- what production used before Silero.
WEBRTC_MODE = 0


class WebrtcDetector:
    """webrtcvad behind the interface the manager expects, for comparison only."""

    def __init__(self, mode=WEBRTC_MODE):
        self._vad = webrtcvad.Vad(mode)

    def is_speech(self, speaker_id, chunk_bytes, sample_rate):
        try:
            return self._vad.is_speech(chunk_bytes, sample_rate)
        except Exception:
            # What the old code did: anything the VAD cannot judge counts as speech.
            return True

    def reset(self, speaker_id):
        self._vad = webrtcvad.Vad(WEBRTC_MODE)


def read_pcm(path):
    """16 kHz mono PCM16 from a WAV, or from anything ffmpeg understands via pydub."""
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as source:
            if source.getsampwidth() == BYTES_PER_SAMPLE and source.getnchannels() == 1 and source.getframerate() == TARGET_SAMPLE_RATE:
                return source.readframes(source.getnframes())
    from pydub import AudioSegment

    audio = AudioSegment.from_file(path).set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).set_sample_width(BYTES_PER_SAMPLE)
    return audio.raw_data


def silent_ms(audio, sample_rate, threshold):
    """How much of this clip is below the loudness floor -- the dead air we ship today."""
    step = sample_rate * VAD_FRAME_MS // 1000 * BYTES_PER_SAMPLE
    frames = [audio[i : i + step] for i in range(0, len(audio), step)]
    return sum(VAD_FRAME_MS for frame in frames if len(frame) == step and calculate_normalized_rms(frame) < threshold)


def segment(audio, detector, params, rms_floor):
    """Run one detector over the clip through the real manager, and report what it would send."""
    emitted = []
    manager = LocalAudioInputManager(
        params=params,
        save_audio_chunk_callback=emitted.append,
        get_participant_callback=lambda speaker_id: {"participant_uuid": "u", "participant_full_name": "You"},
        sample_rate=TARGET_SAMPLE_RATE,
        utterance_size_limit=LOCAL_UTTERANCE_SIZE_LIMIT_SECONDS * TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE,
        silence_duration_limit=params.min_silence_seconds,
        should_print_diagnostic_info=False,
    )
    if detector is not None:
        manager.vad = detector

    end_offset_ms = duration_ms(audio, TARGET_SAMPLE_RATE)
    feed(manager, "mic", audio, EPOCH, 0, TARGET_SAMPLE_RATE)
    flush_remaining(manager, "mic", EPOCH, end_offset_ms)

    sent_ms = sum(duration_ms(message["audio_data"], TARGET_SAMPLE_RATE) for message in emitted)
    dead_ms = sum(silent_ms(message["audio_data"], TARGET_SAMPLE_RATE, rms_floor) for message in emitted)
    return {
        "utterances": len(emitted),
        "sent_ms": sent_ms,
        "silent_ms": dead_ms,
        "silent_pct": (100.0 * dead_ms / sent_ms) if sent_ms else 0.0,
        "durations": [duration_ms(m["audio_data"], TARGET_SAMPLE_RATE) for m in emitted],
    }


def compare(path, params, rms_floor, expect_silent):
    audio = read_pcm(path)
    total_ms = duration_ms(audio, TARGET_SAMPLE_RATE)
    results = {
        "webrtcvad": segment(audio, WebrtcDetector(), params, rms_floor),
        "silero": segment(audio, None, params, rms_floor),
    }

    print(f"\n{os.path.basename(path)}  ({total_ms / 1000:.1f}s)")
    print(f"  {'detector':<12}{'utterances':>12}{'audio sent':>14}{'of the clip':>13}{'silence in it':>15}")
    for name, r in results.items():
        share = 100.0 * r["sent_ms"] / total_ms if total_ms else 0.0
        print(f"  {name:<12}{r['utterances']:>12}{r['sent_ms'] / 1000:>12.1f}s{share:>12.0f}%{r['silent_pct']:>14.0f}%")

    web, sil = results["webrtcvad"], results["silero"]
    if expect_silent:
        verdict = "PASS" if sil["utterances"] == 0 else f"FAIL - Silero produced {sil['utterances']} utterance(s) where nobody spoke"
        print(f"  nobody-speaking check: {verdict}   (webrtcvad produced {web['utterances']})")
        return sil["utterances"] == 0

    if web["sent_ms"]:
        delta = 100.0 * (sil["sent_ms"] - web["sent_ms"]) / web["sent_ms"]
        print(f"  Silero sends {delta:+.0f}% audio vs webrtcvad")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", help="audio files to compare")
    parser.add_argument("--expect-silent", action="store_true", help="nobody speaks in these clips; Silero must produce zero utterances")
    parser.add_argument("--rms-floor", type=float, default=0.002, help="loudness below which audio counts as dead air when reporting (default 0.002 = -54 dBFS)")
    args = parser.parse_args()

    params = LocalVadParams.from_env()
    print(f"threshold={params.threshold} hysteresis={params.hysteresis_offset} min_speech={params.min_speech_ms}ms min_silence={params.min_silence_ms}ms trailing_keep={params.trailing_keep_ms}ms")

    ok = True
    for path in args.clips:
        if not os.path.exists(path):
            print(f"\n{path}: not found", file=sys.stderr)
            ok = False
            continue
        ok &= compare(path, params, args.rms_floor, args.expect_silent)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
