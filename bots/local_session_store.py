"""Redis-backed state for local recording sessions.

A local session has no long-lived process (unlike a meeting bot), so the audio "memory"
that a bot keeps in RAM is rebuilt here in Redis, per session and per source:

* ``queue`` -- a FIFO list of uploaded, not-yet-processed segments.
* ``tail``  -- the unfinished utterance left over after the last drain, glued to the front
  of the next batch so the VAD only ever cuts on real silence.
* ``lock``  -- ensures exactly one drain owns a source's tail at a time.
"""

import base64
import json

import redis
from django.conf import settings

# A local recording always has exactly these two audio sources (the desktop tags every
# chunk with one). Kept here so the views, tasks and finalizer share one definition.
MIC_SOURCE = "mic"
SYSTEM_SOURCE = "system"
LOCAL_SESSION_SOURCES = (MIC_SOURCE, SYSTEM_SOURCE)

TAIL_TTL_SECONDS = 3600
LOCK_TTL_SECONDS = 120
MAX_SEGMENTS_PER_DRAIN = 200


def redis_client():
    return redis.from_url(settings.REDIS_URL_WITH_PARAMS)


def queue_key(bot_id, source):
    return f"local_session_queue:{bot_id}:{source}"


def tail_key(bot_id, source):
    return f"local_session_tail:{bot_id}:{source}"


def lock_key(bot_id, source):
    return f"local_session_lock:{bot_id}:{source}"


def enqueue_segment(bot_id, source, sequence, audio, sample_rate, offset_ms):
    """Append an uploaded segment to its source's FIFO queue, ready for the next drain."""
    payload = {
        "sequence": sequence,
        "audio": base64.b64encode(audio).decode(),
        "sample_rate": sample_rate,
        "offset_ms": offset_ms,
    }
    redis_client().rpush(queue_key(bot_id, source), json.dumps(payload))


def load_tail(client, bot_id, source):
    """Unfinished-utterance audio plus the offset it began at and the last sequence seen."""
    raw = client.get(tail_key(bot_id, source))
    if not raw:
        return {
            "audio": b"",
            "started_offset_ms": None,
            "end_offset_ms": None,
            "last_sequence": -1,
            "sample_rate": None,
        }
    payload = json.loads(raw)
    payload["audio"] = base64.b64decode(payload["audio"])
    return payload


def save_tail(client, bot_id, source, audio, started_offset_ms, end_offset_ms, last_sequence, sample_rate):
    payload = {
        "audio": base64.b64encode(bytes(audio)).decode(),
        "started_offset_ms": started_offset_ms,
        "end_offset_ms": end_offset_ms,
        # Kept even when the audio is empty, so a replayed upload is still recognised.
        "last_sequence": last_sequence,
        # Carried so a final flush can rebuild the VAD without the caller supplying it.
        "sample_rate": sample_rate,
    }
    client.set(tail_key(bot_id, source), json.dumps(payload), ex=TAIL_TTL_SECONDS)


def pop_segments(client, bot_id, source, last_sequence, on_replay=None):
    """FIFO drain, dropping replays.

    Two kinds of replay have to die here: one already folded into the tail (sequence at or
    below the tail's), and one duplicated *within this same batch* -- a retrying uploader can
    land both copies before any drain runs, and checking only against the tail would let the
    second copy through and splice the same audio in twice.
    """
    segments = []
    seen_sequences = set()
    for _ in range(MAX_SEGMENTS_PER_DRAIN):
        raw = client.lpop(queue_key(bot_id, source))
        if raw is None:
            break
        segment = json.loads(raw)
        sequence = segment["sequence"]
        if sequence <= last_sequence or sequence in seen_sequences:
            if on_replay:
                on_replay(sequence)
            continue
        seen_sequences.add(sequence)
        segment["audio"] = base64.b64decode(segment["audio"])
        segments.append(segment)
    segments.sort(key=lambda s: s["sequence"])
    return segments
