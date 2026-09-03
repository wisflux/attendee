"""Celery tasks that segment a local recording's uploaded audio and queue transcription.

The desktop uploads short PCM segments over HTTP; a sentence can straddle two of them, so
audio still mid-utterance when a segment ends is kept as a Redis "tail" (see
``bots.local_session_store``) and glued to the front of the next batch before the VAD (see
``bots.local_audio_processing``) sees it. Ordering, exclusivity and a self-contained
timeline are what keep that correct:

* **Order/dedupe** -- segments drain from a per-source FIFO gated on a monotonic ``sequence``.
* **Exclusivity** -- a per-source lock means exactly one drain owns the tail at a time.
* **Timeline** -- times derive from ``offset_ms`` (ms since the session started), never the
  desktop's wall clock, so clock skew cannot corrupt or drop audio.
"""

import logging

from celery import shared_task

from bots import local_session_store as store
from bots.local_audio_processing import (
    BYTES_PER_SAMPLE,
    build_manager,
    duration_ms,
    feed,
    flush_remaining,
    offset_of_buffered,
)
from bots.models import Bot, BotEventManager, BotEventTypes, BotStates, Participant, Recording, RecordingManager

logger = logging.getLogger(__name__)


@shared_task(bind=True, soft_time_limit=300, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def drain_local_session_audio(self, bot_id, source, is_final=False):
    client = store.redis_client()
    lock_key = store.lock_key(bot_id, source)
    # Exactly one drain may own the tail. A drain that loses the race simply exits: the
    # holder will pick up whatever it queued, and stop re-queues a final drain anyway.
    if not client.set(lock_key, "1", nx=True, ex=store.LOCK_TTL_SECONDS):
        logger.info(f"Local session {bot_id}/{source}: drain already running, skipping")
        return
    try:
        _drain(client, bot_id, source, is_final)
    finally:
        client.delete(lock_key)

    # Anything that arrived while we held the lock still needs draining.
    if client.llen(store.queue_key(bot_id, source)):
        drain_local_session_audio.delay(bot_id, source)


@shared_task(bind=True, soft_time_limit=300, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def finalize_local_session(self, bot_id):
    """Close a stopped session: flush every source's last utterance, then complete once.

    Running as a single task (rather than a final drain per source) is what makes marking
    the recording complete safe -- every source's last utterance already exists by the time
    we get there, so completing the recording can't race an as-yet-uncreated utterance.
    """
    client = store.redis_client()
    for source in store.LOCAL_SESSION_SOURCES:
        lock_key = store.lock_key(bot_id, source)
        if not client.set(lock_key, "1", nx=True, ex=store.LOCK_TTL_SECONDS):
            # A normal drain is still mid-flight for this source; come back once it releases.
            raise self.retry(countdown=1)
        try:
            _drain(client, bot_id, source, is_final=True)
        finally:
            client.delete(lock_key)

    bot = Bot.objects.get(id=bot_id)
    recording = Recording.objects.filter(bot=bot, is_default_recording=True).first()
    if recording is not None:
        RecordingManager.set_recording_complete(recording)

    # End the session -- but only from READY, and only after the recording is COMPLETE.
    # The state guard makes a retried/duplicate stop a no-op instead of an invalid-transition
    # error; completing the recording first stops the post-meeting transition from marking a
    # local (file-less) recording FAILED.
    bot.refresh_from_db()
    if bot.state == BotStates.READY:
        BotEventManager.create_event(bot=bot, event_type=BotEventTypes.LOCAL_SESSION_ENDED)


def _drain(client, bot_id, source, is_final):
    try:
        bot = Bot.objects.get(id=bot_id)
    except Bot.DoesNotExist:
        # Session was deleted while audio was still queued -- drop it quietly, don't retry.
        store.clear_session_state(bot_id)
        return
    recording = Recording.objects.filter(bot=bot, is_default_recording=True).first()
    participant = Participant.objects.filter(bot=bot, uuid=source).first()
    if recording is None or participant is None:
        # Data was deleted (delete_data removes the recording contents + participants) -- the
        # session is gone; clear any leftover Redis state and stop.
        logger.info(f"Local session {bot_id}/{source}: recording/participant gone, dropping audio")
        store.clear_session_state(bot_id)
        return

    tail = store.load_tail(client, bot_id, source)
    segments = store.pop_segments(
        client,
        bot_id,
        source,
        tail["last_sequence"],
        on_replay=lambda seq: logger.info(f"Local session {bot_id}/{source}: dropping replayed segment {seq}"),
    )
    if not segments and not (is_final and tail["audio"]):
        return

    audio = bytearray(tail["audio"])
    start_offset_ms = tail["started_offset_ms"]
    end_offset_ms = tail["end_offset_ms"]
    sample_rate = segments[0]["sample_rate"] if segments else tail["sample_rate"]
    last_sequence = tail["last_sequence"]

    for segment in segments:
        # A gap between what we hold and where this segment starts is real elapsed silence.
        # Feeding it as zeros lets the VAD see the pause instead of splicing speech together.
        if end_offset_ms is not None and segment["offset_ms"] > end_offset_ms:
            gap_ms = segment["offset_ms"] - end_offset_ms
            audio.extend(b"\x00" * int(gap_ms * sample_rate / 1000) * BYTES_PER_SAMPLE)
        if start_offset_ms is None:
            start_offset_ms = segment["offset_ms"]
        audio.extend(segment["audio"])
        end_offset_ms = segment["offset_ms"] + duration_ms(segment["audio"], sample_rate)
        last_sequence = segment["sequence"]

    epoch = bot.created_at.replace(tzinfo=None)
    manager = build_manager(recording, participant, sample_rate, vad_state=tail["vad_state"])
    consumed = feed(manager, source, bytes(audio), epoch, start_offset_ms or 0, sample_rate)

    if is_final:
        # Emit the last unfinished utterance for THIS source only. The recording is marked
        # complete by finalize_local_session, after EVERY source has flushed -- doing it here
        # would let one source close the recording before the other's last utterance exists.
        flush_remaining(manager, source, epoch, end_offset_ms or 0)
        store.save_tail(client, bot_id, source, b"", None, None, last_sequence, sample_rate, manager.export_vad_state())
        return

    remaining = manager.utterances.get(source, b"")
    unconsumed = bytes(audio[consumed:])  # a partial frame we could not hand to the VAD yet
    store.save_tail(
        client,
        bot_id,
        source,
        bytes(remaining) + unconsumed,
        offset_of_buffered(manager, source, epoch),
        end_offset_ms,
        last_sequence,
        sample_rate,
        manager.export_vad_state(),
    )
