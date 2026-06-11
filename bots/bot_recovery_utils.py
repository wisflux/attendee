import logging
from datetime import timedelta

from django.db import IntegrityError, transaction
from django.utils import timezone

from .models import Bot, BotEventManager, BotEventSubTypes, BotEventTypes, BotStates, Recording, TranscriptionTypes

logger = logging.getLogger(__name__)

# Crash sub-types that warrant a replacement: the bot was in the meeting and died unexpectedly.
# Everything else (host removal, could-not-join, normal/auto leaves, out of credits, runtime
# limit) is deliberately excluded -- default-deny, including sub-types added in the future.
# Design + rationale: docs/thin_backend/a4_auto_rejoin_design.md
REPLACEABLE_FATAL_ERROR_SUB_TYPES = [
    BotEventSubTypes.FATAL_ERROR_HEARTBEAT_TIMEOUT,
    BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED,
    BotEventSubTypes.FATAL_ERROR_ATTENDEE_INTERNAL_ERROR,
]
MAX_REPLACEMENTS_PER_MEETING_PER_HOUR = 3


def maybe_create_replacement_bot(bot_id: int, event_sub_type: int) -> Bot | None:
    """Creates and launches a replacement bot for a crashed one when the auto-rejoin policy allows
    it. Called via transaction.on_commit after a FATAL_ERROR transition -- must never raise."""
    try:
        return _maybe_create_replacement_bot(bot_id, event_sub_type)
    except Exception:
        logger.exception(f"Auto-rejoin: replacement check for bot {bot_id} failed")
        return None


def _maybe_create_replacement_bot(bot_id: int, event_sub_type: int) -> Bot | None:
    if event_sub_type not in REPLACEABLE_FATAL_ERROR_SUB_TYPES:
        return None

    dead_bot = Bot.objects.filter(id=bot_id).first()
    if dead_bot is None or not dead_bot.auto_rejoin_on_failure():
        return None

    if not dead_bot.bot_events.filter(event_type=BotEventTypes.BOT_JOINED_MEETING).exists():
        logger.info(f"Auto-rejoin: not replacing bot {dead_bot.object_id} because it never joined the meeting")
        return None

    if dead_bot.project.organization.out_of_credits():
        logger.warning(f"Auto-rejoin: not replacing bot {dead_bot.object_id} because the organization is out of credits")
        return None

    # Imported here to avoid a models -> recovery -> api-utils import cycle at module load time
    from .bots_api_utils import resolve_meeting_lineage, validate_bot_concurrency_limit

    if validate_bot_concurrency_limit(dead_bot.project):
        logger.warning(f"Auto-rejoin: not replacing bot {dead_bot.object_id} because the project is at its concurrent bots limit")
        return None

    # Cap is scoped to this meeting OCCURRENCE via the lineage (a recurring meeting's next
    # occurrence shares the fingerprint but not the replaces-chain, so it is not suppressed).
    one_hour_ago = timezone.now() - timedelta(hours=1)
    recent_replacements = [lineage_bot for lineage_bot in resolve_meeting_lineage(dead_bot) if (lineage_bot.metadata or {}).get("replaces") and lineage_bot.created_at > one_hour_ago]
    if len(recent_replacements) >= MAX_REPLACEMENTS_PER_MEETING_PER_HOUR:
        logger.warning(f"Auto-rejoin: not replacing bot {dead_bot.object_id} -- cap of {MAX_REPLACEMENTS_PER_MEETING_PER_HOUR} replacements/hour reached for this meeting")
        return None

    old_default_recording = Recording.objects.filter(bot=dead_bot, is_default_recording=True).first()
    if old_default_recording is None:
        logger.warning(f"Auto-rejoin: not replacing bot {dead_bot.object_id} because it has no default recording to clone")
        return None

    try:
        with transaction.atomic():
            replacement_bot = Bot.objects.create(
                project=dead_bot.project,
                meeting_url=dead_bot.meeting_url,
                name=dead_bot.name,
                settings=dead_bot.settings,
                metadata={**(dead_bot.metadata or {}), "replaces": dead_bot.object_id},
                meeting_dedup_key=dead_bot.meeting_dedup_key,
                state=BotStates.READY,
            )
            Recording.objects.create(
                bot=replacement_bot,
                recording_type=replacement_bot.recording_type(),
                transcription_type=TranscriptionTypes.NON_REALTIME,
                transcription_provider=old_default_recording.transcription_provider,
                is_default_recording=True,
            )
            BotEventManager.create_event(bot=replacement_bot, event_type=BotEventTypes.JOIN_REQUESTED, event_metadata={"source": "auto_rejoin", "replaces": dead_bot.object_id})
    except IntegrityError as e:
        if "unique_bot_meeting_dedup_key" in str(e):
            logger.info(f"Auto-rejoin: meeting of bot {dead_bot.object_id} is already covered by another active bot, standing down")
            return None
        raise

    from .launch_bot_utils import launch_bot

    launch_bot(replacement_bot)
    logger.info(f"Auto-rejoin: created replacement bot {replacement_bot.object_id} for crashed bot {dead_bot.object_id}")
    return replacement_bot
