"""Background job that generates + saves a meeting's AI title and summary (see docs/meeting-summaries).

Triggered when a default recording's transcript completes (bot + local). Robust by construction:
an atomic PENDING->GENERATING claim (stamped with summary_started_at) makes concurrent fires safe,
every terminal write is a single-column ``.update()`` guarded by that exact claim (so a delete or a
reaper that intervened voids a stale worker's late write), and any failure lands a terminal state
with a safe reason.
"""

import logging
import time
from datetime import timedelta

import requests
from celery import shared_task
from django.utils import timezone

from bots.models import Bot, Credentials, Recording, SummaryStates
from bots.summarization.azure_client import SUMMARY_MAX_TOKENS, SummaryTruncated, build_messages, parse_response, summarize
from bots.summarization.generation import classify_failure, concise_retry_messages, is_transient
from bots.summarization.prompts import format_transcript, is_long_enough
from bots.summarization.transcript import collect_utterances

logger = logging.getLogger(__name__)

# Transient Azure errors are retried within one run; config errors are not.
AZURE_MAX_ATTEMPTS = 3
AZURE_RETRY_BACKOFF_SECS = 5
# A larger budget for the concise retry so a truncation driven by reasoning tokens has room to fit
# both the reasoning and the (now shorter) notes.
CONCISE_MAX_TOKENS = 24000
# A meeting left GENERATING longer than this is assumed dead (crashed worker) and reset to PENDING.
STALE_GENERATING_MINUTES = 30


def _finish(bot_id, claim_time, **fields):
    """Terminal write guarded by the claim: only the worker still owning this exact GENERATING claim
    writes, so a delete or reaper that intervened silently voids a late/stale write. Returns rowcount."""
    return Bot.objects.filter(id=bot_id, summary_state=SummaryStates.GENERATING, summary_started_at=claim_time).update(summary_started_at=None, **fields)


def _fail(bot_id, claim_time, reason):
    _finish(bot_id, claim_time, summary_state=SummaryStates.FAILED, summary_error=reason)


def _build_context(bot, recording, participant_names):
    """Light context hints for the prompt -- all optional."""
    context = {}
    event = bot.calendar_event
    if event and event.name:
        context["title"] = event.name
    if recording and recording.started_at:
        context["date"] = recording.started_at.date().isoformat()
        if recording.completed_at:
            minutes = int((recording.completed_at - recording.started_at).total_seconds() // 60)
            if minutes > 0:
                context["duration"] = f"{minutes} min"
    if participant_names:
        context["participants"] = participant_names
    return context


def _call_azure(credentials, messages):
    """Summarize, retrying transient errors (bounded) and doing ONE concise retry with a larger
    budget if the reply is truncated. The concise attempt is itself transient-retried."""
    current, tokens, concise_tried, transient_attempts = messages, SUMMARY_MAX_TOKENS, False, 0
    while True:
        try:
            return summarize(credentials, current, max_output_tokens=tokens)
        except SummaryTruncated:
            if concise_tried:
                raise
            concise_tried, current, tokens = True, concise_retry_messages(messages), CONCISE_MAX_TOKENS
        except requests.RequestException as exc:
            transient_attempts += 1
            if is_transient(exc) and transient_attempts < AZURE_MAX_ATTEMPTS:
                time.sleep(AZURE_RETRY_BACKOFF_SECS * transient_attempts)
                continue
            raise


@shared_task(soft_time_limit=1200)
def generate_meeting_summary(bot_id):
    # Atomic claim: only the worker that flips PENDING -> GENERATING proceeds; the rest no-op.
    # summary_started_at stamps the claim (used as the lease and the terminal-write guard).
    claim_time = timezone.now()
    if not Bot.objects.filter(id=bot_id, summary_state=SummaryStates.PENDING).update(summary_state=SummaryStates.GENERATING, summary_started_at=claim_time):
        return
    try:
        bot = Bot.objects.select_related("project", "calendar_event").get(id=bot_id)
        recording = Recording.objects.filter(bot=bot, is_default_recording=True).first()
        utterances = collect_utterances(recording)
        if not is_long_enough(format_transcript(utterances)):
            _finish(bot_id, claim_time, summary_state=SummaryStates.SKIPPED)
            return
        credential_row = bot.project.credentials.filter(credential_type=Credentials.CredentialTypes.AZURE_OPENAI).first()
        if credential_row is None:
            _fail(bot_id, claim_time, "No Azure OpenAI credential configured for this project.")
            return

        participants = sorted({u["speaker_name"] for u in utterances if u["speaker_name"]})
        messages = build_messages(utterances, _build_context(bot, recording, participants))
        parsed = parse_response(_call_azure(credential_row.get_credentials(), messages))
        if not parsed["notes"]:
            _fail(bot_id, claim_time, "The model returned an empty summary.")
            return

        fields = {"summary": parsed["notes"], "summary_generated_at": timezone.now(), "summary_state": SummaryStates.DONE, "summary_error": None}
        if parsed["title"]:
            fields["name"] = parsed["title"]  # success-only title overwrite (never blanks a good name)
        _finish(bot_id, claim_time, **fields)
    except SummaryTruncated:
        _fail(bot_id, claim_time, "The meeting was too long to summarize in one pass.")
    except requests.RequestException as exc:
        _fail(bot_id, claim_time, classify_failure(exc))
    except Exception:
        logger.exception("Unexpected error generating summary for bot %s", bot_id)
        _fail(bot_id, claim_time, "Summary generation failed.")


@shared_task
def reap_stale_summary_generations():
    """Reset meetings stuck in GENERATING (a crashed worker) back to PENDING so they can retry.

    Uses summary_started_at (a summary-only lease), not the shared updated_at, so an unrelated
    bot.save() during generation can't hide a dead job. Needs a celery-beat schedule to run
    periodically; the Regenerate endpoint applies the same staleness check for on-demand recovery.
    """
    cutoff = timezone.now() - timedelta(minutes=STALE_GENERATING_MINUTES)
    reset = Bot.objects.filter(summary_state=SummaryStates.GENERATING, summary_started_at__lt=cutoff).update(summary_state=SummaryStates.PENDING, summary_started_at=None)
    if reset:
        logger.warning("Reset %s stale summary generation(s) to PENDING", reset)
