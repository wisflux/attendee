"""
notes/models.py — server-side storage for meeting titles, AI summaries, and
live-recording transcripts. This is a NEW app; it only ADDS tables and never
touches Attendee's existing models.

Design + rationale + edge cases live in the desktop repo:
  dev_docs/feature/12-db-schema-final-spec.md
Conventions mirror bots/models.py (object_id + save() generator, created_at/
updated_at, Integer/TextChoices; line-models have no object_id).
Schema reviewed against the live Attendee code before landing.
"""
import secrets
import string

from django.db import models
from django.db.models import Q

# NOTE: we deliberately do NOT use concurrency.IntegerVersionField on Meeting.
# Meeting is written by several disjoint backend actors (webhook, summary job,
# live-session updates) touching DIFFERENT columns; optimistic locking would raise
# false RecordModifiedError with no user to retry, and QuerySet.update() (used for
# the pending->running flip and targeted field writes) does not bump the version
# anyway. Each writer instead does a targeted Meeting.objects.filter(pk=...).update(...)
# on its own columns — atomic and clobber-free. (Schema-review finding.)


def _generate_object_id(prefix: str) -> str:
    """Attendee convention: <prefix> + 16 random alphanumeric chars."""
    random_string = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))
    return f"{prefix}{random_string}"


class AppUser(models.Model):
    """
    Local mirror of a team.day user. It exists ONLY so Meeting can use a real
    foreign key (+ ON DELETE CASCADE). Upserted on every login and on every bot
    webhook, keyed by `team_day_user_id` (the value carried in the signed JWT /
    bot metadata). Attendee's own accounts.User/Organization are unrelated.
    """
    OBJECT_ID_PREFIX = "usr_"

    object_id        = models.CharField(max_length=32, unique=True, editable=False)
    team_day_user_id = models.BigIntegerField(unique=True, db_index=True)             # from the signed JWT
    org_id           = models.BigIntegerField(null=True, blank=True, db_index=True)   # future org-scope
    email            = models.EmailField()
    name             = models.CharField(max_length=255, blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)
    updated_at       = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.object_id:
            self.object_id = _generate_object_id(self.OBJECT_ID_PREFIX)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"AppUser(team_day_user_id={self.team_day_user_id}, email={self.email})"


class Meeting(models.Model):
    """
    One row per meeting the user sees — bot OR live. Holds the LLM title + owner
    + (for bot meetings) a pointer to the Attendee Bot. Bot transcripts are NOT
    copied here; live transcripts live in LiveUtterance; summaries in MeetingSummary.

    Until the LLM/summarizer exists, `title` stays NULL (the UI shows a fallback
    like the meeting_url or "Live recording · <time>") and `summary_state` stays
    PENDING; a later backfill fills them in. Nothing here requires the LLM.
    """
    OBJECT_ID_PREFIX = "mtg_"

    class Source(models.TextChoices):
        BOT  = "bot", "Bot"
        LIVE = "live", "Live"

    class Status(models.IntegerChoices):
        ACTIVE = 1, "Active"
        ENDED  = 2, "Ended"
        FAILED = 3, "Failed"

    class SummaryState(models.IntegerChoices):
        PENDING = 1, "Pending"
        RUNNING = 2, "Running"
        DONE    = 3, "Done"
        FAILED  = 4, "Failed"

    object_id     = models.CharField(max_length=32, unique=True, editable=False)
    owner         = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name="meetings")
    org_id        = models.BigIntegerField(null=True, blank=True)                  # denormalized snapshot (future org-scope)
    project       = models.ForeignKey("bots.Project", on_delete=models.PROTECT,    # bot tenancy; NULL for live
                                      null=True, blank=True, related_name="+")      # live in-person has no Attendee project
    source        = models.CharField(max_length=8, choices=Source.choices)
    title         = models.CharField(max_length=512, null=True, blank=True)        # LLM title (null until generated)
    platform      = models.CharField(max_length=64, null=True, blank=True)         # null for in-person live
    bot           = models.ForeignKey("bots.Bot", null=True, blank=True,           # bot meetings → pointer
                                      on_delete=models.SET_NULL, related_name="+")  # survives Attendee delete_data
    status        = models.IntegerField(choices=Status.choices, default=Status.ACTIVE)
    summary_state = models.IntegerField(choices=SummaryState.choices, default=SummaryState.PENDING)
    language      = models.CharField(max_length=16, null=True, blank=True)
    started_at    = models.DateTimeField(null=True, blank=True)
    ended_at      = models.DateTimeField(null=True, blank=True)                     # null while running
    metadata      = models.JSONField(null=True, blank=True)                         # calendar id/name, attendee emails…
    created_at    = models.DateTimeField(auto_now_add=True)
    updated_at    = models.DateTimeField(auto_now=True)
    deleted_at    = models.DateTimeField(null=True, blank=True)                     # soft delete

    def save(self, *args, **kwargs):
        if not self.object_id:
            self.object_id = _generate_object_id(self.OBJECT_ID_PREFIX)
        super().save(*args, **kwargs)

    class Meta:
        indexes = [
            models.Index(fields=["owner", "-started_at"]),          # fast per-user list
            models.Index(fields=["org_id", "-started_at"]),         # future org-wide list
        ]
        constraints = [
            # One LIVE-or-bot meeting row per (user, bot), ignoring soft-deleted rows. Partial:
            # live meetings (bot IS NULL) are exempt so a user can have many; excluding deleted_at
            # lets a soft-deleted bot meeting be recreated if the webhook re-fires.
            models.UniqueConstraint(
                fields=["owner", "bot"], name="uniq_meeting_owner_bot",
                condition=Q(bot__isnull=False) & Q(deleted_at__isnull=True),
            ),
        ]

    def __str__(self):
        return f"Meeting({self.object_id}, source={self.source})"


class LiveUtterance(models.Model):
    """
    One transcript line for a LIVE recording (bot lines already live in Attendee's
    Utterance). Mirrors Attendee's Utterance shape — a *line* model, so NO object_id.
    `transcription`/`text` are nullable + `failure_data` is set when STT fails, so a
    failed segment is recorded rather than lost.
    """
    class Source(models.IntegerChoices):
        MIC    = 1, "Mic"       # the local user
        SYSTEM = 2, "System"    # other participants (online calls)

    meeting       = models.ForeignKey(Meeting, on_delete=models.CASCADE, related_name="utterances")
    source_uuid   = models.CharField(max_length=255, null=True)                    # idempotency key; UNIQUE PER MEETING (see Meta)
    speaker       = models.CharField(max_length=32)                                # "speaker_0"/"speaker_1"/"you"
    source        = models.IntegerField(choices=Source.choices)
    timestamp_ms  = models.BigIntegerField()                                       # relative to session start (one clock)
    duration_ms   = models.IntegerField()
    text          = models.TextField(null=True, blank=True)                        # plain text (null if failed/pending)
    transcription = models.JSONField(null=True, default=None)                      # {transcript, words[], language}
    failure_data  = models.JSONField(null=True, default=None)                      # set if STT failed
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        # id is the tiebreaker so two chunks sharing a timestamp_ms have a stable order
        # (mic+system share one clock and WILL collide on a ms). Reads ORDER BY the same.
        indexes = [models.Index(fields=["meeting", "timestamp_ms", "id"])]
        constraints = [
            # source_uuid unique PER MEETING (not global) so a client sessionId collision across
            # meetings can't overwrite another meeting's line; NULLs don't collide.
            models.UniqueConstraint(fields=["meeting", "source_uuid"], name="uniq_utterance_meeting_srcuuid"),
        ]

    def __str__(self):
        return f"LiveUtterance(meeting={self.meeting_id}, ts={self.timestamp_ms}ms)"


class MeetingSummary(models.Model):
    """
    The AI summary + action items for a meeting (bot OR live). One row per meeting;
    the row is created only once the summary has been generated. (The *status* of
    summarization lives on Meeting.summary_state — this row holds the content.)
    """
    OBJECT_ID_PREFIX = "sum_"

    object_id    = models.CharField(max_length=32, unique=True, editable=False)
    meeting      = models.OneToOneField(Meeting, on_delete=models.CASCADE, related_name="summary")
    content      = models.JSONField()          # {"summary": "...", "action_items": [...], "key_points": [...]}
    model        = models.CharField(max_length=128)   # which LLM produced it
    generated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.object_id:
            self.object_id = _generate_object_id(self.OBJECT_ID_PREFIX)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"MeetingSummary(meeting={self.meeting_id})"
