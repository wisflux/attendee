"""Serializer for the member-facing meeting history.

Deliberately NOT BotSerializer: that one exposes `events`, `state`, `transcription_state` and
`recording_state` as method fields, each of which hits the database per row. That is fine for a
single bot, but on a 25-row page it turns one query into dozens (the classic N+1). This carries
only what a history list actually shows, and reads the annotations/prefetches the queryset
already provides, so rendering a page costs a fixed number of queries however long the page is.

Kept in its own module because bots/serializers.py is already ~2400 lines.
"""

from rest_framework import serializers

from . import meeting_status
from .models import SessionTypes

SESSION_TYPE_TO_SOURCE = {
    SessionTypes.BOT: "bot",
    SessionTypes.LOCAL: "local",
    SessionTypes.APP_SESSION: "app",
}


def default_recording_of(bot):
    """The default recording from the prefetched set, or None.

    Iterates the prefetched list rather than filtering, which would issue a fresh query per row
    and reintroduce the N+1 this module exists to avoid.
    """
    for recording in bot.recordings.all():
        if recording.is_default_recording:
            return recording
    return None


class MeetingSerializer(serializers.Serializer):
    id = serializers.CharField(source="object_id")
    source = serializers.SerializerMethodField()
    name = serializers.CharField()
    meeting_url = serializers.CharField()
    status = serializers.SerializerMethodField()
    created_at = serializers.DateTimeField()
    started_at = serializers.SerializerMethodField()
    ended_at = serializers.SerializerMethodField()
    summary = serializers.CharField(allow_null=True)
    summary_state = serializers.CharField()

    def get_source(self, bot):
        return SESSION_TYPE_TO_SOURCE.get(bot.session_type, "bot")

    def get_status(self, bot):
        # Counts come from the queryset annotation when listing; derive_status falls back to
        # counting itself for a single meeting fetched without them.
        return meeting_status.derive_status(
            bot,
            default_recording_of(bot),
            getattr(bot, "pending_utterances", None),
            getattr(bot, "failed_utterances", None),
        )

    def get_started_at(self, bot):
        recording = default_recording_of(bot)
        return recording.started_at if recording else None

    def get_ended_at(self, bot):
        recording = default_recording_of(bot)
        return recording.completed_at if recording else None
