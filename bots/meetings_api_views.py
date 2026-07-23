"""Per-member meeting history: list, status, transcript and delete.

Scoped to the member named by the verified ``X-User-Token``, covering BOTH meeting bots and
local recordings, so the desktop has one history regardless of how a meeting was captured.

Two rules run through everything here:

* **Ownership is a filter, never a check afterwards.** A meeting that isn't yours is
  indistinguishable from one that doesn't exist, so a 404 never confirms it is real.
* **Never build the query from an empty member id.** ``filter(owner_user_id=None)`` compiles to
  ``IS NULL``, which would return every unowned meeting in the project -- so the id is resolved
  (and the request rejected) before any queryset is touched.
"""

import logging

from django.db.models import Count, Q
from rest_framework import status
from rest_framework.pagination import CursorPagination
from rest_framework.response import Response
from rest_framework.views import APIView

from .authentication import ApiKeyAuthentication
from .bots_api_utils import delete_bot
from .bots_api_views import TranscriptView
from .local_session_store import clear_session_state
from .meetings_filters import FilterError, apply_meeting_filters
from .meetings_serializers import MeetingSerializer
from .models import Bot, BotStates, SessionTypes
from .team_day_user_auth import decode_user_id
from .throttling import MemberReadThrottle

logger = logging.getLogger(__name__)


class MeetingCursorPagination(CursorPagination):
    """Newest first, with the row id as a tiebreak.

    The project's existing BotCursorPagination orders by ``created_at`` ascending, which would
    open a member's history on their oldest meeting. The ``-id`` matters as much as the order:
    without a unique tiebreak, meetings sharing a created_at can be skipped or repeated across
    page boundaries.
    """

    ordering = ("-created_at", "-id")
    page_size = 25


def owned_meetings(request, owner_user_id):
    """Base queryset: this member's meetings in this project, ready for listing.

    Annotated with the transcription counts the status needs, so rendering a page costs a fixed
    number of queries instead of a few per row.
    """
    pending = Q(recordings__utterances__transcription__isnull=True, recordings__utterances__failure_data__isnull=True)
    failed = Q(recordings__utterances__failure_data__isnull=False)
    return (
        Bot.objects.filter(project=request.auth.project, owner_user_id=owner_user_id)
        # Deleted meetings keep their row (delete_data wipes contents but marks the row), so
        # they must be excluded or they resurface as blank entries.
        .exclude(state=BotStates.DATA_DELETED)
        # Zoom RTMS app sessions are neither a bot nor a local recording.
        .exclude(session_type=SessionTypes.APP_SESSION)
        .prefetch_related("recordings")
        .annotate(
            pending_utterances=Count("recordings__utterances", filter=pending, distinct=True),
            failed_utterances=Count("recordings__utterances", filter=failed, distinct=True),
        )
    )


def find_owned_meeting(request, object_id, owner_user_id):
    """One meeting, or None. Deleted meetings are still addressable so delete stays idempotent."""
    return Bot.objects.filter(
        object_id=object_id,
        project=request.auth.project,
        owner_user_id=owner_user_id,
    ).first()


class MeetingListView(APIView):
    authentication_classes = [ApiKeyAuthentication]
    throttle_classes = [MemberReadThrottle]

    def get(self, request):
        owner_user_id = decode_user_id(request)
        queryset = owned_meetings(request, owner_user_id)

        # Filters narrow a queryset that is already scoped to this member, so none of them can
        # widen it to somebody else's meetings -- see meetings_filters for why that ordering is
        # the safety property rather than a style choice.
        try:
            queryset = apply_meeting_filters(queryset, request.query_params)
        except FilterError as bad_parameter:
            return Response({"error": str(bad_parameter)}, status=status.HTTP_400_BAD_REQUEST)

        paginator = MeetingCursorPagination()
        page = paginator.paginate_queryset(queryset, request, view=self)
        return paginator.get_paginated_response(MeetingSerializer(page, many=True).data)


class MeetingDetailView(APIView):
    authentication_classes = [ApiKeyAuthentication]
    throttle_classes = [MemberReadThrottle]

    def get(self, request, object_id):
        owner_user_id = decode_user_id(request)
        bot = owned_meetings(request, owner_user_id).filter(object_id=object_id).first()
        if bot is None:
            return Response({"error": "Meeting not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(MeetingSerializer(bot).data, status=status.HTTP_200_OK)

    def delete(self, request, object_id):
        owner_user_id = decode_user_id(request)
        bot = find_owned_meeting(request, object_id, owner_user_id)
        if bot is None:
            return Response({"error": "Meeting not found"}, status=status.HTTP_404_NOT_FOUND)

        # Already deleted: say so calmly. A double-click, a retry after a lost response, or a
        # second device acting on a stale list would otherwise raise out of delete_data().
        if bot.state == BotStates.DATA_DELETED:
            return Response(status=status.HTTP_204_NO_CONTENT)

        # A meeting that hasn't run yet is CANCELLED, not deleted -- a different operation that
        # removes the row outright rather than wiping the contents of something that happened.
        if bot.state == BotStates.SCHEDULED:
            cancelled, error = delete_bot(bot)
            if not cancelled:
                return Response(error, status=status.HTTP_409_CONFLICT)
            logger.info(f"Cancelled scheduled meeting {object_id}")
            return Response(status=status.HTTP_204_NO_CONTENT)

        # Anything still running cannot be deleted: delete_data() only accepts a finished
        # meeting, and letting it raise would surface as a 500 instead of something actionable.
        if bot.state not in BotStates.post_meeting_states():
            return Response(
                {"error": "This meeting is still in progress. Stop it before deleting."},
                status=status.HTTP_409_CONFLICT,
            )

        bot.delete_data()
        if bot.session_type == SessionTypes.LOCAL:
            # Drop any queued audio/tail/lock so a deleted session leaves nothing behind.
            clear_session_state(bot.id)
        logger.info(f"Deleted meeting data for {object_id}")
        return Response(status=status.HTTP_204_NO_CONTENT)


class MeetingTranscriptView(TranscriptView):
    """Owner-gated transcript for bot and local meetings alike.

    Reuses the bot transcript logic unchanged and only adds the ownership gate in front: the
    inherited view scopes by project, and one project API key is shared by every desktop
    install, so on its own it would let any member read any member's transcript.
    """

    throttle_classes = [MemberReadThrottle]

    def get(self, request, object_id):
        owner_user_id = decode_user_id(request)
        if find_owned_meeting(request, object_id, owner_user_id) is None:
            return Response({"error": "Meeting not found"}, status=status.HTTP_404_NOT_FOUND)
        return super().get(request, object_id)
