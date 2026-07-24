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
from .meeting_viewers import remove_viewer
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
        # Membership is "is my id in the viewer list", not "am I the owner": a meeting shared to
        # me (my id appended on dedup) is as much mine to read as one I created. owner_user_id
        # still records the creator, but it no longer gates visibility.
        Bot.objects.filter(project=request.auth.project, viewer_user_ids__contains=[owner_user_id])
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
    """One meeting I may see, or None. Membership is viewer containment, matching the list view,
    so a meeting shared to me is reachable and a deleted one stays addressable (delete stays
    idempotent). The name is kept for continuity; "owned" here means "visible to me"."""
    return Bot.objects.filter(
        object_id=object_id,
        project=request.auth.project,
        viewer_user_ids__contains=[owner_user_id],
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
        """Reference-counted delete: everyone in the viewer list is equal.

        Deleting removes *my* copy. While anyone else still holds the meeting the recording is
        untouched -- only when the list empties is the data actually wiped. So a shared meeting
        survives until its last viewer leaves, and a meeting I alone hold is wiped the moment I
        delete it. owner_user_id (billing) is never consulted here; leaving is not a privilege.
        """
        owner_user_id = decode_user_id(request)
        bot = find_owned_meeting(request, object_id, owner_user_id)
        if bot is None:
            return Response({"error": "Meeting not found"}, status=status.HTTP_404_NOT_FOUND)

        # Others still hold it: just drop my copy. Never touches the recording or its lifecycle,
        # so leaving a meeting that is still being recorded is fine -- the creator keeps it.
        if any(viewer_id != owner_user_id for viewer_id in bot.viewer_user_ids):
            remove_viewer(bot.id, owner_user_id)
            return Response(status=status.HTTP_204_NO_CONTENT)

        # I am the last viewer, so leaving means the data goes with me. The meeting's lifecycle
        # now decides how.
        # Already wiped: idempotent. A double-click or a stale second device just drops my id.
        if bot.state == BotStates.DATA_DELETED:
            remove_viewer(bot.id, owner_user_id)
            return Response(status=status.HTTP_204_NO_CONTENT)

        # Hasn't run yet: CANCELLED, which removes the row outright (delete_bot), so there is no
        # viewer list left to prune.
        if bot.state == BotStates.SCHEDULED:
            cancelled, error = delete_bot(bot)
            if not cancelled:
                return Response(error, status=status.HTTP_409_CONFLICT)
            logger.info(f"Cancelled scheduled meeting {object_id}")
            return Response(status=status.HTTP_204_NO_CONTENT)

        # Still running and I am the last viewer: wiping would orphan a live recording, and
        # delete_data() only accepts a finished meeting. Refuse, and keep me in the list -- a
        # meeting cannot be left with no viewers while it is still being captured.
        if bot.state not in BotStates.post_meeting_states():
            return Response(
                {"error": "This meeting is still in progress. Stop it before deleting."},
                status=status.HTTP_409_CONFLICT,
            )

        remove_viewer(bot.id, owner_user_id)
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
