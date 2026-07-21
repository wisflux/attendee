"""Liveness and pause/resume for a local recording that is currently open.

Pause deliberately stops uploading audio, so without an explicit signal the server cannot
tell "paused" from "the app died" -- both look like silence. Rather than guess from timing
(a long thoughtful pause looks exactly like a crash), the desktop says so, and keeps sending
heartbeats meanwhile so the idle reaper leaves the session alone.
"""

import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .authentication import ApiKeyAuthentication
from .local_session_api_views import find_local_session, mark_session_alive
from .models import BotStates, Recording, RecordingManager, RecordingStates
from .team_day_user_auth import decode_user_id
from .throttling import ProjectPostThrottle

logger = logging.getLogger(__name__)


class LocalSessionLivenessView(APIView):
    """Shared owner lookup for the control endpoints below."""

    authentication_classes = [ApiKeyAuthentication]
    throttle_classes = [ProjectPostThrottle]

    def get_open_session(self, request, object_id):
        """(bot, recording, error_response). Only a session that has not ended is open."""
        bot = find_local_session(request, object_id, decode_user_id(request))
        if bot is None:
            return None, None, Response({"error": "Local session not found"}, status=status.HTTP_404_NOT_FOUND)
        if bot.state in BotStates.post_meeting_states():
            return (
                None,
                None,
                Response(
                    {"error": "This session has already ended"},
                    status=status.HTTP_409_CONFLICT,
                ),
            )
        recording = Recording.objects.filter(bot=bot, is_default_recording=True).first()
        return bot, recording, None


class LocalSessionHeartbeatView(LocalSessionLivenessView):
    def post(self, request, object_id):
        bot, _, error = self.get_open_session(request, object_id)
        if error:
            return error

        mark_session_alive(bot.id)
        return Response(status=status.HTTP_200_OK)


class LocalSessionPauseView(LocalSessionLivenessView):
    def post(self, request, object_id):
        bot, recording, error = self.get_open_session(request, object_id)
        if error:
            return error
        if recording is None:
            return Response({"error": "Session has no recording"}, status=status.HTTP_409_CONFLICT)

        # Idempotent: RecordingManager returns early if it is already paused.
        RecordingManager.set_recording_paused(recording)
        # A paused session is still alive -- keep it away from the idle reaper.
        mark_session_alive(bot.id)
        logger.info(f"Paused local session {bot.object_id}")
        return Response({"status": "paused"}, status=status.HTTP_200_OK)


class LocalSessionResumeView(LocalSessionLivenessView):
    def post(self, request, object_id):
        bot, recording, error = self.get_open_session(request, object_id)
        if error:
            return error
        if recording is None:
            return Response({"error": "Session has no recording"}, status=status.HTTP_409_CONFLICT)
        if recording.state not in (RecordingStates.PAUSED, RecordingStates.IN_PROGRESS):
            # e.g. already finalized -- reported rather than raising out of RecordingManager.
            return Response(
                {"error": "Session is not in a resumable state"},
                status=status.HTTP_409_CONFLICT,
            )

        # Idempotent, and preserves started_at when coming back from PAUSED, so the session's
        # timeline (and therefore every chunk's offset_ms) continues from where it stopped.
        RecordingManager.set_recording_in_progress(recording)
        mark_session_alive(bot.id)
        logger.info(f"Resumed local session {bot.object_id}")
        return Response({"status": "recording"}, status=status.HTTP_200_OK)
