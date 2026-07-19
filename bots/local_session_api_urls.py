from django.urls import path

from . import local_session_api_views

urlpatterns = [
    path(
        "local_sessions",
        local_session_api_views.LocalSessionCreateView.as_view(),
        name="local-session-create",
    ),
    path(
        "local_sessions/<str:object_id>/audio",
        local_session_api_views.LocalSessionAudioView.as_view(),
        name="local-session-audio",
    ),
    path(
        "local_sessions/<str:object_id>/heartbeat",
        local_session_api_views.LocalSessionHeartbeatView.as_view(),
        name="local-session-heartbeat",
    ),
    path(
        "local_sessions/<str:object_id>/stop",
        local_session_api_views.LocalSessionStopView.as_view(),
        name="local-session-stop",
    ),
    # Reuses the bot transcript logic but adds the owner check (see LocalSessionTranscriptView).
    path(
        "local_sessions/<str:object_id>/transcript",
        local_session_api_views.LocalSessionTranscriptView.as_view(),
        name="local-session-transcript",
    ),
]
