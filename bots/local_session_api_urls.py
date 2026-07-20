from django.urls import path

from . import bots_api_views, local_session_api_views

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
        "local_sessions/<str:object_id>/stop",
        local_session_api_views.LocalSessionStopView.as_view(),
        name="local-session-stop",
    ),
    # The bot transcript view keys off object_id + project and ignores session_type,
    # so local sessions reuse it unchanged.
    path(
        "local_sessions/<str:object_id>/transcript",
        bots_api_views.TranscriptView.as_view(),
        name="local-session-transcript",
    ),
]
