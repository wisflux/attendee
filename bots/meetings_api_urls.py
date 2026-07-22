"""Member-facing meeting history.

Split by purpose rather than by session type: ``/local_sessions/*`` controls a recording while
it is happening (start, upload, pause, stop), and ``/meetings/*`` is the read side -- history,
status, transcript and deletion -- covering bot meetings and local recordings alike.
"""

from django.urls import path

from . import meetings_api_views

urlpatterns = [
    path(
        "meetings",
        meetings_api_views.MeetingListView.as_view(),
        name="meeting-list",
    ),
    # GET returns the meeting + status; DELETE removes it (cancelling it if it hasn't run yet).
    path(
        "meetings/<str:object_id>",
        meetings_api_views.MeetingDetailView.as_view(),
        name="meeting-detail",
    ),
    path(
        "meetings/<str:object_id>/transcript",
        meetings_api_views.MeetingTranscriptView.as_view(),
        name="meeting-transcript",
    ),
]
