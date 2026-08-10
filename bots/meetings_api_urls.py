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
    # Must precede "meetings/<object_id>": otherwise "redeem" is captured as an object_id and the
    # POST falls through to the detail view (which has no post handler) as a 405.
    path(
        "meetings/redeem",
        meetings_api_views.MeetingRedeemView.as_view(),
        name="meeting-redeem",
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
    path(
        "meetings/<str:object_id>/share",
        meetings_api_views.MeetingShareView.as_view(),
        name="meeting-share",
    ),
    path(
        "meetings/<str:object_id>/summarize",
        meetings_api_views.MeetingSummarizeView.as_view(),
        name="meeting-summarize",
    ),
]
