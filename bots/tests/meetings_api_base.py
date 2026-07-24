"""Shared setup for the ``/meetings`` API tests.

Not a test module itself (no ``test_`` prefix, so the runner does not collect it): it only
holds the fixtures both meetings test files need -- a project with an API key, two distinct
members, and a factory for meetings owned by either of them.

Two members matter more than they look. Almost every rule this endpoint has is "member A
must not see member B", and a suite with a single member cannot fail that way.
"""

import jwt
from django.core.cache import cache
from django.test import Client, TestCase

from accounts.models import Organization
from bots.models import (
    ApiKey,
    Bot,
    BotStates,
    Project,
    Recording,
    RecordingStates,
    RecordingTranscriptionStates,
    RecordingTypes,
    SessionTypes,
    TranscriptionProviders,
    TranscriptionTypes,
)

JWT_SECRET = "test-team-day-secret"
MEMBER_A = "1001"
MEMBER_B = "1002"

MEETINGS_URL = "/api/v1/meetings"


def member_token(user_id=MEMBER_A, secret=JWT_SECRET, **payload_overrides):
    """A JWT shaped exactly like team.day's: HS256, ``{id, email}``."""
    payload = {"id": int(user_id), "email": f"member{user_id}@example.com"}
    payload.update(payload_overrides)
    return jwt.encode(payload, secret, algorithm="HS256")


class MeetingsApiTestBase(TestCase):
    def setUp(self):
        # MemberReadThrottle counts per member in the cache, which outlives a single test.
        # Without this, a long test class throttles itself and later tests fail as 429s.
        cache.clear()
        self.organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=self.organization)
        self.api_key, self.api_key_plain = ApiKey.create(project=self.project, name="Test API Key")
        self.client = Client()

    def make_meeting(
        self,
        owner_user_id=MEMBER_A,
        *,
        session_type=SessionTypes.BOT,
        state=BotStates.ENDED,
        name="Standup",
        meeting_url="https://meet.google.com/abc-defg-hij",
        created_at=None,
        project=None,
        with_recording=True,
        viewer_user_ids=None,
    ):
        # A real meeting's owner is always its first viewer (dispatch appends the creator; the
        # 0093 backfill seeds existing ones). Mirror that here so ``owner_user_id=X`` produces a
        # meeting X can actually see, without every test having to spell out the list. Pass
        # viewer_user_ids explicitly to model a shared meeting or a hand-built edge case.
        if viewer_user_ids is None:
            viewer_user_ids = [owner_user_id] if owner_user_id else []
        bot = Bot.objects.create(
            project=project or self.project,
            meeting_url=meeting_url,
            name=name,
            state=state,
            session_type=session_type,
            owner_user_id=owner_user_id,
            viewer_user_ids=viewer_user_ids,
        )
        if created_at is not None:
            # created_at is auto_now_add, so it can only be set by writing past the model.
            Bot.objects.filter(id=bot.id).update(created_at=created_at)
            bot.refresh_from_db()
        if with_recording:
            Recording.objects.create(
                bot=bot,
                recording_type=RecordingTypes.AUDIO_AND_VIDEO,
                transcription_type=TranscriptionTypes.NON_REALTIME,
                transcription_provider=TranscriptionProviders.ELEVENLABS,
                is_default_recording=True,
                state=RecordingStates.COMPLETE,
                transcription_state=RecordingTranscriptionStates.COMPLETE,
            )
        return bot

    def get_meetings(self, token=None, url=MEETINGS_URL, **query):
        """GET the history as a member. ``token=None`` means "send no X-User-Token"."""
        headers = {"HTTP_AUTHORIZATION": f"Token {self.api_key_plain}"}
        if token is not None:
            headers["HTTP_X_USER_TOKEN"] = token
        return self.client.get(url, query, **headers)

    def ids_in(self, response):
        return [row["id"] for row in response.json()["results"]]
