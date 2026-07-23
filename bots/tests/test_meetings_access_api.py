"""Characterisation tests for who may read and delete a meeting.

One project API key is shared by every desktop install, so the API key alone authorises
nothing personal -- the ``X-User-Token`` decides whose data a request may touch. These tests
pin both halves of that: the token must be verified (and fail closed when it cannot be), and
a meeting that isn't yours must be indistinguishable from one that does not exist.

The 404-not-403 rule is the subtle one. Answering 403 would confirm the meeting is real and
belongs to *someone*, which leaks the existence of other members' recordings.
"""

from datetime import timedelta

import jwt
from django.test import override_settings
from django.utils import timezone

from bots.models import Bot, BotStates

from .meetings_api_base import (
    JWT_SECRET,
    MEETINGS_URL,
    MEMBER_A,
    MEMBER_B,
    MeetingsApiTestBase,
    member_token,
)


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestMeetingListAuthentication(MeetingsApiTestBase):
    def test_no_user_token_is_rejected(self):
        response = self.get_meetings(token=None)

        self.assertEqual(response.status_code, 401)

    def test_a_token_signed_with_the_wrong_secret_is_rejected(self):
        response = self.get_meetings(member_token(secret="not-the-servers-secret"))

        self.assertEqual(response.status_code, 401)

    def test_an_expired_token_is_rejected(self):
        expired = member_token(exp=timezone.now() - timedelta(hours=1))

        response = self.get_meetings(expired)

        self.assertEqual(response.status_code, 401)

    def test_a_token_carrying_no_user_id_is_rejected(self):
        # Correctly signed, but says nothing about who the caller is. Accepting it would mean
        # querying with an empty owner id -- the IS NULL trap.
        anonymous = jwt.encode({"email": "nobody@example.com"}, JWT_SECRET, algorithm="HS256")

        response = self.get_meetings(anonymous)

        self.assertEqual(response.status_code, 401)

    @override_settings(TEAM_DAY_JWT_SECRET=None)
    def test_an_unconfigured_server_refuses_rather_than_trusting_the_caller(self):
        # No secret means no token can be verified. Failing closed (503) is the only safe
        # answer; treating "cannot verify" as "no owner" would expose every unowned meeting.
        response = self.get_meetings(member_token())

        self.assertEqual(response.status_code, 503)


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestMeetingDetailAccess(MeetingsApiTestBase):
    def detail(self, object_id, member=MEMBER_A):
        return self.get_meetings(member_token(member), url=f"{MEETINGS_URL}/{object_id}")

    def transcript(self, object_id, member=MEMBER_A):
        return self.get_meetings(member_token(member), url=f"{MEETINGS_URL}/{object_id}/transcript")

    def test_my_own_meeting_is_readable(self):
        meeting = self.make_meeting(MEMBER_A)

        response = self.detail(meeting.object_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], meeting.object_id)

    def test_another_members_meeting_looks_like_it_does_not_exist(self):
        meeting = self.make_meeting(MEMBER_B)

        response = self.detail(meeting.object_id, member=MEMBER_A)

        self.assertEqual(response.status_code, 404)

    def test_an_unowned_meeting_is_not_readable_either(self):
        meeting = self.make_meeting(owner_user_id=None)

        self.assertEqual(self.detail(meeting.object_id).status_code, 404)

    def test_another_members_transcript_is_not_readable(self):
        meeting = self.make_meeting(MEMBER_B)

        response = self.transcript(meeting.object_id, member=MEMBER_A)

        self.assertEqual(response.status_code, 404)

    def test_a_deleted_meeting_is_no_longer_readable(self):
        # delete_data() wipes the contents but keeps the row, so the list and detail views
        # exclude it explicitly -- otherwise it returns as a blank entry.
        meeting = self.make_meeting(MEMBER_A, state=BotStates.DATA_DELETED)

        self.assertEqual(self.detail(meeting.object_id).status_code, 404)


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestMeetingDelete(MeetingsApiTestBase):
    def delete(self, object_id, member=MEMBER_A):
        return self.client.delete(
            f"{MEETINGS_URL}/{object_id}",
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(member),
        )

    def test_another_members_meeting_cannot_be_deleted(self):
        meeting = self.make_meeting(MEMBER_B)

        response = self.delete(meeting.object_id, member=MEMBER_A)

        self.assertEqual(response.status_code, 404)
        meeting.refresh_from_db()
        self.assertEqual(meeting.state, BotStates.ENDED, "someone else's meeting was modified")

    def test_a_meeting_still_in_progress_is_refused(self):
        meeting = self.make_meeting(MEMBER_A, state=BotStates.JOINING)

        response = self.delete(meeting.object_id)

        self.assertEqual(response.status_code, 409)
        self.assertIn("still in progress", response.json()["error"])

    def test_deleting_twice_is_not_an_error(self):
        # A double click, a retry after a lost response, or a second device acting on a stale
        # list must not raise out of delete_data().
        meeting = self.make_meeting(MEMBER_A, state=BotStates.DATA_DELETED)

        self.assertEqual(self.delete(meeting.object_id).status_code, 204)

    def test_a_finished_meeting_is_deleted(self):
        meeting = self.make_meeting(MEMBER_A, state=BotStates.ENDED)

        response = self.delete(meeting.object_id)

        self.assertEqual(response.status_code, 204)
        self.assertEqual(Bot.objects.get(id=meeting.id).state, BotStates.DATA_DELETED)
