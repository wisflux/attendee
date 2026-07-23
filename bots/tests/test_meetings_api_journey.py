"""The whole journey, end to end: sign in, open a page, read a transcript.

The other two meetings test files check one rule each. This one answers the question a person
actually asks -- *when I sign in, do I get MY words, on the RIGHT page?* -- by setting up two
members who each have a bot meeting AND a local recording, with different words spoken in
every one of the four. A filter that is subtly wrong shows up here as somebody else's
sentence, which no single-rule test can catch.

The last test is the cross-device claim itself: the same token replayed from a brand-new
client returns the same history, because nothing in the answer depends on the machine asking.
"""

import uuid

from django.test import Client, override_settings

from bots.models import Participant, Recording, SessionTypes, Utterance

from .meetings_api_base import (
    JWT_SECRET,
    MEETINGS_URL,
    MEMBER_A,
    MEMBER_B,
    MeetingsApiTestBase,
    member_token,
)


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestMeetingHistoryJourney(MeetingsApiTestBase):
    def setUp(self):
        super().setUp()
        # Four meetings: each member has one of each kind, and every one holds words that
        # appear nowhere else, so a leak is identifiable rather than merely a count mismatch.
        self.a_bot = self.make_meeting(MEMBER_A, session_type=SessionTypes.BOT, name="A standup")
        self.a_local = self.make_meeting(MEMBER_A, session_type=SessionTypes.LOCAL, name="A voice note")
        self.b_bot = self.make_meeting(MEMBER_B, session_type=SessionTypes.BOT, name="B standup")
        self.b_local = self.make_meeting(MEMBER_B, session_type=SessionTypes.LOCAL, name="B voice note")

        self.say(self.a_bot, "alpha in the bot meeting")
        self.say(self.a_local, "alpha in the local recording")
        self.say(self.b_bot, "bravo in the bot meeting")
        self.say(self.b_local, "bravo in the local recording")

    def say(self, meeting, sentence):
        recording = Recording.objects.get(bot=meeting, is_default_recording=True)
        participant = Participant.objects.create(bot=meeting, uuid=str(uuid.uuid4()), full_name="Speaker")
        return Utterance.objects.create(
            recording=recording,
            participant=participant,
            timestamp_ms=1000,
            duration_ms=2000,
            transcription={"transcript": sentence},
        )

    def transcript_of(self, meeting, member=MEMBER_A, client=None):
        request = (client or self.client).get(
            f"{MEETINGS_URL}/{meeting.object_id}/transcript",
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(member),
        )
        return request

    def spoken_in(self, response):
        return [row["transcription"]["transcript"] for row in response.json()]

    def test_the_bot_page_shows_only_my_bot_meetings(self):
        response = self.get_meetings(member_token(MEMBER_A), source="bot")

        self.assertEqual(self.ids_in(response), [self.a_bot.object_id])

    def test_the_local_page_shows_only_my_recordings(self):
        response = self.get_meetings(member_token(MEMBER_A), source="local")

        self.assertEqual(self.ids_in(response), [self.a_local.object_id])

    def test_each_member_sees_their_own_two_meetings(self):
        for member, expected in ((MEMBER_A, {self.a_bot, self.a_local}), (MEMBER_B, {self.b_bot, self.b_local})):
            response = self.get_meetings(member_token(member))
            self.assertEqual(
                set(self.ids_in(response)),
                {meeting.object_id for meeting in expected},
                f"member {member} got the wrong history",
            )

    def test_a_transcript_carries_my_words_and_nobody_elses(self):
        from_bot = self.transcript_of(self.a_bot, MEMBER_A)
        from_local = self.transcript_of(self.a_local, MEMBER_A)

        self.assertEqual(from_bot.status_code, 200)
        self.assertEqual(self.spoken_in(from_bot), ["alpha in the bot meeting"])
        self.assertEqual(self.spoken_in(from_local), ["alpha in the local recording"])

    def test_another_members_transcript_stays_out_of_reach(self):
        # The ownership gate runs before the inherited transcript view, so the words are never
        # read, let alone returned. Checked for both kinds: the gate is not session-specific.
        for meeting in (self.b_bot, self.b_local):
            response = self.transcript_of(meeting, MEMBER_A)
            self.assertEqual(response.status_code, 404)
            self.assertNotIn("bravo", str(response.content))

    def test_signing_in_on_another_device_returns_the_same_history(self):
        """The cross-device claim: the answer depends on the token, not the machine.

        A second Client shares no cookies, session or connection state with the first, so
        matching results mean the server derived them from the token alone.
        """
        first_device = self.get_meetings(member_token(MEMBER_A))

        second_device = Client()
        other = second_device.get(
            MEETINGS_URL,
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(MEMBER_A),
        )

        self.assertEqual(other.status_code, 200)
        self.assertEqual(self.ids_in(other), self.ids_in(first_device))
        self.assertEqual(
            self.spoken_in(self.transcript_of(self.a_bot, MEMBER_A, client=second_device)),
            ["alpha in the bot meeting"],
        )
