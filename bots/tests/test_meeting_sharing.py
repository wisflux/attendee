"""Sharing through dedup: dispatching into a live meeting adds you as a viewer.

Visibility is now "is my id in the viewer list", so a member who deduplicates into a meeting
another member is already recording gains it in their own history -- the transcript is read, not
copied, and ownership (billing) stays with whoever created the bot. These tests pin that a
sharer sees the meeting and its words, that the creator still does, and that a member who never
dispatched sees nothing.
"""

from django.test import override_settings

from bots.models import Bot, BotStates, Participant, Recording, Utterance

from .meetings_api_base import JWT_SECRET, MEMBER_A, MEMBER_B, MeetingsApiTestBase, member_token

MEETING_URL = "https://meet.google.com/aaa-bbbb-ccc"
DEDUP_KEY = "meet:aaa-bbbb-ccc"
STRANGER = "7777"


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestSharingThroughDedup(MeetingsApiTestBase):
    def setUp(self):
        super().setUp()
        # A meeting MEMBER_A is already recording: owned by A, A its sole viewer, holding the slot.
        self.bot = self.make_meeting(
            MEMBER_A,
            state=BotStates.JOINING,
            meeting_url=MEETING_URL,
            with_recording=True,
        )
        Bot.objects.filter(id=self.bot.id).update(meeting_dedup_key=DEDUP_KEY)
        self.say(self.bot, "the pricing number is forty")

    def say(self, meeting, sentence):
        recording = Recording.objects.get(bot=meeting, is_default_recording=True)
        participant = Participant.objects.create(bot=meeting, uuid=f"u-{sentence[:8]}", full_name="Speaker")
        Utterance.objects.create(
            recording=recording, participant=participant,
            timestamp_ms=1000, duration_ms=2000, transcription={"transcript": sentence},
        )

    def dispatch(self, member):
        return self.client.post(
            "/api/v1/bots",
            data={"meeting_url": MEETING_URL, "bot_name": "Bot"},
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(member),
        )

    def history_ids(self, member):
        return self.ids_in(self.get_meetings(member_token(member)))

    def test_deduping_in_puts_the_meeting_in_the_sharers_history(self):
        self.assertNotIn(self.bot.object_id, self.history_ids(MEMBER_B))  # not shared yet

        response = self.dispatch(MEMBER_B)  # B dispatches into A's live meeting -> dedup

        self.assertEqual(response.status_code, 200, response.content)
        self.assertTrue(response.json().get("deduplicated"))
        self.assertIn(self.bot.object_id, self.history_ids(MEMBER_B), "the shared meeting is missing for B")

    def test_the_creator_still_sees_it_after_it_is_shared(self):
        self.dispatch(MEMBER_B)

        self.assertIn(self.bot.object_id, self.history_ids(MEMBER_A), "sharing removed it from the creator")

    def test_ownership_stays_with_the_creator_after_sharing(self):
        self.dispatch(MEMBER_B)

        self.bot.refresh_from_db()
        self.assertEqual(self.bot.owner_user_id, MEMBER_A)
        self.assertCountEqual(self.bot.viewer_user_ids, [MEMBER_A, MEMBER_B])

    def test_a_sharer_can_read_the_transcript(self):
        self.dispatch(MEMBER_B)

        response = self.get_meetings(member_token(MEMBER_B), url=f"/api/v1/meetings/{self.bot.object_id}/transcript")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([row["transcription"]["transcript"] for row in response.json()], ["the pricing number is forty"])

    def test_a_member_who_never_dispatched_sees_nothing(self):
        self.dispatch(MEMBER_B)

        self.assertNotIn(self.bot.object_id, self.history_ids(STRANGER))
        transcript = self.get_meetings(member_token(STRANGER), url=f"/api/v1/meetings/{self.bot.object_id}/transcript")
        self.assertEqual(transcript.status_code, 404)
