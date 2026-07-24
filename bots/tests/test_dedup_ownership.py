"""Ownership when a bot dispatch is deduplicated.

A bot already holding a meeting's slot is returned instead of a second one being created
(the ``unique_bot_meeting_dedup_key`` constraint). Two things must both hold on that path, and
they pull in opposite directions:

* an **unowned** existing bot must be *claimable* -- otherwise a bot dispatched before ownership
  existed, or while signed out, stays invisible to everyone forever, and
* an **owned** existing bot must be *untouchable* -- otherwise a second member dispatching into
  someone's live meeting silently takes it over.

A single ``UPDATE ... WHERE owner_user_id IS NULL`` satisfies both: it fills an empty owner and
matches zero rows on a filled one. These tests pin each direction, because a fix for one is easy
to write in a way that breaks the other.
"""

from django.test import override_settings

from bots.models import Bot, BotStates, SessionTypes

from .meetings_api_base import JWT_SECRET, MEMBER_A, MeetingsApiTestBase, member_token

MEETING_URL = "https://meet.google.com/aaa-bbbb-ccc"
DEDUP_KEY = "meet:aaa-bbbb-ccc"
OTHER_MEMBER = "9999"


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestDedupOwnership(MeetingsApiTestBase):
    def existing_live_bot(self, owner_user_id):
        """A bot already holding the meeting slot, so a fresh dispatch to it deduplicates."""
        return Bot.objects.create(
            project=self.project,
            meeting_url=MEETING_URL,
            name="Live",
            state=BotStates.JOINING,
            session_type=SessionTypes.BOT,
            owner_user_id=owner_user_id,
            meeting_dedup_key=DEDUP_KEY,
        )

    def dispatch(self, member):
        return self.client.post(
            "/api/v1/bots",
            data={"meeting_url": MEETING_URL, "bot_name": "Test Bot"},
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(member),
        )

    def test_dedup_claims_a_bot_that_had_no_owner(self):
        bot = self.existing_live_bot(owner_user_id=None)

        response = self.dispatch(MEMBER_A)

        self.assertEqual(response.status_code, 200, response.content)
        self.assertTrue(response.json().get("deduplicated"))
        bot.refresh_from_db()
        self.assertEqual(bot.owner_user_id, MEMBER_A, "an unowned bot was not claimed on dedup")

    def test_dedup_never_reassigns_an_already_owned_bot(self):
        bot = self.existing_live_bot(owner_user_id=OTHER_MEMBER)

        response = self.dispatch(MEMBER_A)  # a different member dispatches into it

        self.assertEqual(response.status_code, 200, response.content)
        bot.refresh_from_db()
        self.assertEqual(bot.owner_user_id, OTHER_MEMBER, "a member stole an owned bot via dedup")
