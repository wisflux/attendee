"""Share links: mint one for a meeting I can see, redeem it into my own history.

The round trip and every gate that keeps it safe: only a viewer may mint a link; the raw token
is returned once and only its hash is stored; redeeming proves the redeemer by their own JWT and
refuses a link from another project, an expired one, or one for a wiped meeting -- each as a
generic error that never confirms a meeting the caller could not already reach.
"""

from datetime import timedelta

from django.test import override_settings
from django.utils import timezone

from accounts.models import Organization
from bots.models import BotStates, MeetingShareToken, Project
from bots.tests.meetings_api_base import (
    JWT_SECRET,
    MEETINGS_URL,
    MEMBER_A,
    MEMBER_B,
    MeetingsApiTestBase,
    member_token,
)

OWNER = MEMBER_A
REDEEMER = MEMBER_B


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestMeetingShareLink(MeetingsApiTestBase):
    def make_share(self, member, object_id):
        return self.client.post(
            f"{MEETINGS_URL}/{object_id}/share",
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(member),
        )

    def redeem(self, member, token, api_key_plain=None):
        return self.client.post(
            f"{MEETINGS_URL}/redeem",
            data={"token": token},
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Token {api_key_plain or self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(member),
        )

    def sees(self, member, object_id):
        return object_id in self.ids_in(self.get_meetings(member_token(member)))

    # ── minting ────────────────────────────────────────────────────────────────
    def test_a_viewer_can_mint_a_link_and_only_the_hash_is_stored(self):
        bot = self.make_meeting(OWNER)

        response = self.make_share(OWNER, bot.object_id)

        self.assertEqual(response.status_code, 201, response.content)
        raw_token = response.json()["token"]
        self.assertTrue(raw_token)
        stored = MeetingShareToken.objects.get(bot=bot)
        self.assertNotEqual(stored.token_hash, raw_token, "the raw token was stored, not its hash")
        self.assertEqual(stored.token_hash, MeetingShareToken.hash_token(raw_token))

    def test_a_non_viewer_cannot_mint_a_link(self):
        bot = self.make_meeting(OWNER)

        response = self.make_share(REDEEMER, bot.object_id)

        self.assertEqual(response.status_code, 404)
        self.assertFalse(MeetingShareToken.objects.filter(bot=bot).exists())

    # ── the round trip ───────────────────────────────────────────────────────────
    def test_redeeming_puts_the_meeting_in_the_redeemers_history(self):
        bot = self.make_meeting(OWNER)
        token = self.make_share(OWNER, bot.object_id).json()["token"]
        self.assertFalse(self.sees(REDEEMER, bot.object_id))

        response = self.redeem(REDEEMER, token)

        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(response.json()["id"], bot.object_id)
        self.assertTrue(self.sees(REDEEMER, bot.object_id))
        bot.refresh_from_db()
        self.assertCountEqual(bot.viewer_user_ids, [OWNER, REDEEMER])
        self.assertEqual(bot.owner_user_id, OWNER, "redeeming changed ownership")

    def test_redeeming_twice_is_idempotent(self):
        bot = self.make_meeting(OWNER)
        token = self.make_share(OWNER, bot.object_id).json()["token"]

        self.redeem(REDEEMER, token)
        self.redeem(REDEEMER, token)

        bot.refresh_from_db()
        self.assertEqual(bot.viewer_user_ids.count(REDEEMER), 1, "the redeemer was added twice")

    # ── security gates ───────────────────────────────────────────────────────────
    def test_a_missing_token_is_rejected(self):
        response = self.client.post(
            f"{MEETINGS_URL}/redeem",
            data={},
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(REDEEMER),
        )
        self.assertEqual(response.status_code, 400)

    def test_an_unknown_token_is_a_generic_not_found(self):
        response = self.redeem(REDEEMER, "not-a-real-token")

        self.assertEqual(response.status_code, 404)

    def test_a_non_string_token_is_refused_not_crashed(self):
        # A JSON body can carry token as an object or a number; that must be a 400, never a 500.
        for bad in ({"nested": "obj"}, 12345, ["a", "b"]):
            response = self.client.post(
                f"{MEETINGS_URL}/redeem",
                data={"token": bad},
                content_type="application/json",
                HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
                HTTP_X_USER_TOKEN=member_token(REDEEMER),
            )
            self.assertEqual(response.status_code, 400, f"{bad!r} did not yield 400")

    def test_an_expired_token_is_refused(self):
        bot = self.make_meeting(OWNER)
        token = self.make_share(OWNER, bot.object_id).json()["token"]
        MeetingShareToken.objects.filter(bot=bot).update(expires_at=timezone.now() - timedelta(seconds=1))

        response = self.redeem(REDEEMER, token)

        self.assertEqual(response.status_code, 410)
        self.assertFalse(self.sees(REDEEMER, bot.object_id))

    def test_a_link_cannot_cross_into_another_project(self):
        # A token minted in project A, presented with project B's API key, must not work: the
        # meeting belongs to a different organisation.
        bot = self.make_meeting(OWNER)
        token = self.make_share(OWNER, bot.object_id).json()["token"]
        other_project = Project.objects.create(
            name="Other", organization=Organization.objects.create(name="Other Org")
        )
        from bots.models import ApiKey

        _, other_key_plain = ApiKey.create(project=other_project, name="Other Key")

        response = self.redeem(REDEEMER, token, api_key_plain=other_key_plain)

        self.assertEqual(response.status_code, 404)
        bot.refresh_from_db()
        self.assertNotIn(REDEEMER, bot.viewer_user_ids)

    def test_a_link_for_a_wiped_meeting_cannot_be_redeemed(self):
        bot = self.make_meeting(OWNER)
        token = self.make_share(OWNER, bot.object_id).json()["token"]
        bot.state = BotStates.DATA_DELETED
        bot.save(update_fields=["state"])

        response = self.redeem(REDEEMER, token)

        self.assertEqual(response.status_code, 404)

    def test_redeeming_requires_a_valid_user_token(self):
        bot = self.make_meeting(OWNER)
        token = self.make_share(OWNER, bot.object_id).json()["token"]

        response = self.client.post(
            f"{MEETINGS_URL}/redeem",
            data={"token": token},
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",  # no X-User-Token
        )

        self.assertEqual(response.status_code, 401)
