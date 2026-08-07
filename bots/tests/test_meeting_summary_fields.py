"""
Tests for the meeting summary/title fields on Bot and their exposure in MeetingSerializer.

Covers both session types (bot + local) since they share the Bot model, the correct defaults on a
fresh meeting, and that a populated summary is returned to the member over the API.
"""

from django.test import override_settings

from bots.models import Bot, SessionTypes, SummaryStates
from bots.tests.meetings_api_base import MEMBER_A, MEMBER_B, MeetingsApiTestBase, member_token


@override_settings(TEAM_DAY_JWT_SECRET="test-team-day-secret")
class MeetingSummaryFieldsTest(MeetingsApiTestBase):
    def test_new_meeting_defaults_for_bot_and_local(self):
        """A fresh meeting (either session type) starts with no summary and state PENDING."""
        for session_type in (SessionTypes.BOT, SessionTypes.LOCAL):
            bot = self.make_meeting(session_type=session_type, with_recording=False)
            self.assertIsNone(bot.summary, msg=f"session_type={session_type}")
            self.assertEqual(bot.summary_state, SummaryStates.PENDING)
            self.assertIsNone(bot.summary_generated_at)

    def test_serializer_exposes_default_summary_fields(self):
        """GET /meetings includes summary (null) + summary_state (pending) by default."""
        self.make_meeting(session_type=SessionTypes.BOT)
        response = self.get_meetings(token=member_token(MEMBER_A))

        self.assertEqual(response.status_code, 200)
        row = response.json()["results"][0]
        self.assertIn("summary", row)
        self.assertIn("summary_state", row)
        self.assertIsNone(row["summary"])
        self.assertEqual(row["summary_state"], SummaryStates.PENDING.value)

    def test_serializer_returns_populated_summary_for_local(self):
        """A written summary/state is surfaced verbatim to the owning member (local recording)."""
        bot = self.make_meeting(session_type=SessionTypes.LOCAL)
        # Single-column update mirrors how the generator writes it (avoids the version-field bump).
        Bot.objects.filter(id=bot.id).update(summary="Discussed the Q3 roadmap.", summary_state=SummaryStates.DONE)

        response = self.get_meetings(token=member_token(MEMBER_A))
        row = response.json()["results"][0]
        self.assertEqual(row["summary"], "Discussed the Q3 roadmap.")
        self.assertEqual(row["summary_state"], SummaryStates.DONE.value)

    def test_summary_is_shared_with_every_viewer(self):
        """One summary per meeting: a shared (deduped) meeting shows the same summary to each viewer."""
        bot = self.make_meeting(owner_user_id=MEMBER_A, viewer_user_ids=[MEMBER_A, MEMBER_B], session_type=SessionTypes.BOT)
        Bot.objects.filter(id=bot.id).update(summary="Kickoff notes.", summary_state=SummaryStates.DONE)

        for member in (MEMBER_A, MEMBER_B):
            row = self.get_meetings(token=member_token(member)).json()["results"][0]
            self.assertEqual(row["summary"], "Kickoff notes.", msg=f"member={member}")
            self.assertEqual(row["summary_state"], SummaryStates.DONE.value)
