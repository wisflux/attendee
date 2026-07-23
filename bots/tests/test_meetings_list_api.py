"""Characterisation tests for ``GET /meetings`` -- the member history list.

These pin the behaviour the endpoint has *today*, before search and date filtering are added
on top of it. Two of them exist specifically to catch mistakes the module's own docstring
warns about, because a comment cannot fail a build:

* an unowned meeting must never appear (``filter(owner_user_id=None)`` compiles to ``IS NULL``
  and would hand every member the project's unowned rows), and
* the list must be newest-first with a unique tiebreak, or paging silently skips and repeats
  rows when several meetings share a ``created_at``.
"""

from datetime import timedelta
from urllib.parse import urlparse

from django.db import connection
from django.test import override_settings
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from accounts.models import Organization
from bots.models import BotStates, Project, SessionTypes

from .meetings_api_base import MEMBER_A, MEMBER_B, MeetingsApiTestBase, member_token


@override_settings(TEAM_DAY_JWT_SECRET="test-team-day-secret")
class TestMeetingListOwnership(MeetingsApiTestBase):
    def test_only_my_meetings_are_listed(self):
        mine = self.make_meeting(MEMBER_A)
        self.make_meeting(MEMBER_B)

        response = self.get_meetings(member_token(MEMBER_A))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.ids_in(response), [mine.object_id])

    def test_an_unowned_meeting_belongs_to_nobody(self):
        # The IS NULL trap: a project's unowned bots (dispatched before ownership existed, or
        # by an API customer sending no token) must not surface in anyone's history.
        self.make_meeting(owner_user_id=None)

        for member in (MEMBER_A, MEMBER_B):
            response = self.get_meetings(member_token(member))
            self.assertEqual(self.ids_in(response), [], f"member {member} saw an unowned meeting")

    def test_another_projects_meeting_is_not_mine(self):
        # Same person, different project: one API key must never read another's data.
        other_org = Organization.objects.create(name="Other Organization")
        other_project = Project.objects.create(name="Other Project", organization=other_org)
        self.make_meeting(MEMBER_A, project=other_project)

        response = self.get_meetings(member_token(MEMBER_A))

        self.assertEqual(self.ids_in(response), [])


@override_settings(TEAM_DAY_JWT_SECRET="test-team-day-secret")
class TestMeetingListFiltering(MeetingsApiTestBase):
    def test_source_selects_bot_or_local(self):
        bot_meeting = self.make_meeting(session_type=SessionTypes.BOT)
        local_meeting = self.make_meeting(session_type=SessionTypes.LOCAL)

        bot_only = self.get_meetings(member_token(), source="bot")
        local_only = self.get_meetings(member_token(), source="local")

        self.assertEqual(self.ids_in(bot_only), [bot_meeting.object_id])
        self.assertEqual(self.ids_in(local_only), [local_meeting.object_id])

    def test_omitting_source_returns_both_kinds(self):
        self.make_meeting(session_type=SessionTypes.BOT)
        self.make_meeting(session_type=SessionTypes.LOCAL)

        response = self.get_meetings(member_token())

        self.assertEqual(len(self.ids_in(response)), 2)

    def test_an_unknown_source_is_rejected(self):
        response = self.get_meetings(member_token(), source="carrier-pigeon")

        self.assertEqual(response.status_code, 400)
        self.assertIn("source must be one of", response.json()["error"])

    def test_deleted_and_app_sessions_are_excluded(self):
        # A deleted meeting keeps its row, so without the exclusion it returns as a blank
        # entry. An app session is neither a bot nor a local recording.
        kept = self.make_meeting(state=BotStates.ENDED)
        self.make_meeting(state=BotStates.DATA_DELETED)
        self.make_meeting(session_type=SessionTypes.APP_SESSION)

        response = self.get_meetings(member_token())

        self.assertEqual(self.ids_in(response), [kept.object_id])


@override_settings(TEAM_DAY_JWT_SECRET="test-team-day-secret")
class TestMeetingListOrderingAndPaging(MeetingsApiTestBase):
    def test_history_opens_on_the_newest_meeting(self):
        now = timezone.now()
        oldest = self.make_meeting(created_at=now - timedelta(days=2), name="Oldest")
        newest = self.make_meeting(created_at=now, name="Newest")
        middle = self.make_meeting(created_at=now - timedelta(days=1), name="Middle")

        response = self.get_meetings(member_token())

        self.assertEqual(
            self.ids_in(response),
            [newest.object_id, middle.object_id, oldest.object_id],
        )

    def test_paging_a_tied_timestamp_neither_skips_nor_repeats(self):
        # Every meeting shares one created_at, so ordering by created_at alone would leave the
        # page boundary undefined. The -id tiebreak is what makes this deterministic.
        stamped_at = timezone.now()
        expected = {self.make_meeting(created_at=stamped_at, name=f"Meeting {i}").object_id for i in range(30)}

        first = self.get_meetings(member_token())
        self.assertEqual(first.status_code, 200)
        first_ids = self.ids_in(first)
        self.assertEqual(len(first_ids), 25)

        next_url = first.json()["next"]
        self.assertIsNotNone(next_url, "30 meetings over a page size of 25 must offer a next page")

        parsed = urlparse(next_url)
        second = self.get_meetings(member_token(), url=f"{parsed.path}?{parsed.query}")
        second_ids = self.ids_in(second)

        self.assertEqual(len(second_ids), 5)
        self.assertEqual(set(first_ids) & set(second_ids), set(), "a meeting appeared on both pages")
        self.assertEqual(set(first_ids) | set(second_ids), expected, "a meeting was skipped entirely")

    def test_the_last_page_offers_no_cursor(self):
        self.make_meeting()

        response = self.get_meetings(member_token())

        self.assertIsNone(response.json()["next"])


@override_settings(TEAM_DAY_JWT_SECRET="test-team-day-secret")
class TestMeetingListQueryCount(MeetingsApiTestBase):
    def queries_to_list(self):
        """How many queries one history request costs right now."""
        with CaptureQueriesContext(connection) as captured:
            response = self.get_meetings(member_token())
        self.assertEqual(response.status_code, 200)
        return len(captured)

    def test_a_full_page_costs_the_same_as_a_single_row(self):
        """The N+1 guard: prefetch + annotate mean page size must not drive query count.

        Asserted as "25 rows cost what 1 row costs" rather than a fixed number, so the test
        still means something if auth or throttling changes how many queries surround the view.
        """
        self.make_meeting(name="Only one")
        # Warm up: the first request of a process pays one-off lookups that would otherwise be
        # charged to the single-row measurement and mask a real regression.
        self.get_meetings(member_token())
        one_row = self.queries_to_list()

        for i in range(24):
            self.make_meeting(name=f"Meeting {i}")
        full_page = self.queries_to_list()

        self.assertEqual(
            full_page,
            one_row,
            f"a 25-row page cost {full_page} queries where 1 row cost {one_row} -- the N+1 is back",
        )
