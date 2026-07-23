"""Search and date filtering on the member history list.

The filters narrow a queryset that is already scoped to one member, so the tests that matter
most here are not "does it find things" but "can it ever find *someone else's* things", and
"does it still hold on page two". A filter that works on page one and quietly stops applying
when the cursor is followed is worse than no filter at all.

The literal ``%`` and ``_`` cases are here because those are LIKE wildcards. Django escapes
them, but that is a promise worth pinning rather than assuming.
"""

from datetime import timedelta
from urllib.parse import urlparse

from django.test import override_settings
from django.utils import timezone

from bots.models import SessionTypes

from .meetings_api_base import JWT_SECRET, MEMBER_A, MEMBER_B, MeetingsApiTestBase, member_token


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestMeetingSearch(MeetingsApiTestBase):
    def search(self, term, member=MEMBER_A, **extra):
        return self.get_meetings(member_token(member), q=term, **extra)

    def test_a_name_is_searchable(self):
        standup = self.make_meeting(name="Monday standup")
        self.make_meeting(name="Design review")

        self.assertEqual(self.ids_in(self.search("standup")), [standup.object_id])

    def test_a_meeting_link_is_searchable(self):
        zoom = self.make_meeting(name="Untitled", meeting_url="https://zoom.us/j/123456789")
        self.make_meeting(name="Untitled", meeting_url="https://meet.google.com/abc-defg-hij")

        self.assertEqual(self.ids_in(self.search("zoom.us")), [zoom.object_id])

    def test_matching_nothing_returns_nothing(self):
        self.make_meeting(name="Monday standup")

        self.assertEqual(self.ids_in(self.search("retrospective")), [])

    def test_case_does_not_matter(self):
        meeting = self.make_meeting(name="Monday Standup")

        self.assertEqual(self.ids_in(self.search("STANDUP")), [meeting.object_id])

    def test_an_empty_term_filters_nothing(self):
        # "" must mean "no search", not "match nothing" -- a cleared search box would otherwise
        # empty the list instead of restoring it.
        meeting = self.make_meeting(name="Monday standup")

        self.assertEqual(self.ids_in(self.search("")), [meeting.object_id])
        self.assertEqual(self.ids_in(self.search("   ")), [meeting.object_id])

    def test_like_wildcards_are_searched_for_literally(self):
        # % and _ are LIKE wildcards. Unescaped, "100%" would match everything.
        percent = self.make_meeting(name="100% done")
        self.make_meeting(name="100X done")
        underscore = self.make_meeting(name="draft_two")
        self.make_meeting(name="draftXtwo")

        self.assertEqual(self.ids_in(self.search("100%")), [percent.object_id])
        self.assertEqual(self.ids_in(self.search("draft_two")), [underscore.object_id])

    def test_non_ascii_names_are_searchable(self):
        meeting = self.make_meeting(name="Café planning")

        self.assertEqual(self.ids_in(self.search("Café")), [meeting.object_id])

    def test_a_null_byte_is_refused_rather_than_crashing(self):
        # A NUL survives Django's query parsing, but Postgres refuses it in a string literal,
        # so an unguarded term raises ValueError inside the driver -- a 500 for a bad request.
        self.make_meeting(name="Monday standup")

        response = self.search("stand\x00up")

        self.assertEqual(response.status_code, 400)
        self.assertIn("null bytes", response.json()["error"])

    def test_an_absurdly_long_term_is_refused(self):
        # Longer than any column being searched, so it could never have matched -- but without
        # a cap it still hands the database a megabyte-long LIKE pattern to evaluate.
        response = self.search("x" * 5000)

        self.assertEqual(response.status_code, 400)
        self.assertIn("at most", response.json()["error"])

    def test_search_cannot_reach_another_members_meeting(self):
        # The whole point: a filter narrows a set that is already mine, so however specific the
        # term, it can never widen the answer to somebody else's row.
        self.make_meeting(MEMBER_B, name="Monday standup")

        self.assertEqual(self.ids_in(self.search("standup", member=MEMBER_A)), [])

    def test_search_combines_with_source(self):
        bot_standup = self.make_meeting(name="Monday standup", session_type=SessionTypes.BOT)
        self.make_meeting(name="Monday standup", session_type=SessionTypes.LOCAL)

        found = self.search("standup", source="bot")

        self.assertEqual(self.ids_in(found), [bot_standup.object_id])

    def test_search_still_applies_on_the_second_page(self):
        # A filter that silently stops applying once the cursor is followed looks correct on
        # page one and lies from page two onwards.
        matching = {self.make_meeting(name=f"Standup {i}").object_id for i in range(30)}
        self.make_meeting(name="Design review")

        first = self.search("standup")
        first_ids = self.ids_in(first)
        self.assertEqual(len(first_ids), 25)

        parsed = urlparse(first.json()["next"])
        second_ids = self.ids_in(self.get_meetings(member_token(), url=f"{parsed.path}?{parsed.query}"))

        self.assertEqual(len(second_ids), 5)
        self.assertEqual(set(first_ids) | set(second_ids), matching)


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestMeetingDateRange(MeetingsApiTestBase):
    def setUp(self):
        super().setUp()
        self.now = timezone.now()
        self.today = self.make_meeting(name="Today", created_at=self.now)
        self.last_week = self.make_meeting(name="Last week", created_at=self.now - timedelta(days=7))

    def test_created_after_keeps_only_the_recent_ones(self):
        cutoff = (self.now - timedelta(days=1)).isoformat()

        found = self.get_meetings(member_token(), created_after=cutoff)

        self.assertEqual(self.ids_in(found), [self.today.object_id])

    def test_created_before_keeps_only_the_older_ones(self):
        cutoff = (self.now - timedelta(days=1)).isoformat()

        found = self.get_meetings(member_token(), created_before=cutoff)

        self.assertEqual(self.ids_in(found), [self.last_week.object_id])

    def test_the_range_is_half_open(self):
        """after is inclusive, before is exclusive -- so adjacent ranges tile exactly.

        Without that, a meeting landing precisely on a boundary is either counted twice or
        missed entirely when the desktop pages through consecutive days.
        """
        exactly = self.today.created_at.isoformat()

        self.assertIn(self.today.object_id, self.ids_in(self.get_meetings(member_token(), created_after=exactly)))
        self.assertNotIn(self.today.object_id, self.ids_in(self.get_meetings(member_token(), created_before=exactly)))

    def test_a_plain_date_is_accepted_and_means_midnight(self):
        found = self.get_meetings(member_token(), created_after=self.now.date().isoformat())

        self.assertEqual(self.ids_in(found), [self.today.object_id])

    def test_dates_combine_with_search(self):
        self.make_meeting(name="Today", created_at=self.now - timedelta(days=30))

        found = self.get_meetings(
            member_token(),
            q="today",
            created_after=(self.now - timedelta(days=1)).isoformat(),
        )

        self.assertEqual(self.ids_in(found), [self.today.object_id])

    def test_a_malformed_date_is_refused_rather_than_ignored(self):
        # The project dashboard swallows these and returns everything. On an API that reads as
        # a successful search over the wrong data, so it has to be an error.
        for bad in ("yesterday", "2026-13-45", "07/23/2026"):
            response = self.get_meetings(member_token(), created_after=bad)
            self.assertEqual(response.status_code, 400, f"{bad!r} was accepted")
            self.assertIn("created_after", response.json()["error"])
