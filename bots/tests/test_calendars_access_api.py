"""Ownership isolation for calendar connections and their events.

One project API key is shared by every desktop install, so (as with ``/meetings``, see
``test_meetings_access_api.py``) the ``X-User-Token`` is what decides whose calendar a request
may touch. A calendar is simpler than a meeting though: it is never shared between members, so
there is no viewer list -- ``owner_user_id`` alone gates every read, write, and delete, and (like
a meeting) a calendar that isn't yours must be indistinguishable from one that does not exist.
"""

from datetime import timedelta
from uuid import uuid4

from django.test import Client, TestCase, override_settings
from django.utils import timezone

from accounts.models import Organization
from bots.models import ApiKey, Calendar, CalendarEvent, CalendarPlatform, CalendarStates, Project

from .meetings_api_base import JWT_SECRET, MEMBER_A, MEMBER_B, member_token

CALENDARS_URL = "/api/v1/calendars"
CALENDAR_EVENTS_URL = "/api/v1/calendar_events"


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class CalendarsAccessApiTestBase(TestCase):
    def setUp(self):
        self.organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=self.organization)
        self.api_key, self.api_key_plain = ApiKey.create(project=self.project, name="Test API Key")
        self.client = Client()

    def make_calendar(self, owner_user_id=MEMBER_A, *, deduplication_key=None, project=None):
        return Calendar.objects.create(
            project=project or self.project,
            platform=CalendarPlatform.GOOGLE,
            client_id="client_id",
            state=CalendarStates.CONNECTED,
            deduplication_key=deduplication_key,
            owner_user_id=owner_user_id,
        )

    def make_event(self, calendar, **overrides):
        now = timezone.now()
        defaults = {
            "calendar": calendar,
            "platform_uuid": f"uuid-{uuid4()}",
            "start_time": now + timedelta(hours=1),
            "end_time": now + timedelta(hours=2),
            "name": "Standup",
            "raw": {"title": "Standup"},
        }
        defaults.update(overrides)
        return CalendarEvent.objects.create(**defaults)

    def headers(self, token=None):
        """``token=None`` means "send no X-User-Token" -- matches meetings_api_base's convention."""
        headers = {"HTTP_AUTHORIZATION": f"Token {self.api_key_plain}"}
        if token is not None:
            headers["HTTP_X_USER_TOKEN"] = token
        return headers

    def list_calendars(self, token=None):
        return self.client.get(CALENDARS_URL, **self.headers(token))

    def get_calendar(self, object_id, token=None):
        return self.client.get(f"{CALENDARS_URL}/{object_id}", **self.headers(token))

    def patch_calendar(self, object_id, data, token=None):
        return self.client.patch(f"{CALENDARS_URL}/{object_id}", data=data, content_type="application/json", **self.headers(token))

    def delete_calendar(self, object_id, token=None):
        return self.client.delete(f"{CALENDARS_URL}/{object_id}", **self.headers(token))

    def create_calendar_via_api(self, token=None, **overrides):
        data = {
            "platform": CalendarPlatform.GOOGLE,
            "client_id": "new_client_id",
            "client_secret": "new_client_secret",
            "refresh_token": "new_refresh_token",
        }
        data.update(overrides)
        return self.client.post(CALENDARS_URL, data=data, content_type="application/json", **self.headers(token))

    def list_events(self, token=None, **query):
        return self.client.get(CALENDAR_EVENTS_URL, query, **self.headers(token))


class TestNoTokenIsRejected(CalendarsAccessApiTestBase):
    """Every member-facing view requires a verified token -- there is no optional path here as
    there is for bot dispatch, so a missing token is always a 401, never "unowned"."""

    def test_list_calendars_requires_a_token(self):
        self.assertEqual(self.list_calendars(token=None).status_code, 401)

    def test_create_calendar_requires_a_token(self):
        self.assertEqual(self.create_calendar_via_api(token=None).status_code, 401)

    def test_get_calendar_requires_a_token(self):
        calendar = self.make_calendar(MEMBER_A)
        self.assertEqual(self.get_calendar(calendar.object_id, token=None).status_code, 401)

    def test_patch_calendar_requires_a_token(self):
        calendar = self.make_calendar(MEMBER_A)
        self.assertEqual(self.patch_calendar(calendar.object_id, {"metadata": {"a": "b"}}, token=None).status_code, 401)

    def test_delete_calendar_requires_a_token(self):
        calendar = self.make_calendar(MEMBER_A)
        self.assertEqual(self.delete_calendar(calendar.object_id, token=None).status_code, 401)

    def test_list_events_requires_a_token(self):
        self.assertEqual(self.list_events(token=None).status_code, 401)


class TestCalendarListIsolation(CalendarsAccessApiTestBase):
    def test_a_member_sees_only_their_own_calendars(self):
        mine = self.make_calendar(MEMBER_A, deduplication_key="mine")
        self.make_calendar(MEMBER_B, deduplication_key="theirs")

        response = self.list_calendars(member_token(MEMBER_A))

        self.assertEqual(response.status_code, 200)
        ids = [row["id"] for row in response.json()["results"]]
        self.assertEqual(ids, [mine.object_id])

    def test_an_unowned_calendar_is_not_listed_for_anyone(self):
        self.make_calendar(owner_user_id=None)

        response = self.list_calendars(member_token(MEMBER_A))

        self.assertEqual(response.json()["results"], [])


class TestCalendarDetailIsolation(CalendarsAccessApiTestBase):
    def test_my_own_calendar_is_readable(self):
        calendar = self.make_calendar(MEMBER_A)

        response = self.get_calendar(calendar.object_id, member_token(MEMBER_A))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], calendar.object_id)

    def test_another_members_calendar_looks_like_it_does_not_exist(self):
        calendar = self.make_calendar(MEMBER_B)

        response = self.get_calendar(calendar.object_id, member_token(MEMBER_A))

        self.assertEqual(response.status_code, 404)

    def test_an_unowned_calendar_is_not_readable_either(self):
        calendar = self.make_calendar(owner_user_id=None)

        response = self.get_calendar(calendar.object_id, member_token(MEMBER_A))

        self.assertEqual(response.status_code, 404)

    def test_owner_user_id_is_never_exposed_in_the_response(self):
        calendar = self.make_calendar(MEMBER_A)

        response = self.get_calendar(calendar.object_id, member_token(MEMBER_A))

        self.assertNotIn("owner_user_id", response.json())


class TestCalendarPatchIsolation(CalendarsAccessApiTestBase):
    def test_another_members_calendar_cannot_be_patched(self):
        calendar = self.make_calendar(MEMBER_B)

        response = self.patch_calendar(calendar.object_id, {"metadata": {"stolen": True}}, member_token(MEMBER_A))

        self.assertEqual(response.status_code, 404)
        calendar.refresh_from_db()
        self.assertIsNone(calendar.metadata, "someone else's calendar was modified")

    def test_my_own_calendar_can_be_patched(self):
        calendar = self.make_calendar(MEMBER_A)

        response = self.patch_calendar(calendar.object_id, {"metadata": {"team": "backend"}}, member_token(MEMBER_A))

        self.assertEqual(response.status_code, 200)
        calendar.refresh_from_db()
        self.assertEqual(calendar.metadata, {"team": "backend"})


class TestCalendarDeleteIsolation(CalendarsAccessApiTestBase):
    def test_another_members_calendar_cannot_be_deleted(self):
        calendar = self.make_calendar(MEMBER_B)

        response = self.delete_calendar(calendar.object_id, member_token(MEMBER_A))

        self.assertEqual(response.status_code, 404)
        self.assertTrue(Calendar.objects.filter(id=calendar.id).exists(), "someone else's calendar was deleted")

    def test_my_own_calendar_can_be_deleted(self):
        calendar = self.make_calendar(MEMBER_A)

        response = self.delete_calendar(calendar.object_id, member_token(MEMBER_A))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Calendar.objects.filter(id=calendar.id).exists())


class TestCalendarEventIsolation(CalendarsAccessApiTestBase):
    def test_another_members_calendar_events_are_not_listed(self):
        their_calendar = self.make_calendar(MEMBER_B)
        self.make_event(their_calendar)

        response = self.list_events(member_token(MEMBER_A))

        self.assertEqual(response.json()["results"], [])

    def test_my_own_calendar_events_are_listed(self):
        my_calendar = self.make_calendar(MEMBER_A)
        event = self.make_event(my_calendar)

        response = self.list_events(member_token(MEMBER_A))

        ids = [row["id"] for row in response.json()["results"]]
        self.assertEqual(ids, [event.object_id])

    def test_an_unowned_calendars_events_are_not_listed_either(self):
        unowned = self.make_calendar(owner_user_id=None)
        self.make_event(unowned)

        response = self.list_events(member_token(MEMBER_A))

        self.assertEqual(response.json()["results"], [])


class TestCalendarCreateOwnership(CalendarsAccessApiTestBase):
    """Stamping the owner on create, and the claim-never-steal guarantee.

    Unlike bot dispatch, a calendar is never deduplicated into an *existing* row for a second
    caller to claim -- a colliding deduplication_key raises IntegrityError before any row is
    touched (see calendars_api_utils.create_calendar), so there is no "unowned calendar becomes
    claimable" case to test here, only "an owned one is untouchable".
    """

    def test_the_creating_member_becomes_the_owner(self):
        response = self.create_calendar_via_api(member_token(MEMBER_A), deduplication_key="member-a-cal")

        self.assertEqual(response.status_code, 201, response.content)
        calendar = Calendar.objects.get(object_id=response.json()["id"])
        self.assertEqual(calendar.owner_user_id, MEMBER_A)

    def test_a_second_members_post_with_the_same_dedup_key_does_not_steal_ownership(self):
        first = self.create_calendar_via_api(member_token(MEMBER_A), deduplication_key="shared-key")
        self.assertEqual(first.status_code, 201, first.content)
        calendar_id = first.json()["id"]

        second = self.create_calendar_via_api(member_token(MEMBER_B), deduplication_key="shared-key")

        self.assertEqual(second.status_code, 400, "a colliding dedup key should be rejected, not silently taken over")
        calendar = Calendar.objects.get(object_id=calendar_id)
        self.assertEqual(calendar.owner_user_id, MEMBER_A, "a member stole another member's calendar via a dedup-key collision")
        self.assertEqual(Calendar.objects.count(), 1)

    def test_owner_user_id_is_never_exposed_in_the_create_response(self):
        response = self.create_calendar_via_api(member_token(MEMBER_A), deduplication_key="no-leak")

        self.assertNotIn("owner_user_id", response.json())
