"""The 0093 backfill: an existing meeting's owner becomes its first viewer.

This is what keeps history from vanishing when the read query later switches from owner_user_id
to viewer containment -- every already-owned meeting must carry its owner in the new list. The
migration's function is exercised directly against the real model registry (it takes ``apps``
and uses ``get_model``), which tests the exact code the migration runs without reconstructing a
historical schema.
"""

import importlib

from django.apps import apps as global_apps
from django.test import TestCase

from accounts.models import Organization
from bots.models import Bot, BotStates, Project, SessionTypes

backfill = importlib.import_module(
    "bots.migrations.0093_bot_viewer_user_ids_bot_bot_viewers_gin"
).backfill_viewers_from_owner


class TestViewerBackfill(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            name="P", organization=Organization.objects.create(name="Org")
        )

    def make(self, owner):
        return Bot.objects.create(
            project=self.project,
            meeting_url="https://meet.google.com/aaa-bbbb-ccc",
            name="M",
            state=BotStates.ENDED,
            session_type=SessionTypes.BOT,
            owner_user_id=owner,
        )

    def test_an_owned_meeting_gains_its_owner_as_sole_viewer(self):
        bot = self.make(owner="1001")
        self.assertEqual(bot.viewer_user_ids, [])  # default before backfill

        backfill(global_apps, None)

        bot.refresh_from_db()
        self.assertEqual(bot.viewer_user_ids, ["1001"])

    def test_an_unowned_meeting_gets_no_viewers(self):
        # An unowned bot must stay unowned -- giving it any viewer would leak it into a history.
        bot = self.make(owner=None)

        backfill(global_apps, None)

        bot.refresh_from_db()
        self.assertEqual(bot.viewer_user_ids, [])

    def test_backfill_is_idempotent(self):
        # Migrations can be re-run in some recovery flows; a second pass must not duplicate the
        # owner into the list.
        bot = self.make(owner="1001")

        backfill(global_apps, None)
        backfill(global_apps, None)

        bot.refresh_from_db()
        self.assertEqual(bot.viewer_user_ids, ["1001"])
