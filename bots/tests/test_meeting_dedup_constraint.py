from django.db import IntegrityError, transaction
from django.test import TestCase

from accounts.models import Organization
from bots.models import Bot, BotStates, Project


class TestMeetingDedupKeyConstraint(TestCase):
    """DB-level tests for unique_bot_meeting_dedup_key: at most one bot in a holds-meeting-slot
    state per (project, meeting_dedup_key). Rows with a NULL key never participate."""

    MEETING_KEY = "meet:abc-defg-hij"

    def setUp(self):
        self.organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=self.organization)
        self.other_project = Project.objects.create(name="Other Project", organization=self.organization)

    def create_bot(self, project=None, state=BotStates.JOINING, meeting_dedup_key=MEETING_KEY):
        return Bot.objects.create(
            project=project or self.project,
            meeting_url="https://meet.google.com/abc-defg-hij",
            name="Test Bot",
            state=state,
            meeting_dedup_key=meeting_dedup_key,
        )

    def test_second_active_bot_with_same_key_is_rejected(self):
        self.create_bot(state=BotStates.JOINING)
        with self.assertRaises(IntegrityError) as ctx:
            with transaction.atomic():
                self.create_bot(state=BotStates.READY)
        self.assertIn("unique_bot_meeting_dedup_key", str(ctx.exception))

    def test_every_holds_slot_state_blocks_a_second_bot(self):
        for state in BotStates.holds_meeting_slot_states():
            first = self.create_bot(state=state)
            with self.assertRaises(IntegrityError, msg=f"state {state} did not hold the slot"):
                with transaction.atomic():
                    self.create_bot(state=BotStates.JOINING)
            first.delete()

    def test_post_processing_bot_frees_the_slot(self):
        # A bot that left the meeting (uploading/transcribing) must not block a replacement.
        self.create_bot(state=BotStates.POST_PROCESSING)
        replacement = self.create_bot(state=BotStates.JOINING)
        self.assertIsNotNone(replacement.id)

    def test_terminal_state_bots_free_the_slot(self):
        for state in BotStates.post_meeting_states():
            self.create_bot(state=state)
        new_bot = self.create_bot(state=BotStates.JOINING)
        self.assertIsNotNone(new_bot.id)

    def test_same_key_in_different_projects_is_allowed(self):
        self.create_bot(project=self.project, state=BotStates.JOINING)
        other = self.create_bot(project=self.other_project, state=BotStates.JOINING)
        self.assertIsNotNone(other.id)

    def test_null_keys_never_conflict(self):
        first = self.create_bot(state=BotStates.JOINING, meeting_dedup_key=None)
        second = self.create_bot(state=BotStates.JOINING, meeting_dedup_key=None)
        self.assertIsNotNone(first.id)
        self.assertIsNotNone(second.id)

    def test_different_keys_in_same_project_are_allowed(self):
        self.create_bot(state=BotStates.JOINING, meeting_dedup_key="meet:abc-defg-hij")
        second = self.create_bot(state=BotStates.JOINING, meeting_dedup_key="zoom:84278559171")
        self.assertIsNotNone(second.id)

    def test_slot_frees_when_existing_bot_transitions_out_of_meeting(self):
        # Simulates the kicked-and-re-add flow at the DB level: blocked while active, allowed
        # once the first bot's state moves past the meeting.
        first = self.create_bot(state=BotStates.JOINED_RECORDING)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                self.create_bot(state=BotStates.JOINING)
        Bot.objects.filter(id=first.id).update(state=BotStates.POST_PROCESSING)
        replacement = self.create_bot(state=BotStates.JOINING)
        self.assertIsNotNone(replacement.id)
