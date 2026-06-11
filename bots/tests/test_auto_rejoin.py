from datetime import timedelta
from unittest import mock

from django.test import TestCase
from django.utils import timezone

from accounts.models import Organization
from bots.bot_recovery_utils import MAX_REPLACEMENTS_PER_MEETING_PER_HOUR, maybe_create_replacement_bot
from bots.models import (
    Bot,
    BotEvent,
    BotEventManager,
    BotEventSubTypes,
    BotEventTypes,
    BotStates,
    Project,
    Recording,
    RecordingTypes,
    TranscriptionProviders,
    TranscriptionTypes,
)

MEETING_KEY = "meet:abc-defg-hij"
RECOVERY_ON = {"recovery_settings": {"auto_rejoin_on_failure": True}}


class AutoRejoinTestBase(TestCase):
    def setUp(self):
        self.organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=self.organization)

    def create_dead_bot(self, setting_on=True, joined=True, state=BotStates.FATAL_ERROR, meeting_dedup_key=MEETING_KEY, metadata=None):
        now_timestamp = int(timezone.now().timestamp())
        bot = Bot.objects.create(
            project=self.project,
            meeting_url="https://meet.google.com/abc-defg-hij",
            name="Crash Bot",
            state=state,
            meeting_dedup_key=meeting_dedup_key,
            metadata=metadata,
            settings=dict(RECOVERY_ON) if setting_on else {},
            first_heartbeat_timestamp=now_timestamp,
            last_heartbeat_timestamp=now_timestamp,
        )
        Recording.objects.create(
            bot=bot,
            recording_type=RecordingTypes.AUDIO_AND_VIDEO,
            transcription_type=TranscriptionTypes.NON_REALTIME,
            transcription_provider=TranscriptionProviders.ELEVENLABS,
            is_default_recording=True,
        )
        if joined:
            BotEvent.objects.create(bot=bot, old_state=BotStates.JOINING, new_state=BotStates.JOINED_NOT_RECORDING, event_type=BotEventTypes.BOT_JOINED_MEETING)
        return bot


@mock.patch("bots.launch_bot_utils.launch_bot")
class TestAutoRejoinPolicy(AutoRejoinTestBase):
    def test_replaces_on_each_allowlisted_crash_type(self, mock_launch):
        for index, sub_type in enumerate([BotEventSubTypes.FATAL_ERROR_HEARTBEAT_TIMEOUT, BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED, BotEventSubTypes.FATAL_ERROR_ATTENDEE_INTERNAL_ERROR]):
            dead_bot = self.create_dead_bot(meeting_dedup_key=f"meet:aaa-bbb{index}-ccc")

            replacement = maybe_create_replacement_bot(dead_bot.id, sub_type)

            self.assertIsNotNone(replacement, f"sub_type {sub_type} should replace")
            self.assertEqual(replacement.state, BotStates.JOINING)
            self.assertEqual(replacement.metadata["replaces"], dead_bot.object_id)
            self.assertEqual(replacement.settings, dead_bot.settings)  # incl. recovery setting itself
            self.assertEqual(replacement.meeting_dedup_key, dead_bot.meeting_dedup_key)
            replacement_recording = Recording.objects.get(bot=replacement, is_default_recording=True)
            self.assertEqual(replacement_recording.transcription_provider, TranscriptionProviders.ELEVENLABS)
            mock_launch.assert_called_with(replacement)

    def test_never_replaces_non_allowlisted_failures(self, mock_launch):
        dead_bot = self.create_dead_bot()
        for sub_type in [
            BotEventSubTypes.FATAL_ERROR_OUT_OF_CREDITS,
            BotEventSubTypes.FATAL_ERROR_GLOBAL_RUNTIME_TIMEOUT,
            BotEventSubTypes.FATAL_ERROR_BOT_NOT_LAUNCHED,
            BotEventSubTypes.FATAL_ERROR_UI_ELEMENT_NOT_FOUND,
            BotEventSubTypes.FATAL_ERROR_RTMP_CONNECTION_FAILED,
        ]:
            self.assertIsNone(maybe_create_replacement_bot(dead_bot.id, sub_type), f"sub_type {sub_type} must not replace")
        self.assertEqual(Bot.objects.count(), 1)
        mock_launch.assert_not_called()

    def test_setting_off_never_replaces(self, mock_launch):
        dead_bot = self.create_dead_bot(setting_on=False)
        self.assertIsNone(maybe_create_replacement_bot(dead_bot.id, BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED))
        mock_launch.assert_not_called()

    def test_never_joined_never_replaces(self, mock_launch):
        dead_bot = self.create_dead_bot(joined=False)
        self.assertIsNone(maybe_create_replacement_bot(dead_bot.id, BotEventSubTypes.FATAL_ERROR_HEARTBEAT_TIMEOUT))
        mock_launch.assert_not_called()

    @mock.patch("accounts.models.Organization.out_of_credits", return_value=True)
    def test_out_of_credits_stands_down(self, mock_credits, mock_launch):
        dead_bot = self.create_dead_bot()
        self.assertIsNone(maybe_create_replacement_bot(dead_bot.id, BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED))
        mock_launch.assert_not_called()

    @mock.patch("bots.bots_api_utils.validate_bot_concurrency_limit", return_value={"error": "limit"})
    def test_concurrency_limit_stands_down(self, mock_limit, mock_launch):
        dead_bot = self.create_dead_bot()
        self.assertIsNone(maybe_create_replacement_bot(dead_bot.id, BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED))
        mock_launch.assert_not_called()

    def test_cap_stands_down_after_three_recent_replacements(self, mock_launch):
        original = self.create_dead_bot()
        previous = original
        for _ in range(MAX_REPLACEMENTS_PER_MEETING_PER_HOUR):
            previous = self.create_dead_bot(metadata={"replaces": previous.object_id})

        self.assertIsNone(maybe_create_replacement_bot(previous.id, BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED))
        mock_launch.assert_not_called()

    def test_cap_ignores_replacements_older_than_an_hour(self, mock_launch):
        original = self.create_dead_bot()
        previous = original
        for _ in range(MAX_REPLACEMENTS_PER_MEETING_PER_HOUR):
            previous = self.create_dead_bot(metadata={"replaces": previous.object_id})
        # Age all but the last replacement out of the 1-hour window
        Bot.objects.filter(id__in=[b.id for b in Bot.objects.all() if b.id not in (previous.id, original.id)]).update(created_at=timezone.now() - timedelta(hours=2))

        replacement = maybe_create_replacement_bot(previous.id, BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED)

        self.assertIsNotNone(replacement)

    def test_stands_down_when_meeting_already_covered(self, mock_launch):
        dead_bot = self.create_dead_bot()
        # Another active bot already holds this meeting's dedup slot (e.g. a user re-requested)
        Bot.objects.create(project=self.project, meeting_url=dead_bot.meeting_url, name="User Bot", state=BotStates.JOINING, meeting_dedup_key=MEETING_KEY)

        self.assertIsNone(maybe_create_replacement_bot(dead_bot.id, BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED))
        self.assertEqual(Bot.objects.count(), 2)
        mock_launch.assert_not_called()

    @mock.patch("bots.bot_recovery_utils._maybe_create_replacement_bot", side_effect=Exception("boom"))
    def test_wrapper_never_raises(self, mock_inner, mock_launch):
        self.assertIsNone(maybe_create_replacement_bot(12345, BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED))


@mock.patch("bots.launch_bot_utils.launch_bot")
class TestAutoRejoinHook(AutoRejoinTestBase):
    def create_live_bot(self, setting_on=True):
        bot = self.create_dead_bot(setting_on=setting_on, state=BotStates.JOINED_NOT_RECORDING)
        return bot

    def test_fatal_error_transition_triggers_exactly_one_replacement(self, mock_launch):
        bot = self.create_live_bot()

        with self.captureOnCommitCallbacks(execute=True):
            BotEventManager.create_event(bot=bot, event_type=BotEventTypes.FATAL_ERROR, event_sub_type=BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED)

        replacements = Bot.objects.exclude(id=bot.id)
        self.assertEqual(replacements.count(), 1)
        self.assertEqual(replacements.first().metadata["replaces"], bot.object_id)
        self.assertEqual(mock_launch.call_count, 1)

    def test_fatal_error_transition_with_setting_off_does_not_replace(self, mock_launch):
        bot = self.create_live_bot(setting_on=False)

        with self.captureOnCommitCallbacks(execute=True):
            BotEventManager.create_event(bot=bot, event_type=BotEventTypes.FATAL_ERROR, event_sub_type=BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED)

        self.assertEqual(Bot.objects.count(), 1)
        mock_launch.assert_not_called()
