import uuid

from django.test import Client, TestCase

from accounts.models import Organization
from bots.bots_api_utils import resolve_meeting_lineage
from bots.models import (
    ApiKey,
    Bot,
    BotStates,
    Participant,
    Project,
    Recording,
    RecordingStates,
    RecordingTranscriptionStates,
    RecordingTypes,
    TranscriptionProviders,
    TranscriptionTypes,
    Utterance,
)

MEETING_KEY = "meet:abc-defg-hij"


class MeetingTranscriptTestBase(TestCase):
    def setUp(self):
        self.organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=self.organization)
        self.api_key, self.api_key_plain = ApiKey.create(project=self.project, name="Test API Key")
        self.client = Client()

    def create_bot(self, project=None, meeting_dedup_key=MEETING_KEY, replaces=None, with_recording=True):
        bot = Bot.objects.create(
            project=project or self.project,
            meeting_url="https://meet.google.com/abc-defg-hij",
            name="Test Bot",
            state=BotStates.ENDED,
            meeting_dedup_key=meeting_dedup_key,
            metadata={"replaces": replaces} if replaces else None,
        )
        if with_recording:
            Recording.objects.create(
                bot=bot,
                recording_type=RecordingTypes.AUDIO_AND_VIDEO,
                transcription_type=TranscriptionTypes.NON_REALTIME,
                transcription_provider=TranscriptionProviders.ELEVENLABS,
                is_default_recording=True,
                state=RecordingStates.COMPLETE,
                transcription_state=RecordingTranscriptionStates.COMPLETE,
            )
        return bot

    def add_utterance(self, bot, timestamp_ms, duration_ms=1000, transcript="hello"):
        recording = Recording.objects.get(bot=bot, is_default_recording=True)
        participant = Participant.objects.create(bot=bot, uuid=str(uuid.uuid4()), full_name="Speaker")
        return Utterance.objects.create(
            recording=recording,
            participant=participant,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            transcription={"transcript": transcript},
        )

    def get_meeting_transcript(self, object_id, api_key_plain=None):
        return self.client.get(f"/api/v1/bots/{object_id}/meeting_transcript", HTTP_AUTHORIZATION=f"Token {api_key_plain or self.api_key_plain}")


class TestResolveMeetingLineage(MeetingTranscriptTestBase):
    def test_single_bot_lineage_is_itself(self):
        bot = self.create_bot()
        self.assertEqual([b.id for b in resolve_meeting_lineage(bot)], [bot.id])

    def test_chain_resolves_from_both_ends(self):
        bot_a = self.create_bot()
        bot_b = self.create_bot(replaces=bot_a.object_id)
        bot_c = self.create_bot(replaces=bot_b.object_id)

        expected = [bot_a.id, bot_b.id, bot_c.id]
        self.assertEqual([b.id for b in resolve_meeting_lineage(bot_a)], expected)
        self.assertEqual([b.id for b in resolve_meeting_lineage(bot_b)], expected)
        self.assertEqual([b.id for b in resolve_meeting_lineage(bot_c)], expected)

    def test_fingerprint_mismatch_breaks_link(self):
        bot_a = self.create_bot(meeting_dedup_key="meet:xxx-yyyy-zzz")
        bot_b = self.create_bot(replaces=bot_a.object_id, meeting_dedup_key=MEETING_KEY)

        self.assertEqual([b.id for b in resolve_meeting_lineage(bot_b)], [bot_b.id])
        self.assertEqual([b.id for b in resolve_meeting_lineage(bot_a)], [bot_a.id])

    def test_cross_project_replacement_not_followed(self):
        other_project = Project.objects.create(name="Other Project", organization=self.organization)
        bot_a = self.create_bot()
        self.create_bot(project=other_project, replaces=bot_a.object_id)

        self.assertEqual([b.id for b in resolve_meeting_lineage(bot_a)], [bot_a.id])

    def test_cycle_terminates(self):
        bot_a = self.create_bot()
        bot_b = self.create_bot(replaces=bot_a.object_id)
        bot_a.metadata = {"replaces": bot_b.object_id}
        bot_a.save()

        lineage = resolve_meeting_lineage(bot_a)
        self.assertEqual(sorted(b.id for b in lineage), sorted([bot_a.id, bot_b.id]))


class TestMeetingTranscriptEndpoint(MeetingTranscriptTestBase):
    def test_single_bot_transcript_no_gaps(self):
        bot = self.create_bot()
        self.add_utterance(bot, timestamp_ms=2000, transcript="second")
        self.add_utterance(bot, timestamp_ms=1000, transcript="first")

        response = self.get_meeting_transcript(bot.object_id)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["bots"], [bot.object_id])
        self.assertEqual([u["transcription"]["transcript"] for u in body["utterances"]], ["first", "second"])
        self.assertTrue(all(u["bot_id"] == bot.object_id for u in body["utterances"]))
        self.assertEqual(body["gaps"], [])

    def test_replacement_chain_merges_with_gap(self):
        bot_a = self.create_bot()
        bot_b = self.create_bot(replaces=bot_a.object_id)
        self.add_utterance(bot_a, timestamp_ms=10_000, duration_ms=5_000, transcript="part one")
        self.add_utterance(bot_a, timestamp_ms=16_000, duration_ms=4_000, transcript="part two")
        self.add_utterance(bot_b, timestamp_ms=23_000, duration_ms=2_000, transcript="part three")

        # Fetching via EITHER bot id returns the same stitched meeting
        for object_id in (bot_a.object_id, bot_b.object_id):
            response = self.get_meeting_transcript(object_id)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["bots"], [bot_a.object_id, bot_b.object_id])
            self.assertEqual([u["transcription"]["transcript"] for u in body["utterances"]], ["part one", "part two", "part three"])
            self.assertEqual(body["gaps"], [{"after_bot_id": bot_a.object_id, "before_bot_id": bot_b.object_id, "start_timestamp_ms": 20_000, "end_timestamp_ms": 23_000}])

    def test_bot_without_recording_in_lineage_is_tolerated(self):
        bot_a = self.create_bot()
        bot_b = self.create_bot(replaces=bot_a.object_id, with_recording=False)
        self.add_utterance(bot_a, timestamp_ms=1000)

        response = self.get_meeting_transcript(bot_b.object_id)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["bots"], [bot_a.object_id, bot_b.object_id])
        self.assertEqual(len(body["utterances"]), 1)
        self.assertEqual(body["gaps"], [])

    def test_empty_transcript_text_is_skipped(self):
        bot = self.create_bot()
        self.add_utterance(bot, timestamp_ms=1000, transcript="")
        self.add_utterance(bot, timestamp_ms=2000, transcript="spoken")

        response = self.get_meeting_transcript(bot.object_id)

        self.assertEqual([u["transcription"]["transcript"] for u in response.json()["utterances"]], ["spoken"])

    def test_404_for_unknown_bot(self):
        response = self.get_meeting_transcript("bot_doesnotexist000")
        self.assertEqual(response.status_code, 404)

    def test_404_for_other_projects_bot(self):
        other_project = Project.objects.create(name="Other Project", organization=self.organization)
        _, other_key_plain = ApiKey.create(project=other_project, name="Other Key")
        bot = self.create_bot()

        response = self.get_meeting_transcript(bot.object_id, api_key_plain=other_key_plain)

        self.assertEqual(response.status_code, 404)
