"""Reference-counted deletion: everyone in the viewer list is equal.

Deleting removes *my* copy. The recording survives as long as anyone still holds it, and is
wiped only when the last viewer leaves. These pin every branch a person can hit:

* a shared meeting: one viewer leaves, the other keeps it *and* its transcript,
* a solely-held meeting: deleting it wipes the data,
* leaving does not depend on who created it (no owner privilege), and
* a meeting still being recorded cannot be wiped by its last viewer, but a co-viewer may still
  leave it.
"""

from django.test import override_settings

from bots.models import BotStates, Participant, Recording, Utterance

from .meetings_api_base import JWT_SECRET, MEMBER_A, MEMBER_B, MeetingsApiTestBase, member_token

CREATOR = MEMBER_A
SHARER = MEMBER_B


@override_settings(TEAM_DAY_JWT_SECRET=JWT_SECRET)
class TestReferenceCountedDelete(MeetingsApiTestBase):
    def shared_meeting(self, *, state=BotStates.ENDED, viewers=(CREATOR, SHARER)):
        """A meeting created by CREATOR and shared to SHARER (both in the viewer list)."""
        bot = self.make_meeting(CREATOR, state=state, viewer_user_ids=list(viewers))
        recording = Recording.objects.get(bot=bot, is_default_recording=True)
        participant = Participant.objects.create(bot=bot, uuid="u-1", full_name="Speaker")
        Utterance.objects.create(
            recording=recording, participant=participant,
            timestamp_ms=1000, duration_ms=2000, transcription={"transcript": "shared words"},
        )
        return bot

    def delete(self, object_id, member):
        return self.client.delete(
            f"/api/v1/meetings/{object_id}",
            HTTP_AUTHORIZATION=f"Token {self.api_key_plain}",
            HTTP_X_USER_TOKEN=member_token(member),
        )

    def sees(self, member, object_id):
        return object_id in self.ids_in(self.get_meetings(member_token(member)))

    def test_a_co_viewer_leaving_keeps_the_meeting_for_the_other(self):
        bot = self.shared_meeting()

        response = self.delete(bot.object_id, SHARER)

        self.assertEqual(response.status_code, 204)
        bot.refresh_from_db()
        self.assertEqual(bot.state, BotStates.ENDED, "the recording was wiped when only one left")
        self.assertEqual(bot.viewer_user_ids, [CREATOR])
        self.assertTrue(self.sees(CREATOR, bot.object_id))
        self.assertFalse(self.sees(SHARER, bot.object_id))

    def test_the_one_who_left_can_no_longer_read_the_transcript(self):
        bot = self.shared_meeting()
        self.delete(bot.object_id, SHARER)

        response = self.get_meetings(member_token(SHARER), url=f"/api/v1/meetings/{bot.object_id}/transcript")

        self.assertEqual(response.status_code, 404)

    def test_the_other_still_reads_the_transcript_after_a_co_viewer_leaves(self):
        bot = self.shared_meeting()
        self.delete(bot.object_id, SHARER)

        response = self.get_meetings(member_token(CREATOR), url=f"/api/v1/meetings/{bot.object_id}/transcript")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([r["transcription"]["transcript"] for r in response.json()], ["shared words"])

    def test_the_last_viewer_leaving_wipes_the_data(self):
        bot = self.shared_meeting()
        self.delete(bot.object_id, SHARER)  # SHARER leaves; CREATOR is now the last viewer

        response = self.delete(bot.object_id, CREATOR)

        self.assertEqual(response.status_code, 204)
        bot.refresh_from_db()
        self.assertEqual(bot.state, BotStates.DATA_DELETED)
        self.assertEqual(bot.viewer_user_ids, [])

    def test_leaving_is_not_an_owner_privilege(self):
        # The CREATOR leaving a shared meeting only removes their own copy; the SHARER keeps it
        # and the data stays, exactly as when the SHARER leaves.
        bot = self.shared_meeting()

        response = self.delete(bot.object_id, CREATOR)

        self.assertEqual(response.status_code, 204)
        bot.refresh_from_db()
        self.assertEqual(bot.state, BotStates.ENDED)
        self.assertEqual(bot.viewer_user_ids, [SHARER])

    def test_a_solely_held_meeting_is_wiped_on_delete(self):
        bot = self.make_meeting(CREATOR, state=BotStates.ENDED)  # sole viewer

        response = self.delete(bot.object_id, CREATOR)

        self.assertEqual(response.status_code, 204)
        bot.refresh_from_db()
        self.assertEqual(bot.state, BotStates.DATA_DELETED)

    def test_the_last_viewer_cannot_wipe_a_running_meeting(self):
        bot = self.make_meeting(CREATOR, state=BotStates.JOINING)  # sole viewer, still live

        response = self.delete(bot.object_id, CREATOR)

        self.assertEqual(response.status_code, 409)
        self.assertIn("still in progress", response.json()["error"])
        bot.refresh_from_db()
        self.assertEqual(bot.viewer_user_ids, [CREATOR], "the last viewer was dropped from a live meeting")

    def test_a_co_viewer_may_leave_a_running_meeting(self):
        # Leaving never touches the recording, so a co-viewer walking away from a live meeting is
        # fine -- only wiping the last copy of a running meeting is refused.
        bot = self.make_meeting(CREATOR, state=BotStates.JOINING, viewer_user_ids=[CREATOR, SHARER])

        response = self.delete(bot.object_id, SHARER)

        self.assertEqual(response.status_code, 204)
        bot.refresh_from_db()
        self.assertEqual(bot.state, BotStates.JOINING)
        self.assertEqual(bot.viewer_user_ids, [CREATOR])

    def test_deleting_twice_is_idempotent(self):
        bot = self.make_meeting(CREATOR, state=BotStates.ENDED)

        self.assertEqual(self.delete(bot.object_id, CREATOR).status_code, 204)
        # Second delete: I am no longer a viewer, so it is indistinguishable from "not found".
        self.assertEqual(self.delete(bot.object_id, CREATOR).status_code, 404)

    def test_a_stranger_cannot_delete(self):
        bot = self.shared_meeting()

        response = self.delete(bot.object_id, "7777")

        self.assertEqual(response.status_code, 404)
        bot.refresh_from_db()
        self.assertCountEqual(bot.viewer_user_ids, [CREATOR, SHARER], "a stranger's delete changed the list")
