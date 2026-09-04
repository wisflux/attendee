"""LocalAudioInputManager consulting a verdict cache before asking the detector live.

Drives the real manager, same as test_local_audio_processing.py -- what's pinned here is that
supplying `cached_verdicts` changes WHERE a verdict comes from, and nothing else: the manager's
own decisions (when a buffer opens, when it flushes, what gets emitted) must be identical to the
uncached path given the same underlying verdicts.
"""

from django.test import TestCase

from bots.local_audio_processing import LocalAudioInputManager
from bots.local_audio_processing import build_manager as build_production_manager
from bots.models import Bot, Organization, Participant, Project, Recording, RecordingStates
from bots.tests.test_local_audio_processing import (
    SPEAKER,
    build_manager,
    feed,
    silent_frame,
    speech_frame,
    timeline,
    verdicts_for,
)


class CountingDetector:
    """Wraps another detector, counting how many times it was actually asked."""

    def __init__(self, verdicts):
        self._verdicts = verdicts
        self._index = 0
        self.calls = 0

    def is_speech(self, chunk_bytes):
        self.calls += 1
        verdict = self._verdicts[min(self._index, len(self._verdicts) - 1)]
        self._index += 1
        return verdict

    def export_state(self):
        return {"scripted": True}


class VerdictCacheConsultedFirstTest(TestCase):
    """The whole point: a cached verdict must never reach the detector at all."""

    def test_cached_frames_do_not_call_the_detector(self):
        detector = CountingDetector([True] * 100000)
        manager = LocalAudioInputManager(
            detector=detector,
            cached_verdicts=[True, True, True],
            save_audio_chunk_callback=lambda message: None,
            get_participant_callback=lambda _: {"participant_uuid": SPEAKER, "participant_full_name": "You"},
            sample_rate=16000,
            utterance_size_limit=16000 * 2 * 30,
            silence_duration_limit=1.5,
            should_print_diagnostic_info=False,
        )
        sections = ((30, speech_frame),)  # 3 frames, all covered by the cache
        feed(manager, timeline(*sections))

        self.assertEqual(detector.calls, 0)

    def test_frames_past_the_cache_do_call_the_detector(self):
        detector = CountingDetector([True] * 100000)
        manager = LocalAudioInputManager(
            detector=detector,
            cached_verdicts=[True],  # only the first frame is cached
            save_audio_chunk_callback=lambda message: None,
            get_participant_callback=lambda _: {"participant_uuid": SPEAKER, "participant_full_name": "You"},
            sample_rate=16000,
            utterance_size_limit=16000 * 2 * 30,
            silence_duration_limit=1.5,
            should_print_diagnostic_info=False,
        )
        sections = ((30, speech_frame),)  # 3 frames: 1 cached, 2 genuinely new
        feed(manager, timeline(*sections))

        self.assertEqual(detector.calls, 2)

    def test_with_no_cache_every_frame_calls_the_detector(self):
        """The default: nothing changes for a manager that was never given a cache."""
        detector = CountingDetector([True] * 100000)
        manager = LocalAudioInputManager(
            detector=detector,
            save_audio_chunk_callback=lambda message: None,
            get_participant_callback=lambda _: {"participant_uuid": SPEAKER, "participant_full_name": "You"},
            sample_rate=16000,
            utterance_size_limit=16000 * 2 * 30,
            silence_duration_limit=1.5,
            should_print_diagnostic_info=False,
        )
        sections = ((30, speech_frame),)
        feed(manager, timeline(*sections))

        self.assertEqual(detector.calls, 3)


class BehaviouralEquivalenceTest(TestCase):
    """Caching must change nothing about WHAT the manager decides, only where it asks."""

    def test_a_cached_and_an_uncached_run_over_the_same_verdicts_emit_the_same_thing(self):
        sections = ((300, speech_frame), (2000, silent_frame))
        script = verdicts_for(*sections)
        frames = timeline(*sections)

        uncached_emitted = []
        feed(build_manager(uncached_emitted, verdicts=script), frames)

        cached_emitted = []
        # Half the script pre-supplied as a cache, the rest still scripted live.
        split = len(script) // 2
        feed(build_manager(cached_emitted, verdicts=script[split:], cached_verdicts=script[:split]), frames)

        self.assertEqual(len(uncached_emitted), 1)
        self.assertEqual(len(cached_emitted), 1)
        self.assertEqual(uncached_emitted[0]["audio_data"], cached_emitted[0]["audio_data"])
        self.assertEqual(uncached_emitted[0]["voice_ms"], cached_emitted[0]["voice_ms"])


class BufferedVerdictsTest(TestCase):
    """What a drain needs to carry forward for whatever is still open, unflushed."""

    def test_an_open_utterances_verdicts_are_available_before_it_flushes(self):
        emitted = []
        sections = ((300, speech_frame),)  # still well under the 1.5s silence limit -- stays open
        manager = build_manager(emitted, verdicts=verdicts_for(*sections))
        feed(manager, timeline(*sections))

        self.assertEqual(emitted, [], "precondition: nothing has flushed yet")
        self.assertEqual(manager.buffered_verdicts(SPEAKER), [True] * 30)

    def test_a_closed_source_has_no_buffered_verdicts(self):
        emitted = []
        manager = build_manager(emitted)

        self.assertEqual(manager.buffered_verdicts(SPEAKER), [])


class BuildManagerThreadsCachedVerdictsTest(TestCase):
    """The production wiring: build_manager must actually pass the cache through."""

    def test_build_manager_accepts_and_uses_cached_verdicts(self):
        org = Organization.objects.create(name="Org")
        project = Project.objects.create(name="Proj", organization=org)
        bot = Bot.objects.create(project=project, meeting_url="local_recording")
        recording = Recording.objects.create(bot=bot, recording_type=1, transcription_type=1, state=RecordingStates.IN_PROGRESS)
        participant = Participant.objects.create(bot=bot, uuid=SPEAKER)

        manager = build_production_manager(recording, participant, 16000, cached_verdicts=[True, True, True])
        manager.vad = CountingDetector([True] * 100000)

        sections = ((30, speech_frame),)
        feed(manager, timeline(*sections))

        self.assertEqual(manager.vad.calls, 0, "all 3 frames were cached; the detector must not run")
