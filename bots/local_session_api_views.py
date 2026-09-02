"""API for desktop local recordings (``Bot.session_type == LOCAL``).

A local session reuses the whole bot pipeline -- ``Bot -> Recording -> Utterance`` --
but launches no bot pod. Instead the desktop uploads its own mic/system audio, which is
segmented and transcribed by the same VAD + ``process_utterance`` -> ElevenLabs path
that meeting bots already use.
"""

import logging

from django.db import transaction
from django.db.models import Value
from django.db.models.functions import Coalesce
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .authentication import ApiKeyAuthentication
from .bots_api_views import TranscriptView
from .local_audio_processing import BYTES_PER_SAMPLE
from .local_session_store import MIC_SOURCE, SYSTEM_SOURCE, enqueue_segment
from .models import (
    Bot,
    BotStates,
    Participant,
    Recording,
    RecordingFormats,
    RecordingManager,
    SessionTypes,
    TranscriptionProviders,
    TranscriptionTypes,
)
from .tasks.process_local_audio_segment_task import drain_local_session_audio, finalize_local_session
from .team_day_user_auth import decode_user_id
from .throttling import ProjectPostThrottle

logger = logging.getLogger(__name__)

# Bot.meeting_url is non-nullable and a local recording has no meeting, so it carries a
# sentinel -- exactly the way app sessions use "app_session".
LOCAL_SESSION_MEETING_URL = "local_recording"
LOCAL_SESSION_NAME = "Local Recording"

# A local recording tags every chunk with one of two sources. Labels are OPTIONAL: an
# in-person recording on a single mic has nothing to distinguish, so it sends no speaker_name
# and gets no label. An online meeting sets speaker_name (e.g. "You") for the mic; the system
# track is then "Others". The source ids live in the store module (shared with the finalizer).
MIC_PARTICIPANT_UUID = MIC_SOURCE
SYSTEM_PARTICIPANT_UUID = SYSTEM_SOURCE
SYSTEM_PARTICIPANT_NAME = "Others"

# webrtcvad only understands these rates, and the desktop should be sending 16kHz anyway.
SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 48000)
# A 1-2s segment is ~64KB at 16kHz. The cap keeps a malformed or hostile upload from
# turning into a giant celery/Redis payload.
MAX_SEGMENT_BYTES = 1_000_000


def build_local_session_settings(language_code):
    """Settings for an audio-only recording transcribed by ElevenLabs (defaults to scribe_v2)."""
    # Explicitly off rather than left to the API default: a local recording is a room with
    # keyboards and chairs in it, and every tagged event lands in the transcript as if
    # somebody had said it.
    elevenlabs_settings = {"tag_audio_events": False}
    if language_code:
        elevenlabs_settings["language_code"] = language_code

    return {
        # Drives Bot.recording_type() -> AUDIO_ONLY. The format defaults to MP4 (video) otherwise.
        "recording_settings": {"format": RecordingFormats.MP3},
        # The presence of the "elevenlabs" key is what selects the transcription provider.
        "transcription_settings": {"elevenlabs": elevenlabs_settings},
    }


class LocalSessionCreateView(APIView):
    authentication_classes = [ApiKeyAuthentication]
    throttle_classes = [ProjectPostThrottle]

    def post(self, request):
        # Stamped from the verified token only -- never from the body -- so a session can only
        # ever be owned by the user who created it.
        owner_user_id = decode_user_id(request)

        # Blank by default -> no speaker label (single-mic in-person). The desktop passes
        # "You" only when it also captures system audio and wants the You/Others split.
        speaker_name = request.data.get("speaker_name") or ""
        language_code = request.data.get("language_code")
        metadata = request.data.get("metadata") or None

        with transaction.atomic():
            bot = Bot.objects.create(
                project=request.auth.project,
                settings=build_local_session_settings(language_code),
                metadata=metadata,
                state=BotStates.READY,
                meeting_url=LOCAL_SESSION_MEETING_URL,
                name=LOCAL_SESSION_NAME,
                session_type=SessionTypes.LOCAL,
                owner_user_id=owner_user_id,
                # The creator is the first viewer, set inline because a local session is never
                # deduplicated -- decode_user_id guarantees a real owner here, so there is no
                # unowned case to guard.
                viewer_user_ids=[owner_user_id],
            )

            recording = Recording.objects.create(
                bot=bot,
                recording_type=bot.recording_type(),
                transcription_type=TranscriptionTypes.NON_REALTIME,
                transcription_provider=TranscriptionProviders.ELEVENLABS,
                is_default_recording=True,
            )
            # Utterances refuse to transcribe unless the recording has been started.
            RecordingManager.set_recording_in_progress(recording)

            Participant.objects.create(bot=bot, uuid=MIC_PARTICIPANT_UUID, full_name=speaker_name, is_host=True)
            Participant.objects.create(bot=bot, uuid=SYSTEM_PARTICIPANT_UUID, full_name=SYSTEM_PARTICIPANT_NAME)

        logger.info(f"Created local session {bot.object_id} for project {request.auth.project.object_id}")
        return Response({"id": bot.object_id}, status=status.HTTP_201_CREATED)


def find_local_session(request, object_id, owner_user_id):
    """A local session is reachable only through an API key belonging to its project AND by its
    owner. Filtering on the owner (rather than checking it after) means a mismatch is
    indistinguishable from a missing session -- a 404 never confirms another user's session
    exists."""
    return Bot.objects.filter(
        object_id=object_id,
        project=request.auth.project,
        session_type=SessionTypes.LOCAL,
        owner_user_id=owner_user_id,
    ).first()


def mark_session_alive(bot_id):
    """Stamp the session's heartbeat so the idle reaper can tell a live (or paused) session
    from a crashed one. A single UPDATE -- no read-modify-write -- so concurrent mic and
    system uploads never contend on the row's concurrency version."""
    now_ts = int(timezone.now().timestamp())
    Bot.objects.filter(id=bot_id).update(
        first_heartbeat_timestamp=Coalesce("first_heartbeat_timestamp", Value(now_ts)),
        last_heartbeat_timestamp=now_ts,
    )


class LocalSessionAudioView(APIView):
    authentication_classes = [ApiKeyAuthentication]
    throttle_classes = [ProjectPostThrottle]

    def post(self, request, object_id):
        owner_user_id = decode_user_id(request)
        bot = find_local_session(request, object_id, owner_user_id)
        if bot is None:
            return Response({"error": "Local session not found"}, status=status.HTTP_404_NOT_FOUND)

        source = request.data.get("source")
        if source not in (MIC_PARTICIPANT_UUID, SYSTEM_PARTICIPANT_UUID):
            return Response(
                {"error": f"source must be '{MIC_PARTICIPANT_UUID}' or '{SYSTEM_PARTICIPANT_UUID}'"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            sample_rate = int(request.data["sample_rate"])
            # Milliseconds since the session started -- never a wall clock, so the server's
            # timeline cannot be corrupted by the desktop's clock being off.
            offset_ms = int(request.data["offset_ms"])
            # Monotonic per source: lets the drain order segments and drop replayed uploads.
            sequence = int(request.data["sequence"])
        except (KeyError, TypeError, ValueError):
            return Response(
                {"error": "sample_rate, offset_ms and sequence are required integers"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            return Response(
                {"error": f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if offset_ms < 0 or sequence < 0:
            return Response({"error": "offset_ms and sequence must be >= 0"}, status=status.HTTP_400_BAD_REQUEST)

        audio_file = request.FILES.get("file")
        audio = audio_file.read() if audio_file else b""
        if not audio:
            return Response({"error": "an audio 'file' of 16-bit PCM is required"}, status=status.HTTP_400_BAD_REQUEST)
        if len(audio) % BYTES_PER_SAMPLE:
            # An odd length is a truncated upload; it would raise deep inside the VAD.
            return Response({"error": "audio length must be a whole number of 16-bit samples"}, status=status.HTTP_400_BAD_REQUEST)
        if len(audio) > MAX_SEGMENT_BYTES:
            return Response(
                {"error": f"segment too large; send at most {MAX_SEGMENT_BYTES} bytes per upload"},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        # Queue the audio, then ask for a drain. Segmenting and transcription happen
        # off-request so the desktop can keep streaming without ever waiting on us.
        enqueue_segment(bot.id, source, sequence, audio, sample_rate, offset_ms)
        drain_local_session_audio.delay(bot.id, source)
        # An upload is proof the desktop is alive, so it doubles as a heartbeat.
        mark_session_alive(bot.id)
        return Response(status=status.HTTP_202_ACCEPTED)


class LocalSessionStopView(APIView):
    authentication_classes = [ApiKeyAuthentication]
    throttle_classes = [ProjectPostThrottle]

    def post(self, request, object_id):
        owner_user_id = decode_user_id(request)
        bot = find_local_session(request, object_id, owner_user_id)
        if bot is None:
            return Response({"error": "Local session not found"}, status=status.HTTP_404_NOT_FOUND)

        # Nothing more is coming, so each source drains what it has and emits whatever is
        # still mid-utterance -- otherwise the final sentence would sit in the tail forever.
        # One finalize task flushes every source's last utterance and then marks the
        # recording complete -- once, after all of them, so completion can't race a
        # still-uncreated final utterance.
        finalize_local_session.delay(bot.id)

        logger.info(f"Stopped local session {bot.object_id}")
        return Response({"id": bot.object_id}, status=status.HTTP_200_OK)


class LocalSessionTranscriptView(TranscriptView):
    """Owner-gated live transcript. Reuses the bot transcript logic unchanged, but first
    confirms the caller's token owns this local session so one user can't poll another's."""

    def get(self, request, object_id):
        owner_user_id = decode_user_id(request)
        if find_local_session(request, object_id, owner_user_id) is None:
            return Response({"error": "Local session not found"}, status=status.HTTP_404_NOT_FOUND)
        return super().get(request, object_id)
