"""ElevenLabs batch speech-to-text (scribe_v2).

Extracted verbatim from ``bots/tasks/process_utterance_task.py``, which had grown past the
repo's file-size limit. Behaviour is unchanged by the move.
"""

import json
import logging

import requests

from bots.models import Credentials, TranscriptionFailureReasons
from bots.utils import pcm_to_mp3

logger = logging.getLogger(__name__)


def elevenlabs_error_detail(response):
    # ElevenLabs error bodies look like {"detail": {"status": "<code>", "message": "..."}}. The
    # status code (e.g. quota_exceeded, detected_unusual_activity, invalid_api_key) is what tells
    # an exhausted account apart from a genuinely bad key, so we preserve it in failure_data for
    # downstream surfacing (bot-end event metadata -> webhooks -> user-facing notifications).
    try:
        detail = response.json().get("detail", {})
        if isinstance(detail, dict):
            extracted = {key: detail.get(key) for key in ("status", "message") if detail.get(key)}
            return extracted or None
    except Exception:
        pass
    return None


def get_transcription_via_elevenlabs(utterance):
    recording = utterance.recording
    transcription_settings = utterance.transcription_settings
    elevenlabs_credentials_record = recording.bot.project.credentials.filter(credential_type=Credentials.CredentialTypes.ELEVENLABS).first()
    if not elevenlabs_credentials_record:
        return None, {"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND}

    elevenlabs_credentials = elevenlabs_credentials_record.get_credentials()
    if not elevenlabs_credentials:
        return None, {"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND}

    api_key = elevenlabs_credentials.get("api_key")
    if not api_key:
        return None, {"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND, "error": "api_key not in credentials"}

    # Convert PCM audio to MP3 for ElevenLabs
    payload_mp3 = pcm_to_mp3(utterance.get_audio_blob().tobytes(), sample_rate=utterance.get_sample_rate())

    # Prepare the request for ElevenLabs speech-to-text API
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": api_key,
    }

    # Prepare multipart form data
    files = {"file": ("audio.mp3", payload_mp3, "audio/mpeg")}

    # Add model_id if configured
    data = {}
    if transcription_settings.elevenlabs_model_id():
        data["model_id"] = transcription_settings.elevenlabs_model_id()

    if transcription_settings.elevenlabs_language_code():
        data["language_code"] = transcription_settings.elevenlabs_language_code()

    data["tag_audio_events"] = transcription_settings.elevenlabs_tag_audio_events()

    try:
        response = requests.post(url, headers=headers, files=files, data=data if data else None)

        if response.status_code == 401:
            logger.warning(f"ElevenLabs returned 401 for utterance {utterance.id}: {str(response.text)[:300]}")
            # ElevenLabs (especially the free tier) intermittently returns 401 even for valid keys;
            # the identical request usually succeeds on retry. Treat the first few 401s as a
            # retryable failure so a transient blip doesn't permanently fail the utterance, and
            # only declare the credentials invalid once the retries are exhausted.
            if utterance.transcription_attempt_count < 3:
                return None, {"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "status_code": 401, "provider": "elevenlabs", "transient_auth": True}
            failure_data = {"reason": TranscriptionFailureReasons.CREDENTIALS_INVALID}
            error_detail = elevenlabs_error_detail(response)
            if error_detail:
                failure_data["detail"] = error_detail
            return None, failure_data

        if response.status_code == 429:
            return None, {"reason": TranscriptionFailureReasons.RATE_LIMIT_EXCEEDED, "status_code": response.status_code}

        if response.status_code != 200:
            logger.error(f"ElevenLabs transcription failed with status code {response.status_code}: {response.text}")
            return None, {"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "status_code": response.status_code, "response_text": response.text}

        result = response.json()
        logger.info("ElevenLabs transcription completed successfully")

        if result.get("language_probability", 0.0) < 0.5:
            logger.info(f"ElevenLabs transcription skipped for utterance {utterance.id} because the language probability was less than 0.5")
            return {"transcript": "", "words": []}, None

        # Extract transcript and words from the response
        transcript_text = result.get("text", "")
        words = list(map(lambda word: {"word": word.get("text"), "start": word.get("start"), "end": word.get("end")}, result.get("words", [])))

        # Format the response to match our expected schema
        transcription = {"transcript": transcript_text, "words": words, "language": result.get("language_code", None)}

        return transcription, None

    except requests.exceptions.RequestException as e:
        logger.error(f"ElevenLabs transcription request failed: {str(e)}")
        return None, {"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "error": str(e)}
    except json.JSONDecodeError as e:
        logger.error(f"ElevenLabs transcription response parsing failed: {str(e)}")
        return None, {"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "error": f"Invalid JSON response: {str(e)}"}
    except Exception as e:
        logger.error(f"ElevenLabs transcription unexpected error: {str(e)}")
        return None, {"reason": TranscriptionFailureReasons.INTERNAL_ERROR, "error": str(e)}
