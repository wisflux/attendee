"""ElevenLabs batch speech-to-text (scribe_v2).

Extracted verbatim from ``bots/tasks/process_utterance_task.py``, which had grown past the
repo's file-size limit. Behaviour is unchanged by the move.
"""

import json
import logging

import requests

from bots.models import Credentials, TranscriptionFailureReasons
from bots.transcription_utils import get_mp3_for_utterance_group, split_transcription_by_utterance
from bots.utils import pcm_to_mp3

logger = logging.getLogger(__name__)

# The API rejects a longer list, so a caller that configures more gets the first hundred
# rather than a failed request.
MAX_KEYTERMS = 100
# Below this the model is unsure which language it heard, and the transcript is dropped.
LOW_LANGUAGE_CONFIDENCE = 0.5


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


def elevenlabs_api_key(recording):
    """The project's ElevenLabs key, or the failure that explains why there isn't one."""
    credentials_record = recording.bot.project.credentials.filter(credential_type=Credentials.CredentialTypes.ELEVENLABS).first()
    if not credentials_record:
        return None, {"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND}

    credentials = credentials_record.get_credentials()
    if not credentials:
        return None, {"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND}

    api_key = credentials.get("api_key")
    if not api_key:
        return None, {"reason": TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND, "error": "api_key not in credentials"}
    return api_key, None


def elevenlabs_form_data(transcription_settings):
    """The form fields for a request, identical whether it carries one utterance or a group."""
    data = {}
    if transcription_settings.elevenlabs_model_id():
        data["model_id"] = transcription_settings.elevenlabs_model_id()

    if transcription_settings.elevenlabs_language_code():
        data["language_code"] = transcription_settings.elevenlabs_language_code()

    data["tag_audio_events"] = transcription_settings.elevenlabs_tag_audio_events()

    keyterms = transcription_settings.elevenlabs_keyterms()
    if keyterms:
        # Multipart form values must be scalars, so the list travels as a JSON string.
        data["keyterms"] = json.dumps(keyterms[:MAX_KEYTERMS])
    return data


def get_transcription_via_elevenlabs(utterance):
    recording = utterance.recording
    transcription_settings = utterance.transcription_settings
    api_key, failure = elevenlabs_api_key(recording)
    if failure:
        return None, failure

    # Convert PCM audio to MP3 for ElevenLabs
    payload_mp3 = pcm_to_mp3(utterance.get_audio_blob().tobytes(), sample_rate=utterance.get_sample_rate())

    # Prepare the request for ElevenLabs speech-to-text API
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": api_key,
    }

    # Prepare multipart form data
    files = {"file": ("audio.mp3", payload_mp3, "audio/mpeg")}
    data = elevenlabs_form_data(transcription_settings)

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

        language_probability = result.get("language_probability", 0.0)
        if language_probability < LOW_LANGUAGE_CONFIDENCE:
            # Behaviour is unchanged -- the text is still dropped. It is recorded because this
            # rule discards real speech as readily as wrong-language output, and whether to
            # keep it should be settled by what it actually eats rather than by argument.
            # Metrics at INFO; the text itself only at DEBUG, so meeting content does not land
            # in production logs to satisfy a diagnostic.
            dropped_text = result.get("text", "")
            logger.info(f"ElevenLabs transcription skipped for utterance {utterance.id}: language confidence {language_probability:.2f} below {LOW_LANGUAGE_CONFIDENCE}, detected {result.get('language_code')}, {len(dropped_text)} characters dropped")
            logger.debug(f"Dropped text for utterance {utterance.id}: {dropped_text}")
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


def get_transcription_via_elevenlabs_for_utterance_group(utterances, gaps_seconds=None):
    """Transcribe a run of utterances as ONE request, then hand the words back to their rows.

    Sending each utterance alone is what makes the model re-identify the language every second or
    two, and guess an ending for a sentence it only saw half of. Here the clips are joined with
    the real pauses between them, sent once, and each returned word is placed on the row whose
    window it falls in.

    `gaps_seconds` must be the SAME list used to build the audio, which is why it is threaded
    through rather than recomputed -- see bots.transcription_utils.utterance_windows.

    Returns ({utterance_id: transcription}, None) or (None, failure_data).
    """
    if not utterances:
        return {}, None

    first_utterance = utterances[0]
    api_key, failure = elevenlabs_api_key(first_utterance.recording)
    if failure:
        return None, failure

    identifier = f"utterances {[utterance.id for utterance in utterances]}"
    try:
        payload_mp3 = get_mp3_for_utterance_group(utterances, sample_rate=first_utterance.get_sample_rate(), gaps_seconds=gaps_seconds)
    except (ValueError, RuntimeError) as error:
        logger.error(f"ElevenLabs could not assemble audio for {identifier}: {error}")
        return None, {"reason": TranscriptionFailureReasons.INTERNAL_ERROR, "error": str(error)}

    try:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            files={"file": ("audio.mp3", payload_mp3, "audio/mpeg")},
            data=elevenlabs_form_data(first_utterance.transcription_settings),
        )

        if response.status_code == 401:
            logger.warning(f"ElevenLabs returned 401 for {identifier}: {str(response.text)[:300]}")
            failure_data = {"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "status_code": 401, "provider": "elevenlabs", "transient_auth": True}
            error_detail = elevenlabs_error_detail(response)
            if error_detail:
                failure_data["detail"] = error_detail
            return None, failure_data

        if response.status_code == 429:
            return None, {"reason": TranscriptionFailureReasons.RATE_LIMIT_EXCEEDED, "status_code": response.status_code}

        if response.status_code != 200:
            logger.error(f"ElevenLabs group transcription failed with status code {response.status_code}: {response.text}")
            return None, {"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "status_code": response.status_code, "response_text": response.text}

        result = response.json()

        # DELIBERATELY NOT DROPPED HERE. The single-utterance path discards a transcript scoring
        # below LOW_LANGUAGE_CONFIDENCE, because a one-second clip is too short to identify a
        # language from and the output is usually wrong-language noise. A group carries half a
        # minute of speech, so that reasoning does not hold -- and discarding it would throw away
        # real speech from every row in the group at once. Recorded so the rule can be judged on
        # what it would have eaten.
        language_probability = result.get("language_probability", 0.0)
        if language_probability < LOW_LANGUAGE_CONFIDENCE:
            logger.info(f"ElevenLabs group {identifier}: language confidence {language_probability:.2f} below {LOW_LANGUAGE_CONFIDENCE}, detected {result.get('language_code')} -- kept, a group is long enough to trust")

        words = [{"word": word.get("text"), "start": word.get("start"), "end": word.get("end")} for word in result.get("words", [])]
        transcription = {"transcript": result.get("text", ""), "words": words, "language": result.get("language_code", None)}
        logger.info(f"ElevenLabs group transcription completed for {identifier}")

        return split_transcription_by_utterance(transcription, utterances, gaps_seconds=gaps_seconds), None

    except requests.exceptions.RequestException as e:
        logger.error(f"ElevenLabs group transcription request failed: {str(e)}")
        return None, {"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "error": str(e)}
    except json.JSONDecodeError as e:
        logger.error(f"ElevenLabs group transcription response parsing failed: {str(e)}")
        return None, {"reason": TranscriptionFailureReasons.TRANSCRIPTION_REQUEST_FAILED, "error": f"Invalid JSON response: {str(e)}"}
    except Exception as e:
        logger.error(f"ElevenLabs group transcription unexpected error: {str(e)}")
        return None, {"reason": TranscriptionFailureReasons.INTERNAL_ERROR, "error": str(e)}
