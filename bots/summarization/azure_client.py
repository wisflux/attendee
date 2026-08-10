"""Azure OpenAI client for meeting summarization.

Targets the Azure **Responses API** (`/openai/responses`): the deployment in use is a gpt-5.x
reasoning model, which does not accept chat/completions, `temperature`, or `response_format`.
Split so the brain is testable without a network — `build_messages` and `parse_response` are pure;
`summarize` is the thin HTTP edge. See docs/meeting-summaries.
"""

import json

import requests

from bots.models import DEFAULT_AZURE_OPENAI_API_VERSION

from .prompts import SYSTEM_PROMPT, build_user_message

# Max length of the notes the model may write (~12k words / ~24 pages at 16k). A reasoning model
# also spends hidden reasoning tokens from this same budget; the real ceiling is the context window.
SUMMARY_MAX_TOKENS = 16000
# Hard cap on the title regardless of what the model returns (requirement: at most 20 words).
TITLE_MAX_WORDS = 20
# Generous: a long meeting's notes plus reasoning can take a while, and this runs in a background job.
REQUEST_TIMEOUT_SECS = 120
_FENCE = "```"


class SummaryTruncated(Exception):
    """The model hit max_output_tokens before finishing — the caller retries with a concise instruction."""


def build_messages(utterances, context=None):
    """The chat messages for one summarization call (pure). Passed as the Responses API ``input``."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(utterances, context)},
    ]


def _strip_code_fences(text):
    """Remove a surrounding ```/```json fence if the model wrapped its reply in one.

    Only strips a well-formed multi-line fence (opening line + body + closing fence). Anything else
    is returned untouched, so a malformed/one-line fence can never blank the body.
    """
    text = (text or "").strip()
    if not text.startswith(_FENCE):
        return text
    newline = text.find("\n")
    if newline == -1 or not text.rstrip().endswith(_FENCE):
        return text
    body = text[newline + 1 :].rstrip()
    return body[: -len(_FENCE)].strip()


def _clean_title(raw_title):
    """A single clean line of at most TITLE_MAX_WORDS words, or None if there is nothing usable."""
    if not raw_title:
        return None
    title = " ".join(str(raw_title).replace("#", " ").split())
    if not title:
        return None
    words = title.split(" ")
    if len(words) > TITLE_MAX_WORDS:
        title = " ".join(words[:TITLE_MAX_WORDS])
    return title


def parse_response(raw_content):
    """Extract ``{'title', 'notes'}`` from the model reply, tolerating imperfect output.

    Falls back to treating the whole reply as ``notes`` with ``title=None`` so a good summary is
    never lost to a formatting slip — the caller keeps the existing meeting name when title is None.
    """
    content = _strip_code_fences(raw_content or "")
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        return {"title": _clean_title(data.get("title")), "notes": (data.get("notes") or "").strip()}
    return {"title": None, "notes": content}


def _endpoint_url(credentials):
    endpoint = credentials["endpoint"].rstrip("/")
    api_version = credentials.get("api_version") or DEFAULT_AZURE_OPENAI_API_VERSION
    return f"{endpoint}/openai/responses?api-version={api_version}"


def _extract_output_text(response_json):
    """The assistant's text from a Responses API reply, or '' if there is no message item.

    The ``output`` array holds reasoning items and a message item; we want the message's output_text.
    An empty result (content-filtered / reasoning only) is treated as a failed generation upstream.
    """
    for item in response_json.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text") or ""
    return ""


def summarize(credentials, messages, max_output_tokens=SUMMARY_MAX_TOKENS, timeout=REQUEST_TIMEOUT_SECS):
    """Call the Azure Responses API and return the raw assistant text (I/O).

    ``credentials`` is the dashboard-stored dict ``{endpoint, deployment, api_key, api_version}``.
    Raises ``SummaryTruncated`` when the reply was cut off at ``max_output_tokens`` (caller retries
    concisely) and ``requests.HTTPError`` on a non-2xx response (caller classifies the reason).
    """
    payload = {"model": credentials["deployment"], "input": messages, "max_output_tokens": max_output_tokens}
    headers = {"api-key": credentials["api_key"], "Content-Type": "application/json"}
    response = requests.post(_endpoint_url(credentials), headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if data.get("status") == "incomplete" and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
        raise SummaryTruncated()
    return _extract_output_text(data)
