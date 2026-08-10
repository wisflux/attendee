"""Pure decision helpers for meeting-summary generation (classify failures, build the retry).

No I/O — unit-tested directly. The Celery task in bots/tasks/ orchestrates these around the DB and
the Azure call. See docs/meeting-summaries.
"""

import requests

# HTTP statuses worth retrying (Azure busy / transient), vs. config errors that a retry won't fix.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})

_CONCISE_INSTRUCTION = "Your previous answer was too long and was cut off. Produce the SAME notes document, but more concise — keep every key detail, decision, action item, and open question, just tighter — and stay within the length limit. Return the same JSON {title, notes} shape."


def _status_of(exc):
    """The HTTP status on a requests error, or None for a connection/timeout/other error."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def is_transient(exc):
    """True when the error is worth retrying (Azure busy, or a network blip), not a config problem."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    return _status_of(exc) in TRANSIENT_STATUSES


def classify_failure(exc):
    """A short, safe, member-facing reason for a failed generation (never leaks the key or a trace)."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return "Could not reach Azure OpenAI — try again shortly."
    status = _status_of(exc)
    if status in (401, 403):
        return "Azure authentication failed — check the API key in Credentials."
    if status == 404:
        return "Azure deployment or API version not found — check the deployment name and API version."
    if status == 400:
        return "Azure rejected the request — check the deployment name and API version."
    if status == 429:
        return "Azure rate limit reached — try again shortly."
    if status in TRANSIENT_STATUSES:
        return "Azure had a temporary error — try again shortly."
    if status is not None:
        return f"Azure returned an error (HTTP {status})."
    return "Summary generation failed."


def concise_retry_messages(messages):
    """The original messages plus a follow-up asking for a shorter answer (used after truncation)."""
    return list(messages) + [{"role": "user", "content": _CONCISE_INSTRUCTION}]
