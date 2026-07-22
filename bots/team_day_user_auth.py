"""Verify the team.day member JWT that ties a local session to its owner.

Local-session calls carry two credentials: the project API key (``Authorization: Token``,
authorises the project) and the member's team.day JWT (``X-User-Token``, identifies *which*
user owns the session). This module verifies the latter.

Contract (confirmed against the auth-server that shares team.day's secret --
``apps/auth-server/src/auth``): HS256, signed with ``TEAM_DAY_JWT_SECRET``; payload
``{ id: <int>, email }`` where ``id`` is the team.day user id. Signature is required; ``exp``
is honoured when present but not required, matching the auth-server's own guard. The user id
is returned as a string (the desktop already stringifies the same id elsewhere).
"""

import jwt
from django.conf import settings
from rest_framework import exceptions

USER_TOKEN_HEADER = "HTTP_X_USER_TOKEN"  # WSGI form of the "X-User-Token" request header


class UserAuthNotConfigured(exceptions.APIException):
    status_code = 503
    default_detail = "User authentication is not configured on the server."
    default_code = "user_auth_not_configured"


def decode_user_id(request):
    """Return the verified team.day user id (string) for the request, or raise.

    Fails closed: a missing server secret is a 503 (never silently trust the caller); a
    missing/forged/expired token is a 401.
    """
    secret = settings.TEAM_DAY_JWT_SECRET
    if not secret:
        raise UserAuthNotConfigured()

    token = request.META.get(USER_TOKEN_HEADER)
    if not token:
        raise exceptions.AuthenticationFailed("Missing X-User-Token")

    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        # Wrong signature, malformed, or expired -- all surface as a generic 401.
        raise exceptions.AuthenticationFailed("Invalid or expired user token")

    user_id = payload.get("id", payload.get("sub"))
    if user_id is None:
        raise exceptions.AuthenticationFailed("User token missing a user id")

    return str(user_id)


def quiet_user_id(request):
    """The member id if one can be established, else None -- never raises.

    For throttling, which runs before the view's auth and must not turn a bad token into an
    error of its own. An unattributable request simply goes unthrottled here; the view still
    rejects it.
    """
    try:
        return optional_user_id(request)
    except Exception:
        return None


def optional_user_id(request):
    """Same as decode_user_id, but returns None when no token was sent at all.

    For endpoints that predate per-user ownership (bot dispatch): API customers who send no
    token keep working exactly as before and their bots stay unowned. A token that IS sent
    must still be valid -- silently ignoring a bad one would hand the caller an unowned bot
    that never appears in their history, which looks like data loss.
    """
    if not request.META.get(USER_TOKEN_HEADER):
        return None
    return decode_user_id(request)
