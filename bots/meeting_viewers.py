"""Atomic maintenance of a meeting's ``viewer_user_ids`` list.

Kept apart from the views because the append has to be one statement, not a read-then-write: a
member and a dedup of that same member (or two devices) can dispatch at once, and a
read-modify-write would drop one of the appends. Each function is a single guarded UPDATE, so
Postgres serialises them and the result is order-independent and idempotent.
"""

from django.contrib.postgres.fields import ArrayField
from django.db.models import CharField, F, Func, Value

from .models import Bot


def add_viewer(bot_id, user_id):
    """Add ``user_id`` to the meeting's viewer list unless it is already there.

    One statement: the ``exclude(...__contains)`` guard means an already-present id appends
    nothing, so calling this on every dispatch (new bot or dedup) can never duplicate a viewer.
    """
    if not user_id:
        return
    Bot.objects.filter(id=bot_id).exclude(viewer_user_ids__contains=[user_id]).update(
        viewer_user_ids=Func(
            F("viewer_user_ids"),
            Value(user_id),
            function="array_append",
            output_field=ArrayField(CharField(max_length=255)),
        )
    )


def remove_viewer(bot_id, user_id):
    """Drop ``user_id`` from the meeting's viewer list. One statement, idempotent: ``array_remove``
    strips every occurrence (the append guard keeps it to at most one), so removing an id that is
    already gone is a harmless no-op."""
    if not user_id:
        return
    Bot.objects.filter(id=bot_id).update(
        viewer_user_ids=Func(
            F("viewer_user_ids"),
            Value(user_id),
            function="array_remove",
            output_field=ArrayField(CharField(max_length=255)),
        )
    )
