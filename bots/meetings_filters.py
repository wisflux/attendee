"""Query filters for the member history list.

Kept out of the view so each filter is a small function that can be read on its own, and so
the view stays "resolve the member, filter, paginate".

Every filter here **narrows** an already member-scoped queryset. That ordering is the safety
property: ownership is applied first, and no ``AND`` can widen a result set, so no filter in
this module can reach another member's meetings however wrong its arguments are.

One deliberate difference from the project dashboard (``projects_views.py``), which runs the
same search expression: a malformed date there is swallowed and the filter silently dropped.
On a web page that is a shrug; on an API it returns *everything* while looking like a
successful search, so here it is a 400.
"""

from datetime import datetime, time

from django.db.models import Q
from django.utils.dateparse import parse_date, parse_datetime
from django.utils.timezone import is_naive, make_aware

from .models import SessionTypes

SOURCE_TO_SESSION_TYPE = {"bot": SessionTypes.BOT, "local": SessionTypes.LOCAL}

# Longer than the columns being searched (name is 255, and a link that long matches nothing),
# so the cap can only reject terms that could never have found anything -- while keeping a
# caller from handing the database a megabyte-long LIKE pattern.
MAX_SEARCH_LENGTH = 2048


class FilterError(ValueError):
    """A query parameter the caller got wrong. The message is safe to return verbatim."""


def parse_moment(raw, param):
    """A date (YYYY-MM-DD) or an ISO 8601 datetime, as an aware datetime.

    A bare date means the start of that day, so the two bounds compose into whole days:
    ``created_after=2026-07-01&created_before=2026-07-02`` is exactly the 1st.

    Django's parsers have two failure modes, and only one of them is a return value: they give
    None for a string they cannot read ("yesterday"), but *raise* ValueError for one that is
    well formed and impossible ("2026-13-45"). Both are the caller's mistake, so both become a
    400 -- letting the second escape would turn a typo into a 500.
    """
    try:
        moment = parse_datetime(raw)
        if moment is None:
            day = parse_date(raw)
            moment = datetime.combine(day, time.min) if day else None
    except ValueError:
        moment = None

    if moment is None:
        raise FilterError(f"{param} must be a date (YYYY-MM-DD) or an ISO 8601 datetime")
    return make_aware(moment) if is_naive(moment) else moment


def apply_source(queryset, source):
    """Bot meetings or local recordings. Both live in one table, told apart by session_type."""
    if not source:
        return queryset
    session_type = SOURCE_TO_SESSION_TYPE.get(source)
    if session_type is None:
        raise FilterError(f"source must be one of {sorted(SOURCE_TO_SESSION_TYPE)}")
    return queryset.filter(session_type=session_type)


def apply_search(queryset, term):
    """Substring match on what a person can actually see in the list: the name and the link.

    Deliberately NOT the transcript: searching what was said needs a full-text index and has
    different speed and ranking characteristics, so it belongs behind its own mode rather than
    silently making this box slow.
    """
    term = (term or "").strip()
    if not term:
        return queryset
    # A NUL survives Django's query parsing but Postgres refuses to accept it in a string
    # literal, so passing one through raises ValueError deep in the driver -- a 500 for what is
    # plainly a bad request. Rejected here rather than stripped: silently searching for
    # something other than what was asked for is how a filter comes to return everything.
    if "\x00" in term:
        raise FilterError("q must not contain null bytes")
    if len(term) > MAX_SEARCH_LENGTH:
        raise FilterError(f"q must be at most {MAX_SEARCH_LENGTH} characters")
    return queryset.filter(Q(name__icontains=term) | Q(meeting_url__icontains=term))


def apply_created_range(queryset, after, before):
    """``created_after`` is inclusive, ``created_before`` exclusive -- a half-open range.

    Half-open is what makes adjacent ranges tile without overlapping or dropping the instant
    on the boundary, and it matches how the desktop already computes its day bounds.
    """
    if after:
        queryset = queryset.filter(created_at__gte=parse_moment(after, "created_after"))
    if before:
        queryset = queryset.filter(created_at__lt=parse_moment(before, "created_before"))
    return queryset


def apply_meeting_filters(queryset, params):
    """Every list filter, in one call. Raises FilterError for anything the caller got wrong."""
    queryset = apply_source(queryset, params.get("source"))
    queryset = apply_search(queryset, params.get("q"))
    return apply_created_range(queryset, params.get("created_after"), params.get("created_before"))
