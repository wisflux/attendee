"""Deciding whether a humanized mouse path actually landed on its target element.

The check used to be a bare boolean, which conflated two very different failures:

* the element is still on the page but something else sits at that point -- try another path;
* the element has been **replaced**, so the handle we hold points at a detached node -- no path
  can ever succeed, because the comparison is against a node that is no longer in the document.

Google Meet's pre-join screen re-renders while the humanized mouse journey is in flight (device
lists finish loading, the sign-in tooltip appears), swapping the name field for an identical new
node. Reported as a plain "no path worked", that cost a full browser restart on 72% of Meet joins.
Distinguishing the two lets the caller simply re-fetch the element and carry on.
"""

import os

# Result codes returned by HIT_TEST_JS. Kept as plain strings so the JS stays readable.
HIT = "hit"
STALE = "stale"
COVERED = "covered"

# How many times to re-fetch a replaced element before giving up and letting the caller retry the
# whole join. A page that re-renders continuously would otherwise burn the caller's much larger
# attempt budget at roughly a second and a half per mouse journey.
MAX_STALE_RELOCATE_ATTEMPTS = 3

# `isConnected` is checked FIRST: a detached node can still sit under elementFromPoint's answer,
# so the hit comparison alone cannot tell staleness from being covered.
HIT_TEST_JS = r"""
    var expected = arguments[2];
    if (expected && expected.isConnected === false) { return {result: 'stale', blocker: null}; }
    var el = document.elementFromPoint(arguments[0], arguments[1]);
    if (el && (el === expected || expected.contains(el))) { return {result: 'hit', blocker: null}; }
    var describe = function (n) {
        if (!n) { return 'null'; }
        var cls = (typeof n.className === 'string' && n.className.trim()) ? '.' + n.className.trim().split(/\s+/).slice(0, 2).join('.') : '';
        var box = n.getBoundingClientRect();
        var style = window.getComputedStyle(n);
        return n.tagName.toLowerCase() + cls
            + ' [' + Math.round(box.width) + 'x' + Math.round(box.height)
            + ' ' + style.position + ' z=' + style.zIndex + ']';
    };
    return {result: 'covered', blocker: describe(el)};
"""


class ElementReplacedError(Exception):
    """The handle we hold points at a node that is no longer in the document.

    Raised instead of reporting "no mouse path worked", because no path can ever succeed against
    a detached node -- the caller must re-fetch the element rather than retry the journey.
    """


def relocation_is_enabled() -> bool:
    """Kill switch for the re-fetch behaviour.

    Set ``MEET_RELOCATE_REPLACED_ELEMENTS=false`` in the bot pod's config to fall straight back to
    the previous behaviour without a code change, if re-fetching ever misbehaves in production.
    """
    return os.getenv("MEET_RELOCATE_REPLACED_ELEMENTS", "true").strip().lower() != "false"


def classify_hit_test_result(raw_result) -> tuple[str, str | None]:
    """Map the raw value from HIT_TEST_JS to ``(result, blocker_description)``.

    `blocker_description` names whatever `elementFromPoint` returned instead of the target, so a
    production log says *what* is on top rather than only that the aim missed. It is None unless
    the result is COVERED.

    Anything unrecognised -- including the legacy boolean or a plain string from an older injected
    script, or None from a driver hiccup -- is treated as COVERED, which preserves the original
    behaviour of trying another mouse path.
    """
    if raw_result is True:
        return HIT, None
    if isinstance(raw_result, dict):
        result = raw_result.get("result")
        if result in (HIT, STALE, COVERED):
            return result, raw_result.get("blocker")
        return COVERED, raw_result.get("blocker")
    if raw_result in (HIT, STALE, COVERED):
        return raw_result, None
    return COVERED, None
