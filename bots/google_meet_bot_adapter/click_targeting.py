"""Pure geometry for picking where inside an element a humanized mouse click should land.

Aiming anywhere inside an element's full bounding box lets the mouse path end on its
outermost pixel, where ``document.elementFromPoint()`` resolves to a neighbouring element
(or the element's own border) and the attempt is thrown away. Real users click near the
middle of a control, so the target is inset toward its centre: more accurate *and* more
human than aiming at the edge.

The inset is capped in absolute pixels and floored on span, so a thin control -- Google
Meet's name field is 24px tall -- is not shrunk into a target that no recorded mouse path
can reach. Values were chosen against the real mocap library and real Meet button geometry:
with a flat 25% inset the name field drops from 435 usable paths to 49, while the capped
inset below keeps it at 283.
"""

# Fraction of an element's span to inset per side before capping.
CLICK_TARGET_INSET_RATIO = 0.15
# Never inset more than this per side, so wide elements keep a large target.
MAX_CLICK_TARGET_INSET_PX = 8.0
# Never shrink an axis below this, so tiny controls stay reachable.
MIN_CLICK_TARGET_SPAN_PX = 4.0


def axis_inset(span: float) -> float:
    """Inset to apply to each side of one axis of length ``span`` (CSS pixels)."""
    if span <= MIN_CLICK_TARGET_SPAN_PX:
        return 0.0
    largest_inset_that_keeps_min_span = (span - MIN_CLICK_TARGET_SPAN_PX) / 2.0
    return max(0.0, min(span * CLICK_TARGET_INSET_RATIO, MAX_CLICK_TARGET_INSET_PX, largest_inset_that_keeps_min_span))


def is_rect_centre(x: int, y: int, rect_left: int, rect_top: int, rect_right: int, rect_bottom: int) -> bool:
    """True when (x, y) is the exact centre of the rect.

    ``MocapManager._center_landing_fallback`` -- the last resort of the stretch/rotate search --
    always aims at this point, so an endpoint landing here means another search would test the
    identical spot. Mirrors that function's own rounding.
    """
    return (x, y) == (round((rect_left + rect_right) / 2.0), round((rect_top + rect_bottom) / 2.0))


def compute_click_target_rect(left: float, top: float, width: float, height: float, screen_x: float, screen_y: float, dpr: float) -> tuple[int, int, int, int]:
    """Map an element's CSS bounding box to the monitor-space rect the mouse should land in.

    Args mirror ``getBoundingClientRect()`` plus ``window.screenX/screenY`` and
    ``devicePixelRatio``. Returns ``(left, top, right, bottom)`` in device pixels.
    """
    inset_x = axis_inset(width)
    inset_y = axis_inset(height)

    css_left = left + inset_x
    css_right = left + width - inset_x
    css_top = top + inset_y
    css_bottom = top + height - inset_y

    return (
        int(round((screen_x + css_left) * dpr)),
        int(round((screen_y + css_top) * dpr)),
        int(round((screen_x + css_right) * dpr)),
        int(round((screen_y + css_bottom) * dpr)),
    )
