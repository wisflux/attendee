from django.test import SimpleTestCase

from bots.google_meet_bot_adapter.click_targeting import (
    MIN_CLICK_TARGET_SPAN_PX,
    axis_inset,
    compute_click_target_rect,
    is_rect_centre,
)
from bots.google_meet_bot_adapter.mocap_manager import MocapManager

# Element geometry and mouse positions copied from production bot logs, so the
# regression tests below exercise the rects the bot actually aims at.
NAME_INPUT_RECT = (1214, 476, 1482, 500)
MICROPHONE_BUTTON_RECT = (666, 682, 722, 730)
CAMERA_BUTTON_RECT = (738, 682, 794, 730)
NAME_INPUT_MOUSE_AT = (965, 545)
MICROPHONE_BUTTON_MOUSE_AT = (1223, 489)
CAMERA_BUTTON_MOUSE_AT = (702, 722)

# A dead end starves the join attempt, so every real target must keep a healthy pool.
MIN_ACCEPTABLE_PATHS = 20

# Bot.recording_dimensions() defaults to HD_1080P, so the bot loads join_mocap_1080p_*.json.
# The rects above are 1080p geometry; testing against the 720p library measured the wrong pool.
MEET_VIDEO_FRAME_SIZES = ((1920, 1080), (1280, 720))


class TestAxisInset(SimpleTestCase):
    def test_insets_a_fraction_of_a_mid_sized_span(self):
        # 40px * 0.15 = 6px, below the absolute cap, so the ratio applies as-is.
        self.assertAlmostEqual(axis_inset(40.0), 6.0)

    def test_caps_the_inset_on_a_wide_span(self):
        # Without the cap a 268px name field would lose 40px per side.
        self.assertEqual(axis_inset(268.0), 8.0)

    def test_never_shrinks_a_span_below_the_minimum(self):
        # At 5px the min-span clamp binds: the ratio alone would leave only 3.5px of target.
        self.assertEqual(axis_inset(5.0), 0.5)
        self.assertEqual(5.0 - 2 * axis_inset(5.0), 4.0)

    def test_ratio_binds_once_it_is_tighter_than_the_clamp(self):
        # At 6px the ratio (0.9px) is already tighter than the clamp (1.0px), so it wins.
        self.assertAlmostEqual(axis_inset(6.0), 0.9)

    def test_returns_zero_for_a_span_at_or_below_the_minimum(self):
        self.assertEqual(axis_inset(4.0), 0.0)
        self.assertEqual(axis_inset(1.0), 0.0)

    def test_is_never_negative(self):
        for span in (0.0, 0.5, 1.0, 4.0, 4.1, 10.0, 1000.0):
            self.assertGreaterEqual(axis_inset(span), 0.0, f"negative inset for span {span}")


class TestComputeClickTargetRect(SimpleTestCase):
    def test_target_is_strictly_inside_the_element_box(self):
        left, top, width, height = 100.0, 200.0, 56.0, 48.0
        rect_left, rect_top, rect_right, rect_bottom = compute_click_target_rect(left, top, width, height, 0.0, 0.0, 1.0)

        self.assertGreater(rect_left, left)
        self.assertGreater(rect_top, top)
        self.assertLess(rect_right, left + width)
        self.assertLess(rect_bottom, top + height)

    def test_rect_is_never_inverted(self):
        # Squeeze every axis, including spans small enough to trip a naive inset.
        for width in (1.0, 3.0, 4.0, 5.0, 10.0, 56.0, 268.0):
            for height in (1.0, 3.0, 4.0, 5.0, 24.0, 48.0):
                rect_left, rect_top, rect_right, rect_bottom = compute_click_target_rect(0.0, 0.0, width, height, 0.0, 0.0, 1.0)
                self.assertLessEqual(rect_left, rect_right, f"inverted x for {width}x{height}")
                self.assertLessEqual(rect_top, rect_bottom, f"inverted y for {width}x{height}")

    def test_applies_window_offset_and_device_pixel_ratio(self):
        # With no inset possible on either axis the maths must reduce to the original
        # (screen_offset + css) * dpr conversion this replaced.
        span = MIN_CLICK_TARGET_SPAN_PX
        rect_left, rect_top, rect_right, rect_bottom = compute_click_target_rect(10.0, 20.0, span, span, 100.0, 200.0, 2.0)

        self.assertEqual(rect_left, round((100.0 + 10.0) * 2.0))
        self.assertEqual(rect_top, round((200.0 + 20.0) * 2.0))
        self.assertEqual(rect_right, round((100.0 + 10.0 + span) * 2.0))
        self.assertEqual(rect_bottom, round((200.0 + 20.0 + span) * 2.0))

    def test_thin_element_keeps_a_usable_height(self):
        # Google Meet's name field is 24px tall; a flat ratio would gut it.
        _, rect_top, _, rect_bottom = compute_click_target_rect(0.0, 0.0, 268.0, 24.0, 0.0, 0.0, 1.0)
        self.assertGreaterEqual(rect_bottom - rect_top, MIN_CLICK_TARGET_SPAN_PX)

    def test_inset_is_applied_in_css_pixels_before_scaling(self):
        # The inset must be applied to CSS pixels and the whole rect then scaled by dpr.
        # Applying it after scaling (or dividing it by dpr) still yields a valid rect, so
        # only exact expected values catch it. 56x48 at dpr=2 -> inset 8.0 and 7.2 CSS px.
        rect = compute_click_target_rect(left=100.0, top=200.0, width=56.0, height=48.0, screen_x=0.0, screen_y=0.0, dpr=2.0)
        self.assertEqual(rect, (216, 414, 296, 482))

    def test_returns_integers(self):
        for value in compute_click_target_rect(10.5, 20.5, 56.3, 48.7, 0.0, 0.0, 1.25):
            self.assertIsInstance(value, int)


class TestRealMeetTargetsKeepEnoughMocapPaths(SimpleTestCase):
    """The inset shrinks the target, so prove the recorded paths can still reach it.

    Starving this search is the failure mode the inset could plausibly introduce: fewer
    landing spots means more dead ends, which is what the fix exists to remove.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mocap_managers = {size: MocapManager(video_frame_size=size) for size in MEET_VIDEO_FRAME_SIZES}

    def _paths_landing_in(self, mocap_manager, mouse_at, rect):
        mouse_x, mouse_y = mouse_at
        rect_left, rect_top, rect_right, rect_bottom = rect
        return sum(1 for sequence in mocap_manager.sequences if rect_left <= mouse_x + sequence.total_dx <= rect_right and rect_top <= mouse_y + sequence.total_dy <= rect_bottom)

    def _assert_target_reachable(self, mouse_at, rect):
        for size, mocap_manager in self.mocap_managers.items():
            with self.subTest(video_frame_size=size):
                self.assertGreaterEqual(self._paths_landing_in(mocap_manager, mouse_at, self._inset_rect(rect)), MIN_ACCEPTABLE_PATHS)

    def _inset_rect(self, rect):
        rect_left, rect_top, rect_right, rect_bottom = rect
        return compute_click_target_rect(rect_left, rect_top, rect_right - rect_left, rect_bottom - rect_top, 0.0, 0.0, 1.0)

    def test_name_input_target_keeps_enough_paths(self):
        self._assert_target_reachable(NAME_INPUT_MOUSE_AT, NAME_INPUT_RECT)

    def test_microphone_button_target_keeps_enough_paths(self):
        self._assert_target_reachable(MICROPHONE_BUTTON_MOUSE_AT, MICROPHONE_BUTTON_RECT)

    def test_camera_button_target_keeps_enough_paths(self):
        self._assert_target_reachable(CAMERA_BUTTON_MOUSE_AT, CAMERA_BUTTON_RECT)


class TestIsRectCentre(SimpleTestCase):
    """The mocap manager's last-resort path always aims at the rect centre.

    Landing there means a further stretched search would test the identical point, so the
    caller stops instead of spending its budget. Rounding must match
    MocapManager._center_landing_fallback exactly or the check silently never fires.
    """

    def test_detects_the_exact_centre_of_an_even_rect(self):
        self.assertTrue(is_rect_centre(20, 30, 10, 20, 30, 40))

    def test_detects_the_rounded_centre_of_an_odd_rect(self):
        # (10+15)/2 = 12.5 -> banker's rounding gives 12, matching _center_landing_fallback.
        self.assertTrue(is_rect_centre(round(12.5), round(22.5), 10, 20, 15, 25))

    def test_rejects_a_point_one_pixel_off_centre(self):
        self.assertFalse(is_rect_centre(21, 30, 10, 20, 30, 40))
        self.assertFalse(is_rect_centre(20, 31, 10, 20, 30, 40))

    def test_rejects_a_point_elsewhere_inside_the_rect(self):
        self.assertFalse(is_rect_centre(12, 22, 10, 20, 30, 40))
