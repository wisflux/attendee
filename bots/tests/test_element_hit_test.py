from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from selenium.common.exceptions import StaleElementReferenceException

from bots.google_meet_bot_adapter.element_hit_test import (
    COVERED,
    HIT,
    HIT_TEST_JS,
    MAX_STALE_RELOCATE_ATTEMPTS,
    STALE,
    ElementReplacedError,
    classify_hit_test_result,
    relocation_is_enabled,
)
from bots.google_meet_bot_adapter.google_meet_ui_methods import GoogleMeetUIMethods


class TestClassifyHitTestResult(SimpleTestCase):
    def test_passes_through_the_three_known_codes(self):
        self.assertEqual(classify_hit_test_result("hit"), (HIT, None))
        self.assertEqual(classify_hit_test_result("stale"), (STALE, None))
        self.assertEqual(classify_hit_test_result("covered"), (COVERED, None))

    def test_treats_the_legacy_boolean_true_as_a_hit(self):
        # An older injected script, or a mixed deploy, must not be read as a miss.
        self.assertEqual(classify_hit_test_result(True), (HIT, None))

    def test_falls_back_to_covered_for_anything_unrecognised(self):
        # COVERED preserves the original behaviour: try another mouse path.
        for raw in (False, None, "", "something-else", 0, 1, [], {}):
            self.assertEqual(classify_hit_test_result(raw), (COVERED, None), f"unexpected classification for {raw!r}")


class TestHitTestJs(SimpleTestCase):
    def test_checks_is_connected_before_comparing(self):
        # Order matters: a detached node can still sit under elementFromPoint's answer, so
        # staleness must be decided first or it is misreported as merely covered.
        self.assertLess(HIT_TEST_JS.index("isConnected"), HIT_TEST_JS.index("elementFromPoint"))

    def test_returns_the_codes_the_classifier_knows(self):
        for code in (HIT, STALE, COVERED):
            self.assertIn(f"'{code}'", HIT_TEST_JS)

    def test_reports_the_blocking_element(self):
        # The whole point of the diagnostic: a log must be able to say WHAT was on top.
        self.assertIn("blocker", HIT_TEST_JS)
        self.assertIn("elementFromPoint", HIT_TEST_JS)


class TestClassifierReadsTheBlocker(SimpleTestCase):
    def test_extracts_the_blocker_from_a_covered_result(self):
        raw = {"result": "covered", "blocker": "div.uW2Fw-IE5DDf [1920x1080 fixed z=-1]"}
        self.assertEqual(classify_hit_test_result(raw), (COVERED, "div.uW2Fw-IE5DDf [1920x1080 fixed z=-1]"))

    def test_hit_and_stale_carry_no_blocker(self):
        self.assertEqual(classify_hit_test_result({"result": "hit", "blocker": None}), (HIT, None))
        self.assertEqual(classify_hit_test_result({"result": "stale", "blocker": None}), (STALE, None))

    def test_an_unknown_result_in_a_dict_still_keeps_the_blocker(self):
        raw = {"result": "something-new", "blocker": "div.mystery"}
        self.assertEqual(classify_hit_test_result(raw), (COVERED, "div.mystery"))


class TestHumanizedClickRelocatesReplacedElements(SimpleTestCase):
    """The wrapper's contract: re-fetch a replaced element instead of failing the whole join."""

    def _ui(self, attempt_side_effect):
        ui = GoogleMeetUIMethods()
        ui._humanized_click_attempt = MagicMock(side_effect=attempt_side_effect)
        return ui

    def test_clicks_once_when_nothing_was_replaced(self):
        ui = self._ui([None])
        ui.humanized_navigate_to_and_click_element("element")
        self.assertEqual(ui._humanized_click_attempt.call_count, 1)

    def test_refetches_and_retries_when_the_element_was_replaced(self):
        ui = self._ui([ElementReplacedError("replaced"), None])
        relocate = MagicMock(return_value="fresh-element")

        ui.humanized_navigate_to_and_click_element("stale-element", relocate=relocate)

        self.assertEqual(relocate.call_count, 1)
        self.assertEqual(ui._humanized_click_attempt.call_count, 2)
        # The second attempt must use the element the relocate returned, not the stale one.
        self.assertEqual(ui._humanized_click_attempt.call_args_list[1].args[0], "fresh-element")

    def test_gives_up_after_the_relocate_budget_so_a_rerendering_page_cannot_hang_the_join(self):
        ui = self._ui(ElementReplacedError("replaced"))
        relocate = MagicMock(return_value="fresh-element")

        with self.assertRaises(StaleElementReferenceException):
            ui.humanized_navigate_to_and_click_element("stale-element", relocate=relocate)

        self.assertEqual(ui._humanized_click_attempt.call_count, MAX_STALE_RELOCATE_ATTEMPTS)

    def test_raises_stale_when_no_relocate_was_supplied(self):
        # Callers that were not updated keep today's behaviour rather than silently looping.
        ui = self._ui([ElementReplacedError("replaced")])

        with self.assertRaises(StaleElementReferenceException):
            ui.humanized_navigate_to_and_click_element("stale-element")

        self.assertEqual(ui._humanized_click_attempt.call_count, 1)

    def test_lets_a_failing_relocate_propagate(self):
        # If the element is genuinely gone, that is the real error and must not be swallowed.
        ui = self._ui(ElementReplacedError("replaced"))
        relocate = MagicMock(side_effect=TimeoutError("element really is gone"))

        with self.assertRaises(TimeoutError):
            ui.humanized_navigate_to_and_click_element("stale-element", relocate=relocate)


class TestRelocationKillSwitch(SimpleTestCase):
    """One config value must be able to restore the previous behaviour with no code change."""

    def test_enabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(relocation_is_enabled())

    def test_disabled_only_by_an_explicit_false(self):
        for value, expected in (("false", False), ("FALSE", False), (" False ", False), ("true", True), ("", True), ("anything", True)):
            with patch.dict("os.environ", {"MEET_RELOCATE_REPLACED_ELEMENTS": value}):
                self.assertEqual(relocation_is_enabled(), expected, f"for {value!r}")


class TestRelocationFallbacks(SimpleTestCase):
    def _ui(self, attempt_side_effect):
        ui = GoogleMeetUIMethods()
        ui._humanized_click_attempt = MagicMock(side_effect=attempt_side_effect)
        return ui

    def test_kill_switch_restores_the_old_behaviour(self):
        ui = self._ui(ElementReplacedError("replaced"))
        relocate = MagicMock(return_value="fresh-element")

        with patch.dict("os.environ", {"MEET_RELOCATE_REPLACED_ELEMENTS": "false"}):
            with self.assertRaises(StaleElementReferenceException):
                ui.humanized_navigate_to_and_click_element("stale-element", relocate=relocate)

        relocate.assert_not_called()
        self.assertEqual(ui._humanized_click_attempt.call_count, 1)

    def test_a_relocate_that_returns_nothing_fails_loudly(self):
        # Never carry on with None and produce a confusing failure further down.
        ui = self._ui(ElementReplacedError("replaced"))
        relocate = MagicMock(return_value=None)

        with self.assertRaises(StaleElementReferenceException):
            ui.humanized_navigate_to_and_click_element("stale-element", relocate=relocate)

        self.assertEqual(ui._humanized_click_attempt.call_count, 1)


class TestHitTestJsDegradesSafely(SimpleTestCase):
    def test_missing_is_connected_is_not_treated_as_stale(self):
        # An engine without Node.isConnected must fall through to the original comparison rather
        # than reporting every element as replaced.
        self.assertIn("isConnected === false", HIT_TEST_JS)
