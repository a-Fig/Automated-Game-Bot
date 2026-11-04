"""
Tests for the logic that can run without the game, a GPU, or Windows.
Run with: python -m pytest tests
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# obstacle_instance_detection imports objectFinder (which loads the models) and
# window_info (which needs the game window). Stub both before importing it.
_fake_window = types.ModuleType("window_info")
_fake_window.windowLocation = lambda: (0, 0)
sys.modules.setdefault("window_info", _fake_window)
sys.modules.setdefault("objectFinder", types.ModuleType("objectFinder"))

import obstacle_instance_detection as OID  # noqa: E402
from spell_slots import SpellSlotMapper  # noqa: E402


class Pred:
    def __init__(self, x, class_name="up", confidence=0.9):
        self.x = x
        self.y = 0
        self.class_name = class_name
        self.confidence = confidence


SLOT_X = [100, 150, 200, 250, 300, 350]


def calibrated_mapper():
    m = SpellSlotMapper()
    m.map([Pred(x) for x in SLOT_X])
    return m


def test_uncalibrated_mapper_treats_every_slot_as_cooldown():
    m = SpellSlotMapper()
    result = m.map([Pred(150), Pred(250)])
    assert [s.class_name for s in result] == ["cooldown"] * 6
    assert not m.is_calibrated()


def test_full_frame_calibrates_and_returns_left_to_right():
    m = SpellSlotMapper()
    result = m.map([Pred(x) for x in reversed(SLOT_X)])
    assert [p.x for p in result] == SLOT_X
    assert m.is_calibrated()


def test_missed_middle_slot_does_not_shift_the_others():
    # the model misses slot 1 (W). Slot 4 (D, the shield) must still be slot 4
    m = calibrated_mapper()
    preds = [Pred(102), Pred(198), Pred(251), Pred(349, "cooldown"), Pred(301, "up")]
    result = m.map(preds)
    assert result[1].class_name == "cooldown"  # the missed slot
    assert result[4].x == 301 and result[4].class_name == "up"
    assert result[5].class_name == "cooldown" and result[5].x == 349


def test_duplicate_boxes_on_one_slot_keep_the_more_confident():
    m = calibrated_mapper()
    result = m.map([Pred(200, "cooldown", 0.4), Pred(203, "up", 0.8)])
    assert result[2].class_name == "up"


def P(x, y):
    return OID.Point(x, y)


def square():
    # a 10 by 10 square from (10, 10) to (20, 20)
    return OID.Obstacle(0.9, [P(10, 10), P(20, 10), P(20, 20), P(10, 20)])


def test_line_through_obstacle_intersects():
    assert square().does_intersect(OID.Line(P(0, 15), P(30, 15)))


def test_line_missing_obstacle_does_not_intersect():
    assert not square().does_intersect(OID.Line(P(0, 0), P(30, 0)))


def test_line_through_the_closing_edge_intersects():
    # this only crosses the edge from the last point (10, 20) back to the first
    # (10, 10). before the fix, that edge was never checked
    assert square().does_intersect(OID.Line(P(0, 15), P(12, 15)))
