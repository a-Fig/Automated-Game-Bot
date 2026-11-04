from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from albot.actuator import Input, FocusLost
from albot.behaviors import Farm, Fight, Pickup, Wander
from albot.config import Config, load
from albot.geometry import Point
from albot.hud import MinimapRecovery, hud_reason, minimap_panel_state
from albot.safety import SafetyStop
from albot.state import Status
from albot.vision import build_scene
from albot.window import FixedWindow
from fakes import World, pred


def panel(name):
    return cv2.imread(str(Path(__file__).parent / f"fixtures/minimap/{name}.png"))


def frame(name, scale=1):
    image = np.full((1400, 2240, 3), 80, np.uint8)
    image[1030:1400, 1820:2240] = panel(name)
    image[56:68, 105:245] = (25, 30, 180)
    image[72:80, 105:245] = (180, 30, 25)
    return cv2.resize(image, (round(2240*scale), round(1400*scale))) if scale != 1 else image


def reading(image, at, health=1):
    return Status(at, (), True, False, health_fraction=health, frame=image)


def world():
    w = World()
    w.window = w.ctx.window = FixedWindow(2240, 1400)
    w.workers.scenes.publish(build_scene(w.clock.now(), (2240, 1400),
                                       [pred("player", 1125, 560)], [], w.config))
    w.status(frame=frame("visible"))
    return w


@pytest.mark.parametrize("scale", [1, 0.5])
def test_all_hud_rectangles_drop_detections_and_refuse_click_walk_and_aim(scale, caplog):
    w = world()
    size = (round(2240*scale), round(1400*scale))
    w.window = w.ctx.window = FixedWindow(*size)
    points = [(2100, 1200), (2100, 1380), (200, 100), (1800, 40), (2200, 150), (1100, 1350)]
    with caplog.at_level("INFO"):
        for x, y in points:
            p = Point(x*scale, y*scale)
            scene = build_scene(w.clock.now(), size,
                                [pred(name, *p) for name in ("player", "mob", "red", "hide", "silver")], [], w.config)
            assert not scene.detections and not scene.player_detected
            for behavior in (Farm(), Fight("fight_close", True), Pickup()):
                assert behavior.plan(w.ctx, scene, w.ctx.status()) is None
            assert not w.ctx.movement_allowed(p)
            for send in (w.ctx.click, w.ctx.walk_to, w.ctx.move_to):
                with pytest.raises(SafetyStop):
                    send(p)
    assert not w.input.actions
    assert sum("world click/aim refused" in r.message for r in caplog.records) == len(points)*3
    # Rectangles are half-open; the world immediately outside remains available.
    assert not hud_reason(Point(1819*scale, 1029*scale), size, w.config.safety)


def test_wander_retries_a_hud_direction_instead_of_clicking_it():
    w = world()
    w.config.behavior.wander_range = 1
    # First due south lands in the ability bar; east is free world space.
    choices = iter((np.pi/2, 790/1400, 0, 300/2240))
    w.ctx.random = SimpleNamespace(uniform=lambda *args: next(choices))
    point = Wander.pick_point(w.ctx, w.ctx.scene(), attempts=2)
    assert point == Point(1425, 560)
    assert not hud_reason(point, (2240, 1400), w.config.safety)


@pytest.mark.parametrize("method", ["click", "walk_to", "move_to"])
def test_input_itself_refuses_hud_before_importing_mouse_libraries(method, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyautogui", SimpleNamespace(
        moveTo=lambda *args, **kwargs: pytest.fail("HUD refusal must not move mouse")))
    with pytest.raises(SafetyStop):
        getattr(Input(FixedWindow(2240, 1400), Config()), method)(Point(2197, 1290))


def test_recorded_minimap_frame_visibility_survives_darkening_and_arrow_occlusion():
    visible, hidden = panel("visible"), panel("hidden")
    assert visible.shape == hidden.shape == (370, 420, 3)
    assert minimap_panel_state(visible) == "visible"
    assert minimap_panel_state(hidden) == "hidden"
    assert minimap_panel_state((visible*0.6).astype(np.uint8)) == "visible"
    assert minimap_panel_state((visible*0.25).astype(np.uint8)) == "visible"
    assert minimap_panel_state((hidden*0.6).astype(np.uint8)) == "hidden"
    visible[132:152, 260:282] = (100, 120, 160)  # obscure arrow only
    recovery = MinimapRecovery(Config())
    assert recovery.locator.locate(visible, 0) is None
    image = frame("visible")
    image[1030:1400, 1820:2240] = visible
    for at in range(10):
        assert not recovery.observe(reading(image, at), at)
    assert recovery.attempts == 0 and not recovery.reason
    for value in (0, 20, 80):
        assert minimap_panel_state(np.full(visible.shape, value, np.uint8)) == "unknown"


def test_recorded_tab_overlay_keeps_visible_map_from_toggling_without_an_arrow_fix():
    dimmed = panel("visible_tab_dimmed")
    assert minimap_panel_state(dimmed) == "visible"
    recovery = MinimapRecovery(Config())
    assert recovery.locator.locate(dimmed, 0) is None
    image = frame("visible")
    image[1030:1400, 1820:2240] = dimmed
    for at in range(10):
        assert not recovery.observe(reading(image, at), at)


@pytest.mark.parametrize("scale", [1, 0.5])
def test_hidden_map_requires_fresh_readable_health_and_five_seconds_without_a_fix(scale):
    r = MinimapRecovery(Config())
    image = frame("hidden", scale)
    for at in range(5):
        # Exercise PIL RGB too: same recovery result as numpy BGR.
        assert not r.observe(reading(Image.fromarray(image[:, :, ::-1]), at), at)
    assert not r.observe(reading(image, 4), 8)  # old image cannot confirm anything
    assert not r.observe(reading(image, 9, None), 9)
    for at in range(10, 15):
        assert not r.observe(reading(image, at), at)
    assert r.observe(reading(image, 15), 15)
    assert r.attempts == 0  # input must acknowledge the proposal


def test_recovery_rate_limit_and_three_failed_attempts_latch_stop():
    r = MinimapRecovery(Config())
    image = frame("hidden")
    toggles = []
    for at in range(131):
        if r.observe(reading(image, at), at):
            r.toggled(at)
            toggles.append(at)
    assert toggles == [5, 65, 125]
    assert r.reason == "minimap hidden after 3 failed toggles"
    assert not r.observe(reading(frame("visible"), 131), 131)
    assert r.reason  # no auto-resume on a later restored frame


def test_a_real_fix_resets_failures_but_does_not_bypass_toggle_cooldown():
    r = MinimapRecovery(Config())
    hidden, visible = frame("hidden"), frame("visible")
    for at in range(6):
        proposed = r.observe(reading(hidden, at), at)
    assert proposed
    r.toggled(5)
    assert not r.observe(reading(visible, 6), 6)
    assert r.attempts == 0
    for at in range(7, 65):
        assert not r.observe(reading(hidden, at), at)
    assert r.observe(reading(hidden, 65), 65)


def test_recovery_can_be_disabled_and_unknown_panels_never_propose_a_toggle():
    cfg = Config()
    cfg.safety.minimap_recovery = False
    r = MinimapRecovery(cfg)
    for at in range(10):
        assert not r.observe(reading(frame("hidden"), at), at)
    cfg.safety.minimap_recovery = True
    image = frame("hidden")
    image[1030:1400, 1820:2240] = 0
    for at in range(10, 20):
        assert not r.observe(reading(image, at), at)


def test_uncertain_toggle_dispatch_does_not_retry_immediately():
    w = world()
    w.status(frame=frame("hidden"))
    calls = []

    def failed_input():
        calls.append(w.clock.now())
        raise FocusLost()

    w.input.toggle_minimap = failed_input
    for at in range(20):
        w.clock.t = 1000+at
        w.status()
        try:
            w.ctx.check_safety()
        except FocusLost:
            pass
    assert calls == [1005]
    assert w.ctx.minimap_recovery.attempts == 1


def test_context_only_toggles_named_ui_action_then_stops_after_failed_retries(caplog):
    w = world()
    w.status(frame=frame("hidden"))
    with caplog.at_level("INFO"):
        for at in range(131):
            w.clock.t = 1000+at
            w.status()
            try:
                w.ctx.check_safety()
            except SafetyStop:
                pass
    assert [a for a in w.input.actions if a[0] == "minimap_toggle"] == [("minimap_toggle",)]*3
    assert not any(a[0] in ("click", "walk", "move") for a in w.input.actions)
    assert "minimap hidden; toggled" in caplog.text
    assert w.ctx.quit.is_set() and w.input.actions[-1] == ("release",)
    w.status(frame=frame("visible"))
    before = list(w.input.actions)
    with pytest.raises(SafetyStop):
        w.ctx.click(Point(1200, 500))
    assert w.input.actions == before


def test_no_toggle_during_focus_loss_death_loading_or_unreadable_health():
    for kind in ("focus", "death", "loading", "health"):
        w = world()
        image = frame("hidden") if kind != "loading" else np.zeros((1400, 2240, 3), np.uint8)
        if kind == "focus":
            w.window.active = False
        if kind == "loading":
            w.config.safety.zone_guard = True
        for at in range(10):
            w.clock.t = 1000+at
            w.status(frame=image, health=0 if kind == "death" else None if kind == "health" else 1)
            try:
                w.ctx.check_safety()
            except SafetyStop:
                pass
        assert not any(a[0] == "minimap_toggle" for a in w.input.actions)


def test_toggle_is_scaled_focus_checked_and_bypasses_only_the_hud_click_guard(monkeypatch):
    import sys
    actions = []
    monkeypatch.setitem(sys.modules, "keyboard", SimpleNamespace(press_and_release=lambda *args: actions.append(args)))
    monkeypatch.setitem(sys.modules, "pyautogui", SimpleNamespace(
        moveTo=lambda x, y, **kwargs: actions.append((x, y)), click=lambda: actions.append(("click",)),
        FailSafeException=RuntimeError))
    window = FixedWindow(1120, 700)
    inp = Input(window, Config())
    inp.toggle_minimap()
    assert actions == [(1098, 645), ("click",)]
    with pytest.raises(SafetyStop):
        inp.click(Point(1098, 645))
    window.active = False
    with pytest.raises(FocusLost):
        inp.toggle_minimap()
    assert actions == [(1098, 645), ("click",)]


def test_hud_and_recovery_config_load(tmp_path):
    path = tmp_path / "hud.toml"
    path.write_text('[safety]\nhud_size=[1120,700]\nhud_rects=[[1,2,3,4]]\n'
                    'minimap_panel_rect=[910,515,1120,700]\nminimap_toggle=[1098,645]\n'
                    'minimap_missing_seconds=7\nminimap_toggle_interval=90\nminimap_toggle_attempts=2\n')
    cfg = load(str(path))
    assert cfg.safety.hud_rects == [(1, 2, 3, 4)]
    assert cfg.safety.minimap_panel_rect == (910, 515, 1120, 700)
    assert cfg.safety.minimap_toggle == (1098, 645)
    assert cfg.safety.minimap_missing_seconds == 7 and cfg.safety.minimap_toggle_attempts == 2
