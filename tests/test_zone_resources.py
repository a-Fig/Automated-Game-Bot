import json
from dataclasses import replace

import pytest

from albot.behaviors import Farm, Fight, Pickup
from albot.config import Config
from albot.geometry import Point
from albot.navigation import Zone
from albot.resources import zone_resources
from albot.state import Detection
from fakes import PLAYER, World, pred


TYPES = ["hide", "fiber", "ore", "wood", "stone"]


def zone_config(tmp_path, name, metadata):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "zone.json").write_text(json.dumps(metadata), encoding="utf-8")
    cfg = Config()
    cfg.navigation.assets_folder = str(tmp_path)
    cfg.navigation.zone = name
    cfg.detection.farm_classes = list(TYPES)
    return cfg


@pytest.mark.parametrize("name,types", [("fractured_ground", ["hide", "fiber", "ore"]),
                                         ("vixen_tor", ["wood", "stone", "ore"])])
def test_scene_and_farm_picker_limit_resources_to_configured_zone(tmp_path, name, types, monkeypatch):
    cfg = zone_config(tmp_path, name, {"resources": types})
    monkeypatch.setattr(Zone, "load", lambda *args, **kwargs: pytest.fail("metadata must not load maps/learning"))
    w = World(cfg)
    scene = w.see([pred(kind, PLAYER[0]+100+i*20, PLAYER[1]) for i, kind in enumerate(TYPES)])
    assert {d.class_name for d in scene.farmables} == set(types)
    assert Farm().plan(w.ctx, scene, w.ctx.status()).target.class_name in types
    # A caller supplying its own unfiltered Scene must not bypass the picker.
    absent = next(kind for kind in TYPES if kind not in types)
    unfiltered = replace(scene, farmables=(Detection(absent, 0.99, Point(PLAYER[0]+10, PLAYER[1])),
                                           Detection(types[0], 0.9, Point(PLAYER[0]+100, PLAYER[1]))))
    assert Farm().plan(w.ctx, unfiltered, w.ctx.status()).target.class_name == types[0]
    assert Farm().plan(w.ctx, replace(unfiltered, farmables=unfiltered.farmables[:1]), w.ctx.status()) is None


@pytest.mark.parametrize("metadata", [{}, {"resources": []}])
def test_missing_list_preserves_classes_but_empty_list_disables_farming(tmp_path, metadata):
    w = World(zone_config(tmp_path, "test", metadata))
    scene = w.see([pred(kind, PLAYER[0]+100+i*20, PLAYER[1]) for i, kind in enumerate(TYPES)])
    assert {d.class_name for d in scene.farmables} == (set(TYPES) if "resources" not in metadata else set())
    assert (Farm().plan(w.ctx, scene, w.ctx.status()) is not None) == ("resources" not in metadata)


def test_missing_zone_metadata_does_not_add_a_farming_dependency_on_navigation_assets(tmp_path):
    cfg = Config()
    cfg.navigation.assets_folder = str(tmp_path)
    cfg.navigation.zone = "not_installed"
    cfg.detection.farm_classes = list(TYPES)
    w = World(cfg)
    scene = w.see([pred("wood", PLAYER[0]+100, PLAYER[1])])
    assert scene.farmables[0].class_name == "wood"
    assert Farm().plan(w.ctx, scene, w.ctx.status()) is not None


def test_resource_allowlist_does_not_filter_combat_or_pickups(tmp_path):
    w = World(zone_config(tmp_path, "test", {"resources": []}))
    scene = w.see([pred("mob", PLAYER[0]+100, PLAYER[1]), pred("silver", PLAYER[0]+120, PLAYER[1]),
                   pred("hide", PLAYER[0]+130, PLAYER[1])])
    assert not scene.farmables
    assert Fight("fight_close", True).plan(w.ctx, scene, w.ctx.status()).target.class_name == "mob"
    assert Pickup().plan(w.ctx, scene, w.ctx.status()).target.class_name == "silver"


def test_bundled_resources_are_a_filter_not_automatic_enablement():
    cfg = Config()
    assert zone_resources(cfg.navigation) == {"hide", "fiber", "ore"}
    w = World(cfg)
    scene = w.see([pred("ore", PLAYER[0]+100, PLAYER[1]), pred("hide", PLAYER[0]+130, PLAYER[1])])
    assert [d.class_name for d in scene.farmables] == ["hide"]
    assert Farm().plan(w.ctx, scene, w.ctx.status()).target.class_name == "hide"


def test_changing_configured_zone_uses_its_own_resource_list(tmp_path):
    cfg = zone_config(tmp_path, "first", {"resources": ["fiber"]})
    zone_config(tmp_path, "second", {"resources": ["wood"]})
    assert zone_resources(cfg.navigation) == {"fiber"}
    cfg.navigation.zone = "second"
    assert zone_resources(cfg.navigation) == {"wood"}


@pytest.mark.parametrize("types", [None, "hide", {"hide": True}, [42], [""]])
def test_invalid_resource_lists_report_a_metadata_error(tmp_path, types):
    cfg = zone_config(tmp_path, "test", {"resources": types})
    with pytest.raises(ValueError, match="zone resources must be a list"):
        World(cfg).see([])
