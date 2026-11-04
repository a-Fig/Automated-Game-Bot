import json
from dataclasses import replace

import pytest

from albot.behaviors import Farm, Fight, Pickup
from albot.config import Config
from albot.geometry import Point
from albot.navigation import Zone
from albot.resources import zone_resource_exclusions
from albot.state import Detection
from fakes import PLAYER, World, pred

TYPES = ["hide", "fiber", "ore", "wood", "stone"]


def resource(name, tiers=None):
    return {"type": name, "tiers": tiers, "role": None}


def zone_config(tmp_path, observations, name="test"):
    (tmp_path / "resources.json").write_text(json.dumps({"version": 1, "zones": {
        name: {"observations": observations}}}), encoding="utf-8")
    cfg = Config()
    cfg.navigation.assets_folder = str(tmp_path)
    cfg.navigation.zone = name
    cfg.detection.farm_classes = list(TYPES)
    return cfg


def test_fractured_ground_filters_only_types_absent_from_t2plus_not_current_hide_fiber(monkeypatch):
    cfg = Config()
    cfg.detection.farm_classes = list(TYPES)
    monkeypatch.setattr(Zone, "load", lambda *args, **kwargs: pytest.fail("no map/learning load"))
    assert zone_resource_exclusions(cfg.navigation) == {"wood", "stone"}
    w = World(cfg)
    scene = w.see([pred(kind, PLAYER[0]+100+i*20, PLAYER[1]) for i, kind in enumerate(TYPES)])
    assert {d.class_name for d in scene.farmables} == {"hide", "fiber", "ore"}
    assert Farm().plan(w.ctx, scene, w.ctx.status()) is not None


def test_vixen_fiber_exclusion_applies_to_scene_and_unfiltered_farm_picker():
    cfg = Config()
    cfg.navigation.zone = "vixen_tor"
    cfg.detection.farm_classes = list(TYPES)
    assert zone_resource_exclusions(cfg.navigation) == {"hide", "fiber"}
    w = World(cfg)
    scene = w.see([pred(kind, PLAYER[0]+100+i*20, PLAYER[1]) for i, kind in enumerate(TYPES)])
    assert {d.class_name for d in scene.farmables} == {"wood", "stone", "ore"}
    fake = replace(scene, farmables=(Detection("fiber", .99, Point(PLAYER[0]+10, PLAYER[1])),
                                     Detection("ore", .9, Point(PLAYER[0]+100, PLAYER[1]))))
    assert Farm().plan(w.ctx, fake, w.ctx.status()).target.class_name == "ore"
    assert Farm().plan(w.ctx, replace(fake, farmables=fake.farmables[:1]), w.ctx.status()) is None


@pytest.mark.parametrize("source", ["wiki", "unverified:example", "observed_live"])
def test_advisory_lists_and_live_non_sightings_never_exclude(tmp_path, source):
    cfg = zone_config(tmp_path, [{"source": source, "resources": [resource("hide", [2, 3])],
        "absent_t2plus": ["wood", "stone"], "complete_t2plus": True}])
    assert not zone_resource_exclusions(cfg.navigation)


def test_only_t2plus_positive_evidence_vetoes_a_negative_not_ambient_t1(tmp_path):
    cfg = zone_config(tmp_path, [
        {"source": "user", "resources": [resource("hide")], "absent_t2plus": ["wood", "stone", "fiber"]},
        {"source": "wiki", "resources": [dict(resource("wood", [1, 1]), t1_ambient=True)]},
        {"source": "observed_live", "t1_ambient": True, "resources": [resource("stone", [1, 1])]},
        {"source": "unverified:database", "resources": [resource("fiber", [2, 3])]},
    ])
    assert zone_resource_exclusions(cfg.navigation) == {"wood", "stone"}


def test_complete_user_t2plus_list_ignores_a_lower_tier_ambient_positive(tmp_path):
    cfg = zone_config(tmp_path, [
        {"source": "user", "resources": [resource("wood"), resource("stone"), resource("ore")],
         "complete_t2plus": True},
        {"source": "wiki", "resources": [dict(resource("hide", [1, 1]), t1_ambient=True)]},
    ])
    assert zone_resource_exclusions(cfg.navigation) == {"hide", "fiber"}


@pytest.mark.parametrize("missing", ["file", "zone"])
def test_missing_catalogue_or_unlisted_zone_preserves_farming(tmp_path, missing):
    cfg = Config()
    cfg.navigation.assets_folder = str(tmp_path)
    if missing == "zone":
        (tmp_path / "resources.json").write_text('{"version":1,"zones":{}}')
    w = World(cfg)
    scene = w.see([pred("fiber", PLAYER[0]+100, PLAYER[1])])
    assert scene.farmables and Farm().plan(w.ctx, scene, w.ctx.status()) is not None


@pytest.mark.parametrize("name", ["bridgewatch", "prospectors_hope", "steppe_cross"])
def test_wiki_only_neighbours_do_not_enforce_a_filter(name):
    cfg = Config()
    cfg.navigation.zone = name
    assert not zone_resource_exclusions(cfg.navigation)


def test_resource_rules_do_not_filter_combat_loot_or_enable_unconfigured_ore():
    cfg = Config()
    cfg.navigation.zone = "vixen_tor"
    w = World(cfg)
    scene = w.see([pred("mob", PLAYER[0]+100, PLAYER[1]), pred("silver", PLAYER[0]+120, PLAYER[1]),
                   pred("fiber", PLAYER[0]+130, PLAYER[1]), pred("ore", PLAYER[0]+140, PLAYER[1])])
    assert not scene.farmables
    assert Fight("fight_close", True).plan(w.ctx, scene, w.ctx.status()).target.class_name == "mob"
    assert Pickup().plan(w.ctx, scene, w.ctx.status()).target.class_name == "silver"


def test_fiber_alias_obeys_the_same_verified_negative():
    cfg = Config()
    cfg.navigation.zone = "vixen_tor"
    cfg.detection.farm_classes = ["cotton"]
    assert not World(cfg).see([pred("cotton", PLAYER[0]+100, PLAYER[1])]).farmables


def test_catalogue_preserves_attributed_t1_history_and_unknown_user_tiers():
    path = Zone.asset_folder("fractured_ground").parent / "resources.json"
    zones = json.loads(path.read_text(encoding="utf-8"))["zones"]
    fg = zones["fractured_ground"]["observations"]
    live = [o for o in fg if o["source"] == "observed_live"]
    assert {o["observed_at"] for o in live} == {"2026-11-04T04:14", "2026-11-04T11:08"}
    assert all(o["t1_ambient"] and o["resources"][0]["tiers"] == [1, 1] for o in live)
    wiki = next(o for o in fg if o["source"] == "wiki")
    assert {r["type"] for r in wiki["resources"] if r.get("t1_ambient")} == {"hide", "wood"}
    assert wiki["url"].startswith("https://wiki.albiononline.com/") and wiki["fetched_at"] == "2026-11-04"
    user = next(o for o in zones["vixen_tor"]["observations"] if o["source"].startswith("user"))
    assert all(r["tiers"] is None for r in user["resources"])
