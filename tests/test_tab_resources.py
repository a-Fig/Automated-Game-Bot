import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from albot.tab_resources import TEMPLATES, extract_tab_resources, update_catalog
from albot.config import Config
from albot.resources import zone_resource_exclusions
from tools.zone.tab_resources import main


FIXTURES = Path(__file__).parent / "fixtures/resources"


def frame(n=4):
    image = np.full((1400, 2240, 3), 80, np.uint8)
    image[32:124, 1595:1979] = cv2.imread(str(FIXTURES / f"fg_0{n}_header.png"))
    return image


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_recorded_headers_read_three_t2_to_t3_resources_and_two_faded_slots(n):
    observation = extract_tab_resources(frame(n), f"fg_0{n}.png", "2026-11-04")
    assert observation["source"] == "ingame_tab" and observation["tab_header_min_tier"] == 2
    assert {r["type"]: r["tiers"] for r in observation["resources"]} == {
        "hide": [2, 3], "fiber": [2, 3], "ore": [2, 3]}
    assert observation["absent_t2plus"] == ["wood", "stone"]
    assert not observation["warnings"]
    assert all(r["icon_score"] > .98 for r in observation["readings"].values())


def test_dimmed_pil_header_uses_the_same_templates():
    image = (frame()*.6).astype(np.uint8)
    observation = extract_tab_resources(Image.fromarray(image[:, :, ::-1]), "dim.png", "2026-11-04")
    assert {r["type"] for r in observation["resources"]} == {"hide", "fiber", "ore"}
    assert observation["absent_t2plus"] == ["wood", "stone"]


def test_uncalibrated_tier_caption_keeps_the_type_but_does_not_invent_tiers():
    image = frame()
    # A different long-stroke group instead of the last I; retain a clear tier
    # caption with tall glyphs either side of its separator, but no II-III match.
    image[104:115, 1953:1957] = (45, 55, 65)
    observation = extract_tab_resources(image, "unknown_tier.png", "2026-11-04")
    assert next(r for r in observation["resources"] if r["type"] == "ore")["tiers"] is None
    assert any("ore" in warning for warning in observation["warnings"])


@pytest.mark.parametrize("problem", ["closed", "obscured", "wrong_size", "blank_caption"])
def test_invalid_headers_refuse_to_guess(problem):
    image = frame()
    if problem == "closed":
        image[:] = 80
    elif problem == "obscured":
        image[40:100, 1912:1974] = 80
    elif problem == "wrong_size":
        image = cv2.resize(image, (1120, 700))
    else:
        image[102:118, 1912:1970] = (130, 170, 220)
    with pytest.raises(ValueError):
        extract_tab_resources(image, "bad.png", "2026-11-04")


def test_catalogue_refresh_preserves_user_t1_wiki_and_other_zones(tmp_path):
    path = tmp_path / "resources.json"
    data = {"version": 1, "zones": {"fractured_ground": {"name": "Fractured Ground", "observations": [
        {"source": "observed_live", "t1_ambient": True, "resources": [{"type": "wood", "tiers": [1, 1]}]},
        {"source": "user", "resources": [], "absent_t2plus": ["fiber"]},
        {"source": "wiki", "resources": [{"type": "stone", "tiers": [1, 1], "t1_ambient": True}]},
    ]}, "untouched": {"observations": []}}}
    path.write_text(json.dumps(data))
    observation = extract_tab_resources(frame(), "fg_04.png", "2026-11-04")
    for _ in range(2):
        update_catalog(path, "fractured_ground", observation)
    result = json.loads(path.read_text())
    assert result["zones"]["untouched"] == data["zones"]["untouched"]
    records = result["zones"]["fractured_ground"]["observations"]
    assert records[:3] == data["zones"]["fractured_ground"]["observations"]
    assert len(records) == 4 and records[-1]["tab_header_min_tier"] == 2
    cfg = Config()
    cfg.navigation.assets_folder = str(tmp_path)
    # Header's positive fiber wins; ambient T1 records do not define zone types.
    assert zone_resource_exclusions(cfg.navigation) == {"wood", "stone"}
    assert not path.with_suffix(".json.tmp").exists()


def test_offline_cli_never_captures_or_sends_input_and_bad_image_cannot_overwrite(tmp_path, monkeypatch, capsys):
    from albot import vision, actuator
    monkeypatch.setattr(vision, "Capture", lambda *args: pytest.fail("no live capture"))
    monkeypatch.setattr(actuator, "Input", lambda *args: pytest.fail("no input"))
    image, out = tmp_path / "frame.png", tmp_path / "resources.json"
    cv2.imwrite(str(image), frame())
    args = ["--zone", "fractured_ground", "--image", str(image), "--out", str(out),
            "--captured-at", "2026-11-04"]
    assert main(args+["--dry-run"]) == 0 and not out.exists()
    assert main(args) == 0
    assert json.loads(out.read_text())["zones"]["fractured_ground"]["observations"][0]["source"] == "ingame_tab"
    before = out.read_bytes()
    cv2.imwrite(str(image), np.full((1400, 2240, 3), 80, np.uint8))
    assert main(args) == 1 and out.read_bytes() == before
    assert main(["--zone", "fractured_ground", "--live", "--out", str(out)]) == 1
    assert "--save-image" in capsys.readouterr().err
