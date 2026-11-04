from pathlib import Path

import pytest

from albot.navigation_evidence import LearnedEvidence


@pytest.fixture(autouse=True)
def forbid_runtime_learned_file(monkeypatch):
    """Tests may learn in tmp_path, never consume the game owner's live evidence."""
    assets = Path(__file__).resolve().parents[1] / "assets/zones"
    original = LearnedEvidence.__init__

    def isolated(self, path, *args, **kwargs):
        if path is not None and Path(path).resolve().is_relative_to(assets) and Path(path).name == "learned.json":
            pytest.fail("runtime learned.json in test: use Zone.load(..., learned=False) or tmp_path")
        original(self, path, *args, **kwargs)

    monkeypatch.setattr(LearnedEvidence, "__init__", isolated)
