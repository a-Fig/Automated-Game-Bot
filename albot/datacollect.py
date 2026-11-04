"""Saving screenshots for training data: on a hotkey, and sampled at random during play."""
from __future__ import annotations

import logging
import os
import random
import uuid

log = logging.getLogger(__name__)


def to_pil(frame):
    """mss frames are numpy BGR; pyautogui frames are already PIL images."""
    if hasattr(frame, "save"):
        return frame
    from PIL import Image
    return Image.fromarray(frame[:, :, ::-1])


class Saver:
    def __init__(self, folder: str, grab) -> None:
        self.folder = folder
        self.grab = grab          # returns a fresh frame of the game window
        self.saved = 0

    def save(self, tag: str) -> None:
        try:
            path = os.path.join(self.folder, tag)
            os.makedirs(path, exist_ok=True)
            name = os.path.join(path, f"{tag}_{uuid.uuid4()}.png")
            to_pil(self.grab()).save(name)
            self.saved += 1
            log.info("screenshot saved: %s", name)
        except Exception:
            log.exception("could not save screenshot")

    def maybe(self, tag: str, one_in: int) -> None:
        if one_in > 0 and random.randint(1, one_in) == 1:
            self.save(tag)
