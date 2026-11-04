"""Observed standing positions and inferred collision edges, in live map pixels."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np


def open_observed_disc(mask, point, scale, radius=1.0):
    """Same bounded evidence disc for offline mask building and live learning."""
    x, y = np.rint(np.asarray(point)*scale).astype(int)
    r = math.ceil(radius*scale)
    x0, y0 = max(0, x-r), max(0, y-r)
    x1, y1 = min(mask.shape[1], x+r+1), min(mask.shape[0], y+r+1)
    disc = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.circle(disc, (x-x0, y-y0), r, 1, -1)
    patch = mask[y0:y1, x0:x1]
    opened = int(np.count_nonzero((disc != 0) & (patch == 0)))
    patch[disc != 0] = True
    return opened


class LearnedEvidence:
    def __init__(self, path, zone, size, scale):
        self.path = Path(path) if path is not None else None
        self.zone, self.size, self.scale = zone, tuple(size), scale
        self.positions = {}  # one disc per native cell; repeated fixes add no new ground
        self.edges = []
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (data.get("version") != 1 or data.get("zone") != zone
                    or data.get("size") != list(size) or data.get("scale") != scale
                    or data.get("coordinate_space") != "live_minimap"):
                raise ValueError("learned evidence does not match zone/map coordinates")
            for point in data["positions"]:
                self._check(point)
                self.positions[tuple(np.rint(np.asarray(point)*scale).astype(int))] = point
            for edge in data["edges"]:
                for key in ("player", "target", "a", "b"):
                    self._check(edge[key])
                self.edges.append(edge)

    def _check(self, point):
        if (len(point) != 2 or not all(math.isfinite(v) for v in point)
                or not all(0 <= v < bound for v, bound in zip(point, self.size))
                or not all(round(v*self.scale) < bound*self.scale for v, bound in zip(point, self.size))):
            raise ValueError("learned point outside map")

    def copy(self, persist=False):
        result = copy.deepcopy(self)
        if not persist:
            result.path = None
        return result

    def save(self):
        if self.path is None:
            return
        data = {"version": 1, "coordinate_space": "live_minimap", "zone": self.zone,
                "size": list(self.size), "scale": self.scale,
                "positions": list(self.positions.values()), "edges": self.edges}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
        temp.replace(self.path)

    def observe(self, point):
        self._check(point)
        cell = tuple(np.rint(np.asarray(point)*self.scale).astype(int))
        if cell in self.positions:
            return False
        self.positions[cell] = [v/self.scale for v in cell]
        self.save()
        return True

    def stall(self, player, target, clearance, length, now):
        self._check(player)
        self._check(target)
        direction = np.asarray(target, float)-player
        distance = float(np.linalg.norm(direction))
        # A transverse segment models a suspected ledge, not a disc around the
        # player. It cannot be placed safely if the waypoint is too close.
        offset = clearance+2/self.scale
        a = b = np.asarray(player, float)
        if distance > offset:
            direction /= distance
            center = np.asarray(player)+direction*min(distance, max(offset, min(3.0, distance/2)))
            side = np.array((-direction[1], direction[0]))*length/2
            a, b = center-side, center+side
        limit = np.asarray(self.size)-1/self.scale
        self.edges.append({"player": list(player), "target": list(target), "at": now,
                           "a": np.clip(a, 0, limit).tolist(), "b": np.clip(b, 0, limit).tolist()})
        self.save()
        return distance > offset

    def apply(self, grid, protect=()):
        for point in self.positions.values():
            open_observed_disc(grid.mask, point, grid.scale)
        blocked = np.zeros(grid.mask.shape, np.uint8)
        for edge in self.edges:
            a, b = [tuple(np.rint(np.asarray(edge[k])*grid.scale).astype(int)) for k in ("a", "b")]
            cv2.line(blocked, a, b, 1, 1)
        # Positive standing evidence wins over a guessed edge. Protect a full
        # clearance halo, including raster rounding, on every reload and replan.
        radius = math.ceil(grid.clearance*grid.scale)+1
        positions = [*self.positions.values(), *(e["player"] for e in self.edges), *protect]
        for point in positions:
            cell = tuple(np.rint(np.asarray(point)*grid.scale).astype(int))
            cv2.circle(blocked, cell, radius, 0, -1)
        grid.mask[blocked != 0] = False
        grid._inflate()
        return grid
