"""List/retract persisted terrain evidence. No capture, game input or implicit removal."""
import json
import math

from .navigation import Zone
from .navigation_evidence import LearnedEvidence


def cmd_edges(cfg, args):
    zone = Zone.load(args.zone or cfg.navigation.zone, cfg.navigation, learned=False)
    path = args.file or zone.folder / "learned.json"
    scale = zone.grid.scale
    size = (zone.grid.mask.shape[1]//scale, zone.grid.mask.shape[0]//scale)
    evidence = LearnedEvidence(path, zone.name, size, scale)
    if args.action == "list":
        if args.index is not None or args.near is not None:
            raise ValueError("selectors belong to nav-edges retract")
        print(json.dumps({"file": str(path), "edges": [dict(index=i, **edge)
                         for i, edge in enumerate(evidence.edges)],
                         "pending_road_stalls": evidence.road_stalls}, indent=2))
        return 0
    if (args.index is None) == (args.near is None):
        raise ValueError("retract requires either --index or --near x,y")
    if not math.isfinite(args.radius) or args.radius <= 0:
        raise ValueError("radius must be finite and positive")
    near = zone.goal(args.near) if args.near is not None else None
    indices, pending = evidence.retract(args.index or (), near, args.radius)
    print(json.dumps({"file": str(path), "retracted_indices": indices,
                      "retracted_pending_stalls": pending, "remaining_edges": len(evidence.edges)}))
    return 0
