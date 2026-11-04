"""
What the bot does, one behavior per activity.

Each behavior has two halves:
  plan(ctx, scene, status) -> Plan or None
      Pure: looks at one snapshot and says what it would do, without acting.
      Replay mode and the debug view use this on its own.
  run(ctx, plan) -> bool
      Acts. Reads the newest scene and status as it goes (ctx.scene(),
      ctx.status()), so waits and sleeps never leave it working on old data.
      Returns True if it did something useful.

The brain asks each behavior for a plan in priority order and runs the first one.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

from .geometry import Point, find_detour
from .state import Detection, Scene, Status


@dataclass(frozen=True)
class Plan:
    behavior: str
    description: str
    target: Optional[Detection] = None
    waypoint: Optional[Point] = None     # set when walking around an obstacle first
    point: Optional[Point] = None        # for plans aimed at a spot, not a detection


def danger(scene: Scene, status: Optional[Status], config, count_low_health: bool = True) -> bool:
    """Threat visible, health low, or a reachable mob close by."""
    if scene.threats:
        return True
    if count_low_health and status is not None and status.low_health:
        return True
    for mob in scene.mobs:
        if scene.is_close(mob.center, config.detection.close_range) and scene.reachable(mob.center):
            return True
    return False


def activity(scene: Scene) -> bool:
    """Anything worth stopping a wander for."""
    if scene.threats:
        return True
    for items in (scene.mobs, scene.farmables, scene.pickups, scene.wisps):
        if any(scene.reachable(d.center) for d in items):
            return True
    return False


def pick_target(ctx, scene: Scene, items: Sequence[Detection], behavior: str, verb: str) -> Optional[Plan]:
    """
    The nearest item with a clear straight path. If none has one, the nearest
    that a single detour waypoint can reach. Spots the bot keeps failing at are
    skipped for a while (see Blacklist).
    """
    candidates = [d for d in items if not ctx.blacklist.blocked(d.center)
                  and (behavior != "farm" or not ctx.resources.blocked(scene, d))]
    for d in candidates:
        if scene.reachable(d.center):
            return Plan(behavior, f"{verb} {d.class_name}", target=d)
    b = ctx.config.behavior
    if b.detours and scene.obstacles:
        for d in candidates[:3]:
            w = find_detour(scene.player, d.center, scene.obstacles, b.detour_margin, scene.window_size)
            if w is not None:
                return Plan(behavior, f"detour to {verb} {d.class_name}", target=d, waypoint=w)
    return None


def walk_detour(ctx, plan: Plan) -> bool:
    """Walk to the waypoint, then hand back so the next tick re-plans from a fresh frame."""
    scene = ctx.scene()
    ctx.walk_to(plan.waypoint)
    b = ctx.config.behavior
    walk_time = min(b.max_detour_walk, scene.player.dist(plan.waypoint) / b.walk_speed_px_per_s)
    ctx.wait(walk_time, interrupt=ctx.danger)
    ctx.check_progress(plan.target)
    return True


class Behavior:
    name = "behavior"

    def plan(self, ctx, scene: Scene, status: Status) -> Optional[Plan]:
        raise NotImplementedError

    def run(self, ctx, plan: Plan) -> bool:
        raise NotImplementedError


class Shield(Behavior):
    """A threat ("red") is visible and the shield slot is ready."""
    name = "shield"

    def plan(self, ctx, scene, status):
        c = ctx.config.combat
        if scene.threats and status.ready(ctx.slot(c.shield_slot)):
            return Plan(self.name, "shield up")
        return None

    def run(self, ctx, plan):
        c = ctx.config.combat
        ctx.press(c.shield_slot)
        ctx.wait(c.shield_wait)
        return True


class Heal(Behavior):
    """Health is low and the leash slot is ready: leash, then wait to recover."""
    name = "heal"

    def plan(self, ctx, scene, status):
        if status.low_health and status.ready(ctx.slot(ctx.config.combat.leash_slot)):
            return Plan(self.name, "leash and heal")
        return None

    def run(self, ctx, plan):
        c = ctx.config.combat
        point = self.escape_point(ctx.scene(), ctx.config)
        if point is None:
            return False
        ctx.move_to(point)
        ctx.press(c.leash_slot)
        ctx.resources.clear()  # the 16 m dash is not calibrated in screen pixels

        # wait for health to come back, unless something attacks. danger() has to
        # ignore low health here, or this wait would end immediately (it did in the original)
        def threat():
            return ctx.danger(count_low_health=False)

        ctx.wait_until(lambda: not ctx.status().low_health, c.heal_max_wait, interrupt=threat)
        ctx.wait(c.heal_settle, interrupt=threat)
        return True

    @staticmethod
    def escape_point(scene, config) -> Optional[Point]:
        nearby = [d.center for d in (*scene.mobs, *scene.threats)
                  if scene.is_close(d.center, config.detection.close_range)]
        angle = 0.0
        if nearby:
            dx = scene.player.x - sum(p.x for p in nearby) / len(nearby)
            dy = scene.player.y - sum(p.y for p in nearby) / len(nearby)
            angle = math.atan2(dy, dx)
        # Try directly away first, then nearby directions if terrain blocks it.
        for turn in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8):
            a = angle + turn * math.pi / 8
            p = Point(scene.player.x + math.cos(a) * config.combat.escape_distance,
                      scene.player.y + math.sin(a) * config.combat.escape_distance)
            if (0 <= p.x < scene.window_size[0] and 0 <= p.y < scene.window_size[1]
                    and scene.reachable(p)):
                return p
        return None


class Fight(Behavior):
    """Attack the nearest reachable mob. Two instances: close mobs first, any mob later."""

    def __init__(self, name: str, close_only: bool) -> None:
        self.name = name
        self.close_only = close_only

    def plan(self, ctx, scene, status):
        if not scene.mobs:
            return None
        if self.close_only and not scene.is_close(scene.mobs[0].center, ctx.config.detection.close_range):
            return None
        return pick_target(ctx, scene, scene.mobs, self.name, "attack")

    def run(self, ctx, plan):
        if plan.waypoint is not None:
            return walk_detour(ctx, plan)
        c = ctx.config.combat
        for step in c.openers:
            ctx.cast_if_ready(step)
        ctx.click(plan.target.center)
        ctx.press(ctx.config.keys.attack_after_click)
        for step in c.rotation:
            if ctx.cast_if_ready(step):
                return True
        ctx.cast(c.fallback)
        return True


class Pickup(Behavior):
    """Silver, cups and caged wisps, nearest first."""
    name = "pickup"

    def plan(self, ctx, scene, status):
        items = scene.by_distance(list(scene.pickups) + list(scene.wisps))
        return pick_target(ctx, scene, items, self.name, "pick up")

    def run(self, ctx, plan):
        if plan.waypoint is not None:
            return walk_detour(ctx, plan)
        b = ctx.config.behavior
        is_wisp = plan.target.class_name in ctx.config.detection.wisp_classes
        ctx.click(plan.target.center)
        ctx.wait(b.wisp_wait if is_wisp else b.pickup_wait, interrupt=ctx.danger)
        ctx.check_progress(plan.target)
        return True


class Farm(Behavior):
    name = "farm"

    def plan(self, ctx, scene, status):
        return pick_target(ctx, scene, scene.farmables, self.name, "farm")

    def run(self, ctx, plan):
        if plan.waypoint is not None:
            return walk_detour(ctx, plan)
        b = ctx.config.behavior
        ctx.want_loading_bar(True)
        try:
            clicked_at = ctx.now()
            spot = ctx.resources.world_point(ctx.scene(), plan.target.center)
            ctx.click(plan.target.center)

            def loading():
                s = ctx.status()
                return s.timestamp > clicked_at and bool(s.loading_bar)

            # walk to the node: wait for the gathering bar to appear
            if not ctx.wait_until(loading, b.farm_walk_timeout, interrupt=ctx.danger):
                if ctx.danger():
                    return False
                # No bar and no danger: remember the original world spot, even
                # if the camera moved while approaching the depleted node.
                ctx.resources.failure(plan.target.class_name, spot)
                return False
            ctx.resources.move(ctx.scene(), None)  # gathering stopped the approach

            # gathering: wait for the bar to go away
            finished = ctx.wait_until(lambda: not loading(), b.farm_max_seconds, interrupt=ctx.danger)
            if ctx.danger():
                return False
            if not finished:
                ctx.resources.failure(plan.target.class_name, spot)
                return False
            ctx.wait_for_new_scene()
            return True
        finally:
            ctx.want_loading_bar(False)


class Eat(Behavior):
    name = "eat"

    def plan(self, ctx, scene, status):
        if ctx.now() - ctx.last_ate >= ctx.config.behavior.eat_every_minutes * 60:
            return Plan(self.name, "eat food")
        return None

    def run(self, ctx, plan):
        ctx.press(ctx.config.keys.food)
        ctx.last_ate = ctx.now()
        ctx.wait(ctx.config.behavior.eat_wait)
        return True


class Wander(Behavior):
    """Nothing to do for a while: walk somewhere reachable nearby."""
    name = "wander"
    resets_boredom = False   # keep wandering until something turns up, as the original did

    def plan(self, ctx, scene, status):
        if ctx.bored <= ctx.config.behavior.bored_ticks:
            return None
        point = self.pick_point(ctx, scene)
        if point is None:
            return None
        return Plan(self.name, "wander", point=point)

    @staticmethod
    def pick_point(ctx, scene: Scene, attempts: int = 50) -> Optional[Point]:
        b = ctx.config.behavior
        w, h = scene.window_size
        for _ in range(attempts):
            angle = ctx.random.uniform(0, 2 * math.pi)
            r = ctx.random.uniform(0, 1)
            p = Point(scene.player.x + math.cos(angle) * r * w * b.wander_range,
                      scene.player.y + math.sin(angle) * r * h * b.wander_range)
            if not (0 <= p.x < w and 0 <= p.y < h):
                continue
            # reject points that are blocked OR too close to bother with
            if scene.is_close(p, b.wander_min_distance) or not scene.reachable(p):
                continue
            return p
        return None

    def run(self, ctx, plan):
        b = ctx.config.behavior
        for _ in range(b.wander_clicks):
            ctx.walk_to(plan.point)
            if not ctx.wait(b.wander_click_interval, interrupt=lambda: activity(ctx.scene())):
                break
        return True


def default_behaviors() -> dict[str, Behavior]:
    items = [Shield(), Heal(), Fight("fight_close", True), Pickup(), Farm(),
             Fight("fight_far", False), Eat(), Wander()]
    return {b.name: b for b in items}


class Blacklist:
    """
    Skips screen spots the bot keeps failing at.

    The camera follows the player, so a target's screen position only stays
    fixed when the player isn't moving. Failing at the same spot several times in
    a row therefore means the bot is stuck: clicking a node that won't gather, or
    an item it can't reach. That spot is skipped for a while. Moving changes all
    the positions, which is why entries also expire.
    """

    def __init__(self, config, clock) -> None:
        self.b = config.behavior
        self.clock = clock
        self._last_spot: Optional[Point] = None
        self._repeats = 0
        self._blocked: list[tuple[Point, float]] = []

    def failure(self, p: Point) -> None:
        if self._last_spot is not None and self._last_spot.dist(p) <= self.b.stuck_radius:
            self._repeats += 1
        else:
            self._last_spot, self._repeats = p, 1
        if self._repeats >= self.b.stuck_repeats:
            self._blocked.append((p, self.clock.now() + self.b.blacklist_seconds))
            self._last_spot, self._repeats = None, 0

    def success(self, p: Point) -> None:
        self._last_spot, self._repeats = None, 0

    def blocked(self, p: Point) -> bool:
        now = self.clock.now()
        self._blocked = [(q, until) for q, until in self._blocked if until > now]
        return any(q.dist(p) <= self.b.stuck_radius for q, _ in self._blocked)

    def __len__(self) -> int:
        return len(self._blocked)


class SystemRandom:
    """Indirection so tests can seed it."""

    def __init__(self, seed=None) -> None:
        self._r = random.Random(seed)

    def uniform(self, a: float, b: float) -> float:
        return self._r.uniform(a, b)
