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

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

from .geometry import Point, find_detour
from .state import Detection, Scene, Status

log = logging.getLogger(__name__)


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
        if (scene.threats or ctx.under_attack()) and status.ready(ctx.slot(c.shield_slot)):
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
        if (self.close_only and not ctx.under_attack()
                and not scene.is_close(scene.mobs[0].center, ctx.config.detection.close_range)):
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


class Travel(Behavior):
    """One navigation/retreat step per fresh frame; forward travel yields to combat."""
    name = "travel"

    def __init__(self):
        self.task = None
        self.goal = ""
        self.timestamp = None
        self.last_state = ""
        self.trail: list[Point] = []
        self.attack_spots: list[Point] = []
        self.blocked_spots: list[Point] = []
        self.retreat_episode = None
        self.retreat_index = -1
        self.last_dash_at = -math.inf
        self.moved_at = None
        self.clicked_at = -math.inf

    def plan(self, ctx, scene, status):
        goal = ctx.config.behavior.travel_goal
        if goal:
            return Plan(self.name, f"travel to {goal}")
        return None

    def run(self, ctx, plan):
        ctx.check_safety()
        goal = ctx.config.behavior.travel_goal
        if ctx.retreat_required():
            if goal != self.goal:
                self._start_task(ctx, goal)
            return self._retreat(ctx)
        if ctx.higher_priority_applies(self.name):
            ctx.stop_travel()
            return False  # re-select the newly applicable combat behavior this tick
        if not goal:
            return True
        if ctx.under_attack():
            ctx.stop_travel()
            self._wait(ctx)
            return True
        if goal != self.goal:
            self._start_task(ctx, goal)
        status = ctx.status()
        frame = ctx.frame()
        if frame is None or status.timestamp == self.timestamp:
            if frame is None:
                ctx.stop_travel()
            self._wait(ctx)
            return True
        self.timestamp = status.timestamp
        step = self.task.tick(frame, ctx.now(), moving=ctx.travel_moving)
        # Models/health can publish while path planning runs. Never dispatch a
        # proposal without rechecking both safety and the higher-priority plans.
        ctx.check_safety()
        if ctx.retreat_required():
            return self._retreat(ctx)
        if ctx.higher_priority_applies(self.name):
            ctx.stop_travel()
            return False
        if ctx.under_attack():
            ctx.stop_travel()
            self._wait(ctx)
            return True
        if step.status != self.last_state:
            log.info("travel %s: %s", step.status, step.message)
            self.last_state = step.status
        if step.location is not None:
            point = step.location.point
            if not self.trail or point.dist(self.trail[-1]) >= ctx.config.navigation.trail_spacing:
                self.trail.append(point)
                self.trail = self.trail[-ctx.config.navigation.trail_limit:]
        if step.status == "arrived":
            log.info("travel arrived at %s; clearing goal", goal)
            ctx.config.behavior.travel_goal = ""
            ctx.travel_moving = False
            self.task, self.goal = None, ""
            return True  # arrival sends no input and does not fall through to farming
        give_up = ctx.config.behavior.travel_give_up_seconds
        if step.click is not None and step.status in ("move", "sidestep"):
            self.moved_at = ctx.now()
        elif self.moved_at is None:
            self.moved_at = ctx.now()
        elif give_up > 0 and ctx.now()-self.moved_at >= give_up:
            # A journey that can't move must not hold the bot still forever
            # (2026-11-04: 15 minutes of no_route before anyone noticed).
            log.error("travel gave up on %s: no move for %.0fs (%s: %s)",
                      goal, give_up, step.status, step.message)
            ctx.config.behavior.travel_goal = ""
            ctx.stop_travel()
            self.task, self.goal, self.moved_at = None, "", None
            return True
        if step.click is not None and step.status in ("move", "sidestep"):
            # Live 2026-11-04: a stop-key + click on every fresh frame (~5/s)
            # kept halting the character, so NavigateTo saw no progress and
            # sidestepped after 4 s. Keep walking: re-click about once a second.
            interval = ctx.config.behavior.travel_click_interval
            if step.status == "sidestep" or ctx.now()-self.clicked_at >= interval:
                # Right-click walk: a left click on a rock or plant starts
                # gathering it (seen live 2026-11-04: "+2 Rough Stone" mid-route).
                ctx.walk_to(step.click)
                self.clicked_at = ctx.now()
            ctx.travel_moving = True
            ctx.travel_retreating = False
        elif step.status != "waiting":
            ctx.stop_travel()
        self._wait(ctx)
        return True

    def _start_task(self, ctx, goal):
        from .navigation import NavigateTo
        from .actuator import DryRunInput
        n = ctx.config.navigation
        if (n.trail_spacing <= 0 or n.trail_limit < 2 or n.retreat_tolerance <= 0
                or n.attack_repeat_radius <= 0 or n.attack_block_radius < 1):
            raise ValueError("invalid travel retreat trail settings")
        ctx.stop_travel()
        self.task = NavigateTo(goal, config=n, learn=not isinstance(ctx.input, DryRunInput))
        self.goal, self.timestamp = goal, None
        self.trail, self.attack_spots, self.blocked_spots = [], [], []
        self.retreat_episode = None
        self.moved_at = None

    def _retreat(self, ctx):
        from .navigation import screen_click
        guard, n = ctx.travel_guard, ctx.config.navigation
        status, frame = ctx.status(), ctx.frame()
        new_episode = self.retreat_episode != guard.retreat_started
        if frame is None or not n.calibrated or (status.timestamp == self.timestamp and not new_episode):
            if frame is None or not n.calibrated:
                ctx.stop_travel()
            self._wait(ctx)
            return True
        self.timestamp = status.timestamp
        location = self.task.locator.locate(frame, ctx.now())
        if location is None:
            ctx.stop_travel()
            self._wait(ctx)
            return True
        point = location.point
        if ctx.travel_moving:
            self.task.record_position(location)
        if self.retreat_episode != guard.retreat_started:
            self.retreat_episode = guard.retreat_started
            self.retreat_index = len(self.trail)-1
            if any(point.dist(old) <= n.attack_repeat_radius for old in self.attack_spots):
                self.task.block_hazard(point, n.attack_block_radius)
                self.blocked_spots.append(point)
                self.blocked_spots = self.blocked_spots[-n.trail_limit:]
                self.task.route = []
                log.warning("travel: repeated attack near %.1f,%.1f; blocking radius %d",
                            point.x, point.y, n.attack_block_radius)
            self.attack_spots.append(point)
            self.attack_spots = self.attack_spots[-n.trail_limit:]
            log.warning("travel retreat: %d recorded locations to backtrack", len(self.trail))
            if not self.trail:
                log.error("travel retreat has no recorded route; cannot invent an escape direction")
        # A dash can pass several breadcrumbs. Only skip to an older point if
        # the observed player actually reached it; never shortcut across unknown ground.
        reached = [i for i in range(self.retreat_index+1)
                   if point.dist(self.trail[i]) <= n.retreat_tolerance]
        if reached:
            self.retreat_index = min(reached)-1
        outside_hazards = all(point.dist(hazard) > n.attack_block_radius+self.task.grid.clearance+0.5
                              for hazard in self.blocked_spots)
        if guard.critical_since is None and guard.calm(ctx.now()) and outside_hazards:
            ctx.stop_travel()
            # Discard the portion we just walked backwards, so a later retreat
            # never targets a more recent breadcrumb back toward the attacker.
            self.trail = self.trail[:max(0, self.retreat_index+1)]
            self.trail.append(point)
            guard.finish_retreat()
            self.task.route = []  # resume with a new route, retaining hazard blocks
            self.timestamp = None
            log.info("travel retreat: HP stable for %.1fs; resuming journey",
                     ctx.config.safety.retreat_calm_seconds)
            return True
        if self.retreat_index < 0:
            ctx.stop_travel()
            self._wait(ctx)  # oldest visited point reached (or no recorded route)
            return True
        target = self.trail[self.retreat_index]
        click = screen_click(Point(target.x-point.x, target.y-point.y), n)
        ctx.check_safety()
        if ctx.higher_priority_applies(self.name):
            ctx.stop_travel()
            return False
        if click is not None:
            s = ctx.config.safety
            slot = ctx.config.combat.leash_slot
            if ctx.status().ready(ctx.slot(slot)) and ctx.now()-self.last_dash_at >= s.retreat_dash_cooldown:
                ctx.move_to(click)
                ctx.press(slot)
                self.last_dash_at = ctx.now()
                ctx.resources.clear()
            ctx.walk_to(click)
            ctx.travel_moving = ctx.travel_retreating = True
        self._wait(ctx)
        return True

    def _wait(self, ctx):
        ctx.wait(ctx.config.behavior.idle_sleep,
                 interrupt=lambda: ctx.higher_priority_applies(self.name))


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

            # Walk to the node: wait for the gathering bar to appear. A fixed 3 s
            # gave up on any node more than ~3 s of walking away, so allow for
            # the distance too.
            distance = ctx.scene().player.dist(plan.target.center)
            walk_timeout = max(b.farm_walk_timeout, 1.0 + distance / b.walk_speed_px_per_s)
            if not ctx.wait_until(loading, walk_timeout, interrupt=ctx.danger):
                if ctx.danger():
                    return False
                log.info("farm %s at %.0f px: no gathering bar within %.1f s",
                         plan.target.class_name, distance, walk_timeout)
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
                log.info("farm %s: still gathering after %.0f s", plan.target.class_name, b.farm_max_seconds)
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
    items = [Shield(), Heal(), Fight("fight_close", True), Travel(), Pickup(), Farm(),
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
