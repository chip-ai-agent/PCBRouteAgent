"""Scripted Detailed executor + the event-driven Global/Detailed orchestrator.

``ScriptedDetailed`` is a deterministic Detailed role: it walks a DetailedSubtask stage by
stage through the tool gateway (start_route -> route_to / route_via -> finish), picks
candidate points inside the overlap / transition regions, retries locally with
alternative candidates and modes, and returns FeedbackPackets. It exists so the whole
closed loop (reset -> guide -> tools -> feedback -> patch -> save/reload) can be exercised
and measured without an LLM; an LLM Detailed provider drives the same gateway.

``Orchestrator`` runs tasks in net order on stable checkpoints, routes feedback through the
policy router, asks the Global provider for a patch when the router says ``replan``, and
stops on the budget / replan caps.
"""
from __future__ import annotations

import os
import math
import time

_TRACE_NETS = bool(os.environ.get("ROUTEAGENT_TRACE_NETS"))   # debug: pad groups of every work-set net per task
from dataclasses import dataclass, field
from typing import Callable

from route_agent.exec.gateway import ToolGateway
from route_agent.exec.maze import maze_waypoints
from route_agent.exec.waypoints import _pt_seg_dist, expand, foreign_obstacles, obstacle_aware_candidates, via_sites_around
from route_agent.guide.geometry import center, clamp_point, contains, shrink
from route_agent.loop.budget import Budget, BudgetExhausted
from route_agent.loop.router import Decision, StallTracker, route_feedback
from route_agent.schema.feedback import Attempt, FailureEvidence, FeedbackPacket
from route_agent.schema.tasks import DetailedSubtask


def candidate_points(bbox: list[float], keep_out_mm: float, n: int = 5) -> list[list[float]]:
    """Centre first, then a diamond of offsets at 25% of the (shrunk) region extent."""
    inner = shrink(bbox, keep_out_mm) or bbox
    cx, cy = center(inner)
    x1, y1, x2, y2 = inner
    dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
    pts = [[cx, cy], [cx + dx, cy], [cx - dx, cy], [cx, cy + dy], [cx, cy - dy],
           [cx + dx, cy + dy], [cx - dx, cy - dy], [cx + dx, cy - dy], [cx - dx, cy + dy]]
    return [clamp_point(inner, *p) for p in pts[:n]]


@dataclass
class StepRecord:
    stage: str
    kind: str
    target: list[float]
    ok: bool
    reason: str
    calls: int
    mode: int = 2                       # routing mode that produced this record (2 walkaround, 1 shove)


@dataclass
class TaskResult:
    task_id: str
    net_code: int
    outcome: str                        # done | blocked | partial | unsupported
    steps: list[StepRecord]
    feedback: FeedbackPacket | None
    pad_groups: int
    deviations: list[dict]
    calls: int
    checkpoint: int | None = None


class ScriptedDetailed:
    SHOVE = 1

    OBSTACLE_MARGIN_MM = 1.5          # how far around a region inspect_region looks for foreign copper
    VIA_SITE_RADII_MM = (0.7, 1.0, 1.4, 1.9, 2.5)   # rings around a boxed-in head reported as via sites
    MAZE_CELL_MM = 0.5                # occupancy grid cell of the same-layer maze guidance
    MAZE_MARGIN_MM = 2.0              # how far beyond the stage bbox the maze may look
    _SESSION_REFUSALS = ("no net selected", "needs an idle session (cancel/finish first)")

    def __init__(self, gateway: ToolGateway, snapshot_id: str, guide_id: str, mode: int = 2,
                 finish_first: bool = True, shove_fallback: bool = True, obstacle_aware: bool = True,
                 backtrack: int = 2, maze: int = 1, alt_anchors: int = 2) -> None:
        """``finish_first``: at the entry of every stage try ``start_route(head) + finish`` before
        any explicit waypoint, so PNS gets to find its own legal path whenever one exists; the
        corridor walk is the fallback. Off = the v0 behaviour (finish only in the last stage).

        ``shove_fallback``: a commit that comes back ``stuck`` in the task's mode (default 2 =
        walkaround) is tried once more in shove mode (1), which may push other nets' tracks
        aside; pre-check rejections (``pad_graze``, ``via_on_thru_pad``) are not retried since
        no mode fixes them. The shove retry costs facade calls but no local-retry token.

        ``obstacle_aware``: waypoint candidates come from ``exec.waypoints`` — one
        ``inspect_region`` per stage boundary, grid points that keep clearance from foreign
        copper, ranked along the straight intent; the blind centre + diamond is the fallback
        when nothing in the region is clear. The source anchor is the one nearest the first
        boundary instead of the first one found.

        ``backtrack``: when a boundary (or the final stage) runs out of candidates, roll the board
        back to the sub-checkpoint taken before the previous boundary's commit and try that
        boundary's next candidate, at most this many times per task. Sub-checkpoints go through
        ``gateway.call_local`` (normal executor pool, not the recovery reserve); each backtrack
        also spends one local-retry token. 0 = v0 behaviour (first failure ends the task).

        ``maze``: same-layer detours per task (``exec/maze.py``). When a straight commit to a
        waypoint or to the target is stuck, a 0.5 mm occupancy grid over the stage bbox grown by
        2 mm (foreign copper + clearance blocked) is searched with A* and the simplified corners
        are committed one ``route_to`` at a time, the original point last; any hop failing rolls
        the detour back. PNS still routes every hop. No via, no layer change, never beyond the
        margin: what that cannot solve is reported to the Global (``maze`` attempt with
        ``no_same_layer_path_within_margin_or_too_long``). Measured: 124 of 131 D3-B and 381 of 387
        D3-C maze calls found no same-layer path at all, the few detours were net-neutral, so the
        default is 1: one look per edge, mostly for the evidence. 0 = off.

        ``alt_anchors``: when the chosen start pad is boxed in, restart the walk from other copper
        already joined to the source cluster on the stage's layer (pads merged by earlier edges,
        track ends, vias; ``connected_points``) within the first stage's bbox grown by 3 mm, nearest
        the target first, at most this many times per task. No via, no copper. Motive: half of the
        D3-B and 39% of the D3-C failed edges have such copper nearby. 0 = off."""
        self.gw = gateway
        self.snapshot_id = snapshot_id
        self.guide_id = guide_id
        self.mode = mode
        self.finish_first = finish_first
        self.shove_fallback = shove_fallback
        self.obstacle_aware = obstacle_aware
        self.backtrack = int(backtrack)
        self.maze = int(maze)
        self.alt_anchors = int(alt_anchors)

    # -- helpers --------------------------------------------------------------
    def _call(self, tool: str, **args) -> dict:
        """Gateway call that turns a budget refusal into ``BudgetExhausted`` so the orchestrator
        rolls back and stops instead of misreading refused calls as routing failures."""
        out = self.gw.call(tool, **args)
        if self.gw.exhausted:
            raise BudgetExhausted(self.gw.exhausted)
        return out

    def _anchor(self, task: DetailedSubtask, anchors: list[dict], stage, toward: list[float] | None = None) -> dict | None:
        """An anchor of the cluster inside ``stage`` on its layer; with ``toward`` the one nearest
        that point (the first portal / the target), otherwise the first found."""
        ok = [a for a in anchors if stage.layer in a["layers"] and contains(stage.bbox, a["x_mm"], a["y_mm"])]
        if not ok:
            return None
        if toward is None or not self.obstacle_aware:
            return ok[0]
        return min(ok, key=lambda a: (a["x_mm"] - toward[0]) ** 2 + (a["y_mm"] - toward[1]) ** 2)

    def _candidates(self, task, region_bbox, keep, layers, head, hint) -> tuple[list[list[float]], int]:
        """Waypoints for one boundary: obstacle-aware grid when enabled and something is clear,
        else the blind centre + diamond. Returns (points, number of foreign obstacles seen)."""
        if not self.obstacle_aware:
            return candidate_points(region_bbox, keep), -1
        reg = self._call("inspect_region", bbox=expand(region_bbox, self.OBSTACLE_MARGIN_MM))
        obst = foreign_obstacles(reg, task.net_code)
        n_obst = len(obst["pads"]) + len(obst["tracks"]) + len(obst["vias"])
        pts = obstacle_aware_candidates(region_bbox, keep, obst, layers, head, hint)
        return (pts or candidate_points(region_bbox, keep)), n_obst

    def _head_attempt(self, kind, pt, ok, cost, reason, region, mode=None) -> Attempt:
        return Attempt(kind=kind, target=[round(pt[0], 4), round(pt[1], 4)], mode=self.mode if mode is None else mode,
                       ok=ok, cost=cost, native_reason=reason or ("" if ok else "unknown"), region_ref=region)

    def _modes(self) -> list[int]:
        return [self.mode, self.SHOVE] if self.shove_fallback and self.mode != self.SHOVE else [self.mode]

    def _commit(self, kind, head, layer, pt, region_ref, stage_id, attempts, steps) -> tuple[bool, str, list[float]]:
        """One waypoint commit: ``start_route(head)`` + ``kind(pt)`` in the task's mode, retried once in
        shove mode when the first try was ``stuck``. Returns (ok, last native reason, landing point):
        for ``route_via`` the landing point is the via position read back from the engine (PNS
        accepts an arrival within the via radius), otherwise the requested point."""
        last_reason = ""
        for mode in self._modes():
            c0 = self.gw.adapter.log.calls
            s = self._call("start_route", x=head[0], y=head[1], layer=layer)
            if not s.get("ok"):
                last_reason = s.get("reason", "start_route refused")
                attempts.append(self._head_attempt("start_route", head, False, self.gw.adapter.log.calls - c0, last_reason, region_ref, mode))
                return False, last_reason, list(pt)
            r = self._call(kind, x=pt[0], y=pt[1], mode=mode)
            ok = bool(r.get("ok"))
            cost = self.gw.adapter.log.calls - c0
            last_reason = r.get("native_reason") or r.get("reason", "")
            attempts.append(self._head_attempt(kind, pt, ok, cost, last_reason, region_ref, mode))
            steps.append(StepRecord(stage_id, kind, [pt[0], pt[1]], ok, last_reason, cost, mode))
            if ok:
                landed = r.get("via_xy") if kind == "route_via" and r.get("via_xy") else [pt[0], pt[1]]
                return True, "", [float(landed[0]), float(landed[1])]
            if last_reason != "stuck":          # pre-check rejection or refusal: shove cannot help
                return False, last_reason, list(pt)
        return False, last_reason, list(pt)

    def _merged_points(self, task: DetailedSubtask, tol: float = 1e-3) -> list[list[float]]:
        """Anchor points now joined to the source cluster that are not its own anchors: what a
        ``finish`` merged when it closed to a cluster other than the target."""
        own = [(float(a["x_mm"]), float(a["y_mm"])) for a in task.source_anchors]
        out: list[list[float]] = []
        for a in task.source_anchors[:1]:
            layer = a["layers"][0] if a.get("layers") else task.stages[0].layer
            for pt in self._call("connected_points", x=a["x_mm"], y=a["y_mm"], layer=layer).get("points", []):
                q = [round(float(pt[0]), 4), round(float(pt[1]), 4)]
                if q not in out and not any(abs(pt[0] - ox) <= tol and abs(pt[1] - oy) <= tol for ox, oy in own):
                    out.append(q)
        return out[:8]

    # -- stage-level backtracking ----------------------------------------------
    def _local_checkpoint(self) -> int | None:
        r = self.gw.call_local("checkpoint")
        return r.get("handle") if r.get("ok") else None

    def _local_rollback(self, handle: int, net_code: int) -> bool:
        r = self.gw.call_local("rollback", handle=handle)
        if not r.get("ok"):
            return False
        return bool(self._call("select_net", net_code=net_code).get("ok"))   # rollback deselects the net

    def _release_subs(self, subs: list) -> None:
        for k, h in enumerate(subs):
            if h is not None:
                try:
                    self.gw.adapter.release_checkpoint(h)
                except Exception:  # noqa: BLE001 - bookkeeping
                    pass
                subs[k] = None

    def _backtrack_to(self, i: int, subs: list, ptr: list, cands: list, task, budget) -> int:
        """Find the nearest earlier boundary j < i with an untried candidate, restore the board to
        before its commit, invalidate everything after it. Returns j, or -1 when none."""
        j = i - 1
        while j >= 0 and (subs[j] is None or cands[j] is None or ptr[j] + 1 >= len(cands[j])):
            j -= 1
        if j < 0 or not budget.can_retry(task.task_id):
            return -1
        if not self._local_rollback(subs[j], task.net_code):
            return -1
        budget.use_retry(task.task_id)
        ptr[j] += 1
        for k in range(j + 1, len(cands)):
            ptr[k], cands[k] = 0, None
            if subs[k] is not None:
                try:
                    self.gw.adapter.release_checkpoint(subs[k])
                except Exception:  # noqa: BLE001
                    pass
                subs[k] = None
        return j

    def _frozen_signature(self) -> dict[int, tuple[int, int]]:
        """(tracks, vias) per frozen-scope net; equal before and after a task = scope respected."""
        sig = {}
        for code in self.gw.frozen_scope or []:
            info = self._call("inspect_net", net_code=int(code))
            sig[int(code)] = (len(info.get("tracks", [])), len(info.get("vias", [])))
        return sig

    def _try_finish(self, head, layer, stage, attempts, steps) -> bool:
        """``start_route(head, layer)`` + ``finish``: PNS closes to the nearest unconnected pad of the
        net on this layer, or commits nothing; retried once in shove mode. Records every attempt."""
        for mode in self._modes():
            c0 = self.gw.adapter.log.calls
            s = self._call("start_route", x=head[0], y=head[1], layer=layer)
            if not s.get("ok"):
                return False
            r = self._call("finish", mode=mode)
            ok = bool(r.get("ok"))
            cost = self.gw.adapter.log.calls - c0
            reason = "" if ok else "finish failed"
            attempts.append(self._head_attempt("finish", head, ok, cost, reason, stage.corridor_id, mode))
            steps.append(StepRecord(stage.corridor_id, "finish", [head[0], head[1]], ok, reason, cost, mode))
            if ok:
                return True
        return False

    # -- same-layer maze detour ---------------------------------------------------
    def _maze_commit(self, task, head, layer, kind, pt, region_bbox, region_ref, stage_id, trk_r, attempts, steps):
        """Detour from ``head`` to ``pt`` on ``layer`` around foreign copper: A* corners committed as
        ``route_to`` hops, the last hop as ``kind`` (``route_to`` or ``route_via`` at ``pt``). All or
        nothing (sub-checkpoint). Returns (ok, landing point)."""
        from route_agent.guide.geometry import normalize
        rb = normalize(region_bbox)
        box = expand([min(rb[0], head[0], pt[0]), min(rb[1], head[1], pt[1]),
                      max(rb[2], head[0], pt[0]), max(rb[3], head[1], pt[1])], self.MAZE_MARGIN_MM)
        reg = self._call("inspect_region", bbox=box)
        wps = maze_waypoints(box, self.MAZE_CELL_MM, foreign_obstacles(reg, task.net_code), layer, trk_r, head, pt)
        if not wps or len(wps) < 2:
            # no same-layer path on the grid within the margin, or the straight line is already clear
            # on it (the real blocker is finer than the grid): recorded for the Global as evidence
            why = "no_same_layer_path_within_margin_or_too_long" if not wps else "grid_sees_straight_line_clear"
            attempts.append(self._head_attempt("maze", pt, False, 1, why, f"{region_ref}:maze"))
            return False, head
        h = self._local_checkpoint()
        cur = head
        for k, wp in enumerate(wps):
            last = k == len(wps) - 1
            ok, _reason, landed = self._commit(kind if last else "route_to", cur, layer, wp, f"{region_ref}:maze",
                                               stage_id, attempts, steps)
            if not ok:
                if h is not None:
                    self._local_rollback(h, task.net_code)
                    self._release_one(h)
                return False, head
            cur = landed
        self._release_one(h)
        return True, cur

    def _release_one(self, h) -> None:
        if h is not None:
            try:
                self.gw.adapter.release_checkpoint(h)
            except Exception:  # noqa: BLE001 - bookkeeping
                pass

    # -- boxed-in evidence for the Global ------------------------------------
    def _boxed_in(self, reason: str) -> bool:
        """The stage ran out of candidates for a routing reason (stuck, finish failed, pre-check
        rejections), not because the session or the net was refused."""
        return not (reason.startswith("start_route") or reason in self._SESSION_REFUSALS)

    def _other_layer(self, layer: str) -> str:
        names = list(self.gw.adapter.layers.names)
        return names[-1] if layer == names[0] else names[0]

    def _is_thru(self, task, head, tol: float = 1e-3) -> bool:
        """The head sits on one of the task's anchors that spans every copper layer."""
        n = len(self.gw.adapter.layers.names)
        for a in list(task.source_anchors) + list(task.target_anchors):
            if abs(float(a["x_mm"]) - head[0]) <= tol and abs(float(a["y_mm"]) - head[1]) <= tol:
                return len(a.get("layers") or []) >= n
        return False

    def _nearby_starts(self, task, stage, layer, exclude: set, toward) -> list[list[float]]:
        """Copper already joined to the source cluster on ``layer`` within the first stage's bbox
        grown by 2 x OBSTACLE_MARGIN_MM (pads merged by earlier edges, track ends, vias), minus
        ``exclude`` (rounded (x, y) keys), nearest ``toward`` first."""
        near = expand(stage.bbox, 2 * self.OBSTACLE_MARGIN_MM)
        starts: dict = {}
        for a in task.source_anchors:
            lay = a["layers"][0] if a.get("layers") else layer
            for p in self._call("connected_points", x=a["x_mm"], y=a["y_mm"], layer=lay).get("points", []):
                if len(p) > 2 and p[2] != layer:
                    continue
                key = (round(float(p[0]), 3), round(float(p[1]), 3))
                if key not in exclude and contains(near, float(p[0]), float(p[1])):
                    starts[key] = [key[0], key[1]]
        return sorted(starts.values(), key=lambda q: (q[0] - toward[0]) ** 2 + (q[1] - toward[1]) ** 2)

    def _boxed_evidence(self, task, head, layer, stage, via_r, toward, near_mm: float = 2.0):
        """Everything the Global needs to decide a layer change for a head that could not leave
        its spot on ``layer`` (one ``inspect_region`` + one ``connected_points`` per source anchor):

        * ``via_sites`` - clearance-checked ring points next to the head where a via fits on both
          layers (foreign copper and the net's own thru-hole pads are obstacles: the engine refuses
          a via inside a thru pad or grazing its rim), nearest first, toward the target first;
        * ``blockers`` - the foreign pads / tracks within ``near_mm`` of the head, with position
          and layer(s);
        * ``thru`` - the head is a thru-hole pad (a corridor on the other layer needs no via);
        * ``alt_starts`` - other copper already joined to the source cluster on this layer near the
          first stage (pads merged by earlier edges, track ends), nearest the target first.
        Returns (via_sites, blockers, thru, alt_starts)."""
        other = self._other_layer(layer)
        hx, hy = float(head[0]), float(head[1])
        look = self.VIA_SITE_RADII_MM[-1] + self.OBSTACLE_MARGIN_MM
        reg = self._call("inspect_region", bbox=expand([hx, hy, hx, hy], look))
        obst = foreign_obstacles(reg, task.net_code)
        n_layers = len(self.gw.adapter.layers.names)
        obst["vias"] = list(obst["vias"]) + [
            {"x_mm": p["x_mm"], "y_mm": p["y_mm"], "net_code": p["net_code"],
             "diameter_mm": max(float(p.get("w_mm", 1.0)), float(p.get("h_mm", 1.0)))}
            for p in reg.get("pads", [])
            if int(p.get("net_code", -1)) == int(task.net_code) and len(p.get("layers") or []) >= n_layers]
        sites = via_sites_around(head, via_r, obst, [layer, other], toward, radii=self.VIA_SITE_RADII_MM, n=5)
        blockers = []
        for p in reg.get("pads", []):
            if int(p.get("net_code", -1)) == int(task.net_code):
                continue
            reach = near_mm + max(float(p.get("w_mm", 0)), float(p.get("h_mm", 0))) / 2
            if math.hypot(float(p["x_mm"]) - hx, float(p["y_mm"]) - hy) <= reach:
                blockers.append({"id": p["id"], "kind": "pad", "net_code": p["net_code"], "source": "head_region",
                                 "x_mm": round(float(p["x_mm"]), 3), "y_mm": round(float(p["y_mm"]), 3),
                                 "layers": list(p.get("layers") or [])})
        for t in reg.get("tracks", []):
            if int(t.get("net_code", -1)) == int(task.net_code):
                continue
            x1, y1, x2, y2 = t["seg"]
            if _pt_seg_dist(hx, hy, x1, y1, x2, y2) <= near_mm:
                blockers.append({"id": f"track@{x1:.2f},{y1:.2f}", "kind": "track", "net_code": t["net_code"],
                                 "source": "head_region", "layer": t.get("layer"),
                                 "seg": [round(float(v), 3) for v in (x1, y1, x2, y2)]})
        alt = self._nearby_starts(task, stage, layer, {(round(hx, 3), round(hy, 3))}, toward)[:5]
        return sites, blockers[:12], self._is_thru(task, head), alt

    def _deviations(self, task: DetailedSubtask) -> list[dict]:
        """Soft-guide deviations: committed segments of the net with an endpoint outside every
        corridor on their layer. Recorded, never treated as a DRC error."""
        info = self._call("inspect_net", net_code=task.net_code)
        out = []
        for seg in info.get("tracks", []):
            x1, y1, x2, y2, layer = seg
            boxes = [s.bbox for s in task.stages if s.layer == layer]
            for (x, y) in ((x1, y1), (x2, y2)):
                if boxes and not any(contains(b, x, y, tol=0.05) for b in boxes):
                    out.append({"layer": layer, "point": [round(x, 3), round(y, 3)]})
                    break
        return out

    # -- main -----------------------------------------------------------------
    def run_task(self, task: DetailedSubtask, budget: Budget) -> TaskResult:
        gw = self.gw
        steps: list[StepRecord] = []
        attempts: list[Attempt] = []
        calls0 = gw.adapter.log.calls
        if not task.execution.get("supported", True):
            fb = self._packet(task, "unsupported", [0, 0, -1], attempts, task.execution.get("reason", ""), "", "alternate_layer")
            return TaskResult(task.task_id, task.net_code, "unsupported", steps, fb, 0, [], 0)

        self._scope_before = self._frozen_signature()
        if self._edge_joined(task):
            # the two clusters are already one piece of copper (another edge of the net or a rip-up
            # re-route joined them): nothing to route, the edge is done. Before this the engine
            # answered "no ratsnest edges for this net" and the edge was reported failed and carried
            # through batch rounds as a phantom (MOD-MPU9150 n2_e9: four rounds, Global note "redundant")
            a0 = task.source_anchors[0]
            fb = self._packet(task, "done", [a0["x_mm"], a0["y_mm"], -1], attempts, "already joined",
                              task.stages[0].corridor_id, "", stage_ref=task.stages[0].corridor_id)
            return TaskResult(task.task_id, task.net_code, "done", steps, fb, self._groups(task), [], gw.adapter.log.calls - calls0)
        prof = task.profile
        if "track_width_mm" in prof:
            self._call("apply_profile", name=prof.get("name", "Default"))
        r = self._call("select_net", net_code=task.net_code)
        if not r.get("ok"):
            fb = self._packet(task, "blocked", [0, 0, -1], attempts, r.get("reason", "net not selectable"), task.stages[0].corridor_id, "")
            return TaskResult(task.task_id, task.net_code, "blocked", steps, fb, 0, [], gw.adapter.log.calls - calls0)

        stage0 = task.stages[0]
        first_portal = stage0.exit or (task.stages[1].entry if len(task.stages) > 1 else None)
        dst_hint = self._anchor(task, task.target_anchors, task.stages[-1]) or task.target_anchors[0]
        dst_xy = [dst_hint["x_mm"], dst_hint["y_mm"]]
        src = self._anchor(task, task.source_anchors, stage0, toward=first_portal or dst_xy)
        if src is None:
            fb = self._packet(task, "blocked", [0, 0, -1], attempts, "no source anchor inside first stage", stage0.corridor_id, "widen")
            self._call("end_net")
            return TaskResult(task.task_id, task.net_code, "blocked", steps, fb, 0, [], gw.adapter.log.calls - calls0)
        head = [src["x_mm"], src["y_mm"]]
        layer = stage0.layer
        via_r = float(prof.get("via_diameter_mm", 0.6)) / 2 + float(prof.get("clearance_mm", 0.2))
        trk_r = float(prof.get("track_width_mm", 0.2)) / 2 + float(prof.get("clearance_mm", 0.2))
        tr_by_from = {t.from_stage: t for t in task.transitions}
        from route_agent.guide.geometry import intersect
        nb = len(task.stages) - 1                       # boundaries between consecutive stages
        heads: list = [None] * (nb + 1)
        layers_at: list = [None] * (nb + 1)
        heads[0], layers_at[0] = [head[0], head[1]], layer
        ptr = [0] * nb                                 # next candidate index per boundary
        cands: list = [None] * nb                      # candidates per boundary, computed on first visit
        spec: list = [None] * nb                       # (kind, region_bbox, region_ref, keep)
        subs: list = [None] * nb                       # sub-checkpoint: board before boundary i's commit
        backtracks = 0
        mazes = 0
        anchor_switches = 0
        tried_starts = {(round(head[0], 3), round(head[1], 3))}
        self._backtracks = 0
        self._mazes = 0
        self._anchor_switches = 0
        failed = None                                  # (i, kind, region_ref, stage_id, reason)
        last = task.stages[-1]
        ok = False
        last_reason = ""
        i = 0
        while i <= nb:
            stage = task.stages[i]
            head, layer = heads[i], layers_at[i]
            if i == nb:
                # final stage: PNS finish, then an explicit route_to of the target anchor
                ok = self._try_finish(head, layer, stage, attempts, steps)
                if ok and not self._edge_joined(task):
                    self._release_subs(subs)
                    return self._progress(task, head, layer, stage, attempts, steps, calls0)
                last_reason = "" if ok else "finish failed"
                if not ok:
                    dst = self._anchor(task, task.target_anchors, stage) or task.target_anchors[0]
                    ok, last_reason, _landed = self._commit("route_to", head, layer, [dst["x_mm"], dst["y_mm"]],
                                                            stage.corridor_id, stage.corridor_id, attempts, steps)
                    if not ok and mazes < self.maze and self._boxed_in(last_reason):
                        mazes += 1
                        self._mazes = mazes
                        ok, _landed = self._maze_commit(task, head, layer, "route_to", [dst["x_mm"], dst["y_mm"]], stage.bbox,
                                                        stage.corridor_id, stage.corridor_id, trk_r, attempts, steps)
                if ok:
                    break
                if backtracks < self.backtrack:
                    j = self._backtrack_to(i, subs, ptr, cands, task, budget)
                    if j >= 0:
                        backtracks += 1
                        i = j
                        continue
                if i == 0 and anchor_switches < self.alt_anchors and self._boxed_in(last_reason):
                    alts = self._nearby_starts(task, stage, layer, tried_starts, dst_xy)
                    if alts:                                  # restart from other copper of the source cluster
                        anchor_switches += 1
                        self._anchor_switches = anchor_switches
                        tried_starts.add((round(alts[0][0], 3), round(alts[0][1], 3)))
                        attempts.append(self._head_attempt("switch_anchor", alts[0], True, 0, "", stage.corridor_id))
                        heads[0] = [alts[0][0], alts[0][1]]
                        continue
                failed = (i, "finish", stage.corridor_id, stage.corridor_id, last_reason or "target not reached")
                break
            nxt = task.stages[i + 1]
            if cands[i] is None:
                # first visit of this boundary: let PNS try to close, else compute candidates
                if self.finish_first and self._try_finish(head, layer, stage, attempts, steps):
                    if self._edge_joined(task):
                        ok = True
                        break
                    # PNS closed to another cluster of the same net: legal copper, not this edge.
                    # Report progress; the orchestrator keeps the copper and re-runs the task.
                    self._release_subs(subs)
                    return self._progress(task, head, layer, stage, attempts, steps, calls0)
                tr = tr_by_from.get(stage.corridor_id)
                if tr is not None:
                    spec[i] = ("route_via", tr.bbox, tr.via_region_id, via_r)
                    layers_needed = [layer, nxt.layer]             # a via must be clear on both faces
                else:
                    spec[i] = ("route_to", intersect(stage.bbox, nxt.bbox) or nxt.bbox,
                               f"{stage.corridor_id}->{nxt.corridor_id}", trk_r)
                    layers_needed = [layer]
                hint = nxt.exit if nxt.exit else dst_xy          # where the walk goes after this boundary
                cands[i], _n_obst = self._candidates(task, spec[i][1], spec[i][3], layers_needed, head, hint)
                ptr[i] = 0
                subs[i] = self._local_checkpoint() if self.backtrack > 0 else None
            kind, region_bbox, region_ref, keep = spec[i]
            ok = False
            last_reason = ""
            while ptr[i] < len(cands[i]):
                pt = cands[i][ptr[i]]
                ok, last_reason, landed = self._commit(kind, head, layer, pt, region_ref, stage.corridor_id, attempts, steps)
                if ok:
                    heads[i + 1] = landed
                    layers_at[i + 1] = nxt.layer if kind == "route_via" else layer
                    break
                ptr[i] += 1
                if last_reason.startswith("start_route") or last_reason in ("no net selected", "needs an idle session (cancel/finish first)"):
                    break
                if not budget.can_retry(task.task_id):
                    break
                budget.use_retry(task.task_id)
            if not ok and mazes < self.maze and cands[i] and self._boxed_in(last_reason):
                mazes += 1
                self._mazes = mazes
                ok, landed = self._maze_commit(task, head, layer, kind, cands[i][0], region_bbox, region_ref,
                                               stage.corridor_id, trk_r, attempts, steps)
                if ok:
                    heads[i + 1] = landed
                    layers_at[i + 1] = nxt.layer if kind == "route_via" else layer
            if ok:
                i += 1
                continue
            if backtracks < self.backtrack:
                j = self._backtrack_to(i, subs, ptr, cands, task, budget)
                if j >= 0:
                    backtracks += 1
                    i = j
                    continue
            if i == 0 and anchor_switches < self.alt_anchors and self._boxed_in(last_reason):
                alts = self._nearby_starts(task, stage, layer, tried_starts, dst_xy)
                if alts:                                      # restart from other copper of the source cluster
                    anchor_switches += 1
                    self._anchor_switches = anchor_switches
                    tried_starts.add((round(alts[0][0], 3), round(alts[0][1], 3)))
                    attempts.append(self._head_attempt("switch_anchor", alts[0], True, 0, "", stage.corridor_id))
                    heads[0] = [alts[0][0], alts[0][1]]
                    self._release_one(subs[i])
                    subs[i], cands[i] = None, None             # re-walk this boundary from the new head
                    continue
            failed = (i, kind, region_ref, stage.corridor_id, last_reason)
            break
        self._release_subs(subs)
        self._backtracks = backtracks
        if failed is not None:
            i, kind, region_ref, stage_id, reason = failed
            head, layer = heads[i], layers_at[i]
            hl = gw.adapter.layers.human(layer)
            if kind == "finish":
                outcome = "partial" if any(a.ok for a in attempts) else "blocked"
                request = "widen"
            else:
                outcome, request = "blocked", ("alternate_layer" if kind == "route_via" else "widen")
            sites, head_blockers, thru, alt = [], [], False, []
            if self._boxed_in(reason):
                # the head could not leave this spot on its layer: report where a via would fit,
                # what fences the head in and where else the net's copper is, and ask for a layer
                # change. The Detailed never places that via itself.
                sites, head_blockers, thru, alt = self._boxed_evidence(task, head, layer, task.stages[i], via_r, dst_xy)
                if sites or thru:
                    request = "alternate_layer"
            groups = self._groups(task)
            dev = self._deviations(task)
            self._call("end_net")
            fb = self._packet(task, outcome, [head[0], head[1], hl], attempts, reason, region_ref, request,
                              stage_ref=stage_id, deviations=dev, via_sites=sites, extra_blockers=head_blockers,
                              head_is_thru=thru, alt_starts=alt)
            return TaskResult(task.task_id, task.net_code, outcome, steps, fb, groups, dev, gw.adapter.log.calls - calls0)
        head, layer = heads[i] if i <= nb and heads[i] is not None else head, layers_at[i] if i <= nb and layers_at[i] is not None else layer
        groups = self._groups(task)
        joined = self._edge_joined(task)
        dev = self._deviations(task)
        self._call("end_net")
        hl = gw.adapter.layers.human(layer)
        if joined:
            fb = self._packet(task, "done", [head[0], head[1], hl], attempts, "", last.corridor_id, "", stage_ref=last.corridor_id, deviations=dev)
            return TaskResult(task.task_id, task.net_code, "done", steps, fb, groups, dev, gw.adapter.log.calls - calls0)
        fb = self._packet(task, "partial" if any(a.ok for a in attempts) else "blocked", [head[0], head[1], hl], attempts,
                          last_reason or "target not reached", last.corridor_id, "widen", stage_ref=last.corridor_id, deviations=dev)
        return TaskResult(task.task_id, task.net_code, fb.outcome, steps, fb, groups, dev, gw.adapter.log.calls - calls0)

    def _progress(self, task, head, layer, stage, attempts, steps, calls0) -> TaskResult:
        """``finish`` joined the source to a cluster of the net other than the target: legal, useful
        copper that must not be rolled back. Outcome ``progress`` -> router ``continue``."""
        merged = self._merged_points(task)
        groups = self._groups(task)
        dev = self._deviations(task)
        self._call("end_net")
        hl = self.gw.adapter.layers.human(layer)
        fb = self._packet(task, "progress", [head[0], head[1], hl], attempts, "finish_joined_other_cluster",
                          stage.corridor_id, "", stage_ref=stage.corridor_id, deviations=dev)
        fb.connectivity["merged_points"] = merged
        return TaskResult(task.task_id, task.net_code, "progress", steps, fb, groups, dev, self.gw.adapter.log.calls - calls0)

    def _groups(self, task) -> int:
        return int(self._call("inspect_net", net_code=task.net_code).get("pad_groups", 0))

    def _edge_joined(self, task: DetailedSubtask, tol: float = 1e-3) -> bool:
        """Edge completion = a source anchor and a target anchor sit in ONE committed connectivity
        cluster (engine ``get_connected_points``). ``pad_groups == 1`` is the wrong test for a net
        whose connection tree has several edges: the first edge of a 3-cluster net is done while
        the net still has two groups."""
        targets = [(float(a["x_mm"]), float(a["y_mm"])) for a in task.target_anchors]
        for a in task.source_anchors:
            layer = a["layers"][0] if a.get("layers") else task.stages[0].layer
            r = self._call("connected_points", x=a["x_mm"], y=a["y_mm"], layer=layer)
            for pt in r.get("points", []):
                if any(abs(pt[0] - tx) <= tol and abs(pt[1] - ty) <= tol for tx, ty in targets):
                    return True
        return False

    def _packet(self, task, outcome, head, attempts, reason, region_ref, request, stage_ref="", deviations=None,
                via_sites=None, extra_blockers=None, head_is_thru=False, alt_starts=None) -> FeedbackPacket:
        info = self._call("inspect_net", net_code=task.net_code)
        blockers = list(extra_blockers or [])
        if region_ref and outcome in ("blocked", "partial"):
            # evidence: foreign copper / pads inside the region we failed in and the stage we were
            # in (source: inspect_region). A same-layer portal "a->b" is the overlap of a and b.
            by_id = {s.corridor_id: s for s in task.stages}
            boxes = []
            if "->" in region_ref:
                a, b = region_ref.split("->", 1)
                if a in by_id and b in by_id:
                    from route_agent.guide.geometry import intersect
                    boxes.append(intersect(by_id[a].bbox, by_id[b].bbox) or by_id[b].bbox)
            elif region_ref in by_id:
                boxes.append(by_id[region_ref].bbox)
            else:
                tb = next((t.bbox for t in task.transitions if t.via_region_id == region_ref), None)
                if tb is not None:
                    boxes.append(tb)
            if stage_ref in by_id and by_id[stage_ref].bbox not in boxes:
                boxes.append(by_id[stage_ref].bbox)
            seen = {b["id"] for b in blockers}
            for bbox in boxes:
                reg = self._call("inspect_region", bbox=bbox)
                for p in reg.get("pads", []):
                    if p["net_code"] != task.net_code and p["id"] not in seen:
                        seen.add(p["id"])
                        blockers.append({"id": p["id"], "kind": "pad", "net_code": p["net_code"], "source": "inspect_region"})
                for t in reg.get("tracks", []):
                    tid = f"track@{t['seg'][0]:.2f},{t['seg'][1]:.2f}"
                    if t["net_code"] != task.net_code and tid not in seen:
                        seen.add(tid)
                        blockers.append({"id": tid, "kind": "track", "net_code": t["net_code"], "source": "inspect_region"})
        # real checks: the connectivity above was read at the adapter's current revision, and the
        # frozen-scope nets carry the same copper as at task start (shove may not move them)
        revision = self.gw.adapter.state.revision
        scope_ok = self._frozen_signature() == getattr(self, "_scope_before", {})
        return FeedbackPacket(
            snapshot_ref=self.snapshot_id, guide_id=self.guide_id, edge_id=task.task_id, net_code=task.net_code,
            outcome=outcome, last_head=[round(head[0], 4), round(head[1], 4), head[2]], attempts=attempts,
            connectivity={"pad_groups": info.get("pad_groups"), "remaining_edges": len(info.get("remaining_edges", [])),
                          "tracks": len(info.get("tracks", [])), "vias": len(info.get("vias", []))},
            failure_evidence=FailureEvidence(native_reason=reason or ("" if outcome in ("done", "progress") else "unknown"),
                                             blocker_candidates=blockers[:12], region_ref=region_ref,
                                             via_sites=[[round(float(p[0]), 3), round(float(p[1]), 3)] for p in (via_sites or [])],
                                             head_is_thru=bool(head_is_thru), alt_starts=list(alt_starts or [])),
            checks={"fresh": int(info.get("revision", revision)) == revision, "revision": revision, "scope_ok": scope_ok,
                    "backtracks": int(getattr(self, "_backtracks", 0)), "mazes": int(getattr(self, "_mazes", 0)),
                    "anchor_switches": int(getattr(self, "_anchor_switches", 0))},
            request=request, stage_ref=stage_ref, deviations=deviations or [], cost=sum(a.cost for a in attempts))


# --------------------------------------------------------------------------- orchestrator
@dataclass
class LoopEvent:
    t: float
    kind: str
    detail: dict = field(default_factory=dict)


@dataclass
class LoopReport:
    tasks: list[dict] = field(default_factory=list)
    events: list[LoopEvent] = field(default_factory=list)
    replans: int = 0
    stopped: str = ""
    failed: list[str] = field(default_factory=list)     # edges given up on (on_unroutable="skip")
    budget: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)


class Orchestrator:
    """Event-driven loop. ``replan_fn(packet, decision) -> RouteGuidePlan | None`` is the Global
    provider hook (it applies the provider's patch or validates its full plan and returns the
    next plan); ``recompile_fn(plan) -> list[DetailedSubtask]`` re-compiles the edges to run
    again. Every task runs on its own engine checkpoint, which is released once the task is
    accepted or rolled back; recovery calls go through the budget's reserve."""

    def __init__(self, gateway: ToolGateway, detailed: ScriptedDetailed, budget: Budget,
                 stall_threshold: int = 2, replan_fn: Callable | None = None,
                 recompile_fn: Callable | None = None, target_net_names: list[str] | None = None,
                 on_unroutable: str = "stop", ripup_cap: int = 1, rippable_nets: set[int] | None = None,
                 replans_per_edge: int = 4, batch_replan: bool = False, batch_rounds: int = 5,
                 edge_history: bool = True) -> None:
        """``on_unroutable``: what to do when the router asks for a replan and there is no Global
        provider — ``stop`` the run (default, closed-loop semantics) or ``skip`` the edge after
        rolling it back and go on with the next task (baseline / benchmark semantics).

        ``ripup_cap``: before retry / replan / skip of a failed edge, rip up the work-set net whose
        tracks dominate the failure evidence (never a frozen net, at most ``ripup_cap`` times per
        net), re-run the edge, and on success re-queue the victim's edges; on failure restore the
        board (victim copper included). ``rippable_nets`` limits victims (default: any net the
        evidence names). 0 disables.

        ``replans_per_edge``: Global replans spent on one edge before the loop gives it up (skip
        when ``on_unroutable="skip"``, else stop) instead of burning the whole replan cap on it."""
        if on_unroutable not in ("stop", "skip"):
            raise ValueError("on_unroutable must be 'stop' or 'skip'")
        self.gw = gateway
        self.detailed = detailed
        self.budget = budget
        self.stalls = StallTracker(stall_threshold)
        self.replan_fn = replan_fn
        self.recompile_fn = recompile_fn
        self.target_net_names = target_net_names
        self.on_unroutable = on_unroutable
        self.ripup_cap = int(ripup_cap)
        self.rippable_nets = set(int(n) for n in rippable_nets) if rippable_nets is not None else None
        self.report = LoopReport()
        self._continues: dict[str, int] = {}
        self._rips: dict[int, int] = {}
        self._tasks_by_net: dict[int, list] = {}
        self.replans_per_edge = int(replans_per_edge)
        self._edge_replans: dict[str, int] = {}
        # batch mode: replans are not asked one edge at a time but deferred; when the queue is
        # empty, ONE Global call carries every deferred edge's feedback (packets list) and may
        # return a patch over several edges; up to batch_rounds such passes
        self.batch_replan = bool(batch_replan)
        self.batch_rounds = int(batch_rounds)
        self._pending: dict[str, tuple] = {}
        self._batch_rounds_used = 0
        # per edge, the rounds the Global already saw: plan shape executed + where it died
        self.edge_history = bool(edge_history)
        self._edge_history: dict[str, list[dict]] = {}
        self._history_seen: set[int] = set()

    def _ev(self, kind, **detail):
        self.report.events.append(LoopEvent(time.time(), kind, detail))

    def _release(self, handle) -> None:
        """Free the checkpoint clone (board + DRC state) held in the C++ store."""
        if handle is None:
            return
        try:
            self.gw.adapter.release_checkpoint(handle)
        except Exception as ex:  # noqa: BLE001 - bookkeeping must never abort the loop
            self._ev("release_failed", handle=handle, error=str(ex))

    def _rollback(self, handle) -> bool:
        """Restore the stable snapshot through the recovery reserve, then free the clone."""
        if self.gw.adapter.e.is_routing():
            self.gw.call_recovery("cancel")
        r = self.gw.call_recovery("rollback", handle=handle)
        ok = bool(r.get("ok"))
        self._ev("rollback", handle=handle, ok=ok, reason=r.get("reason", ""))
        self._release(handle)
        return ok

    def _stop(self, reason: str, **detail) -> None:
        self.report.stopped = reason
        self._ev("stop", reason=reason, **detail)

    def _index_tasks(self, tasks) -> None:
        for t in tasks:
            lst = self._tasks_by_net.setdefault(int(t.net_code), [])
            if all(x.task_id != t.task_id for x in lst):
                lst.append(t)

    def _pick_victim(self, task, fb: FeedbackPacket) -> int | None:
        """The other net whose tracks dominate the failure evidence and may still be ripped up.

        Only the region-level evidence (``source`` ``inspect_region``: foreign tracks inside the
        region the walk failed in) counts, as before. The head-region fence tracks added for the
        Global (``source`` ``head_region``) are NOT victims: picking them ripped nets routed earlier
        that could not be re-routed and then ate the replan budget (D3-B Claude run, 3 boards:
        rip-ups 16 -> 21, done-then-lost edges 4 -> 12, Global calls 10 -> 17)."""
        counts: dict[int, int] = {}
        for b in fb.failure_evidence.blocker_candidates:
            if b.get("kind") != "track" or b.get("source") == "head_region":
                continue
            n = int(b.get("net_code", -1))
            if n < 0 or n == int(task.net_code) or n in set(int(f) for f in self.gw.frozen_scope):
                continue
            if self.rippable_nets is not None and n not in self.rippable_nets:
                continue
            if self._rips.get(n, 0) >= self.ripup_cap:
                continue
            counts[n] = counts.get(n, 0) + 1
        return max(counts, key=lambda n: (counts[n], -n)) if counts else None

    def _ripup_and_retry(self, task, victim: int, queue: list, done: set) -> tuple[dict | None, bool]:
        """Rip ``victim`` up, re-run ``task``. Returns (record, helped). On success the victim's
        edges leave ``done`` and go to the front of the queue; otherwise the board is restored."""
        ck = self.gw.call("checkpoint")
        handle = ck.get("handle")
        if handle is None:
            return None, False
        self._rips[victim] = self._rips.get(victim, 0) + 1
        d = self.gw.call("delete_net_routing", net_codes=[int(victim)])
        if not d.get("ok"):
            self._rollback(handle)
            return None, False
        self._ev("ripup", task=task.task_id, victim=victim, removed=d.get("removed"))
        if _TRACE_NETS:
            self._ev("nets", after="ripup_delete", victim=victim, groups={str(n): self.gw.adapter.inspect_net(int(n))["pad_groups"] for n in sorted(self.rippable_nets or [])})
        try:
            res = self._run_task_guarded(task)
        except BudgetExhausted as ex:
            ok = self._rollback(handle)
            self._stop(str(ex), task=task.task_id)
            return {"task_id": task.task_id, "net_code": task.net_code, "outcome": "aborted", "decision": "stop",
                    "pad_groups": None, "calls": None, "steps": [], "deviations": [], "feedback": None,
                    "rolled_back": ok, "ripup": {"victim": victim, "removed": d.get("removed"), "helped": False},
                    "_stop": True}, False
        dec = route_feedback(res.feedback, self.budget, self.stalls)
        if _TRACE_NETS:
            self._ev("nets", after="ripup_retry", victim=victim, outcome=res.outcome, groups={str(n): self.gw.adapter.inspect_net(int(n))["pad_groups"] for n in sorted(self.rippable_nets or [])})
        rec = {"task_id": task.task_id, "net_code": task.net_code, "outcome": res.outcome, "decision": dec.kind,
               "pad_groups": res.pad_groups, "calls": res.calls, "steps": [s.__dict__ for s in res.steps],
               "deviations": res.deviations, "feedback": res.feedback.__dict__ if res.feedback else None,
               "ripup": {"victim": victim, "removed": d.get("removed"), "helped": dec.kind == "done"}}
        if dec.kind == "done":
            done.add(task.task_id)
            self._release(handle)
            victims = list(self._tasks_by_net.get(int(victim), []))
            ids = {vt.task_id for vt in victims}
            for tid in ids:
                done.discard(tid)
            queue[:] = victims + [t for t in queue if t.task_id not in ids]
            self._ev("ripup_requeue", victim=victim, edges=sorted(ids))
            return rec, True
        rec["rolled_back"] = self._rollback(handle)
        if _TRACE_NETS:
            self._ev("nets", after="ripup_rollback", victim=victim, groups={str(n): self.gw.adapter.inspect_net(int(n))["pad_groups"] for n in sorted(self.rippable_nets or [])})
        return rec, False

    def _net_groups(self, exclude_net: int) -> dict[int, int]:
        """Pad groups of every other work-set net (adapter read, not a gateway call).

        One connectivity rebuild serves every net: the engine reports the pad groups of all nets
        at once, while ``inspect_net`` rebuilds the whole board's connectivity for each one. This
        check runs before and after every task, so per-net reads cost 2 x nets rebuilds a task
        (on a board with 163 nets at ~0.1 s each: about 40 s a task, hours a board)."""
        e = self.gw.adapter.e
        e.build_connectivity()
        groups = e.get_pad_groups()
        return {int(n): int(groups.get(int(n), 0))
                for n in sorted(self.rippable_nets or []) if int(n) != int(exclude_net)}

    def _run_task_guarded(self, task) -> TaskResult:
        """``run_task`` plus a check that no OTHER work-set net lost connectivity meanwhile.

        PNS shove (the Detailed's fallback mode) can push a routed track of another net off its
        pad. Measured on sensorboard (no model, deterministic): the shove ``finish`` of n2_e5 split
        net 9, done long before, into two pad groups; nothing re-checked it, the loop reported
        every edge done and the official count came out one edge short. Here the pad groups of the
        other work-set nets are read before and after the task; when one of them grew, the attempt
        is rolled back and the task runs once more with the shove fallback off (walkaround does not
        move other copper). If even that disturbs a net the attempt is rolled back and reported
        blocked, so the router treats it like any other failure."""
        if not self.rippable_nets or not getattr(self.detailed, "shove_fallback", False):
            return self.detailed.run_task(task, self.budget)
        before = self._net_groups(task.net_code)
        ck = self.gw.call("checkpoint")
        handle = ck.get("handle")
        res = self.detailed.run_task(task, self.budget)
        if handle is None:
            return res
        hurt = [n for n, g in self._net_groups(task.net_code).items() if g > before.get(n, g)]
        if not hurt:
            self._release(handle)
            return res
        self._ev("disturbed", task=task.task_id, nets=hurt, outcome=res.outcome, retry="no shove")
        self._rollback(handle)
        ck2 = self.gw.call("checkpoint")
        handle2 = ck2.get("handle")
        self.detailed.shove_fallback = False
        try:
            res = self.detailed.run_task(task, self.budget)
        finally:
            self.detailed.shove_fallback = True
        hurt2 = [n for n, g in self._net_groups(task.net_code).items() if g > before.get(n, g)]
        if handle2 is None:
            return res
        if not hurt2:
            self._release(handle2)
            return res
        self._ev("disturbed", task=task.task_id, nets=hurt2, outcome=res.outcome, retry="given up")
        self._rollback(handle2)
        res.outcome = "blocked"
        if res.feedback is not None:
            res.feedback.outcome = "blocked"
            res.feedback.request = res.feedback.request or "widen"
            res.feedback.failure_evidence.native_reason = f"disturbed other nets {hurt2}; attempt rolled back"
        return res

    def _with_history(self, task, pkt: FeedbackPacket) -> FeedbackPacket:
        """Attach this edge's earlier rounds to the packet the Global is about to see, then record
        the present one. An entry is the shape the Detailed executed (stages with layer and bbox,
        via regions) and where it died. Without it the Global sees only the current plan and the
        latest failure, and re-sends near copies of a shape that already failed (measured: four
        rounds of the same F.Cu entry into one fenced pad). A packet handed over twice (the batch
        asked again after an unusable answer) is recorded once."""
        if not self.edge_history:
            return pkt
        hist = self._edge_history.setdefault(task.task_id, [])
        pkt.history = [dict(h) for h in hist]
        if id(pkt) not in self._history_seen:
            self._history_seen.add(id(pkt))
            hist.append({"round": len(hist) + 1,
                         "stages": [{"corridor": s.corridor_id, "layer": s.layer, "bbox": list(s.bbox)} for s in task.stages],
                         "via_regions": [t.via_region_id for t in task.transitions],
                         "died_at": pkt.stage_ref or pkt.failure_evidence.region_ref, "head": list(pkt.last_head),
                         "reason": pkt.failure_evidence.native_reason, "request": pkt.request})
        return pkt

    def _flush_batch(self, done: set) -> list:
        """Batch mode, queue empty: one Global call for every edge deferred in this pass. The
        replan_fn receives the LIST of their feedback packets and a decision with request
        ``batch``; the patch it returns may cover several edges. Returns the recompiled tasks to
        run next, [] when nothing more is to be done. Edges the patch did not touch, and every
        pending edge once the rounds or the replan cap are spent, are skipped (failed)."""
        if not self.batch_replan or not self._pending or self.report.stopped:
            return []
        pending, self._pending = dict(self._pending), {}
        if self.replan_fn is None or self._batch_rounds_used >= self.batch_rounds or not self.budget.can_replan():
            for tid, (_t, _p, d) in pending.items():
                self.report.failed.append(tid)
                self._ev("skip", task=tid, reason="batch replan rounds or replan cap exhausted", request=d.request)
            return []
        self._batch_rounds_used += 1
        packets = [self._with_history(t, p) for (t, p, _d) in pending.values()]
        dec = Decision("replan", f"end-of-pass batch: {len(packets)} failed edge(s)", request="batch",
                       edge_id=",".join(pending))
        n = self.budget.use_replan(dec.reason)
        self.report.replans = n
        for tid in pending:
            self._edge_replans[tid] = self._edge_replans.get(tid, 0) + 1
        new_plan = self.replan_fn(packets, dec)
        if new_plan is None:
            if self._batch_rounds_used < self.batch_rounds and self.budget.can_replan():
                # the answer failed its checks even after the repair round (about 1 patch in 20):
                # the edges keep their place in the batch and the next round asks again — giving
                # up here once threw away 11 edges of one board with 10 replans still unspent
                self._pending = pending
                for tid, (_t, _p, d) in pending.items():
                    self._ev("defer_replan", task=tid, request=d.request,
                             reason="Global returned no usable plan / patch; asked again in the next round")
                return self._flush_batch(done)
            for tid, (_t, _p, d) in pending.items():
                self.report.failed.append(tid)
                self._ev("skip", task=tid, reason="Global returned no usable plan / patch", request=d.request)
            return []
        new_tasks = self.recompile_fn(new_plan) if self.recompile_fn else []
        self._index_tasks(new_tasks)
        ids = {t.task_id for t in new_tasks}
        self._ev("batch_replan", n=n, round=self._batch_rounds_used, deferred=sorted(pending), edges=sorted(ids & set(pending)))
        for tid, (t, p, d) in pending.items():
            if tid not in ids:
                # the Global left this edge out of its patch: it stays in the batch for the next round
                # (measured: a 6-packet batch came back with 2-4 edges patched, all of which routed)
                # and is given up only when the rounds or the cap are spent
                if self._batch_rounds_used < self.batch_rounds and self.budget.can_replan():
                    self._pending[tid] = (t, p, d)
                    self._ev("defer_replan", task=tid, request=d.request, reason="not addressed by the batch patch; kept for the next round")
                else:
                    self.report.failed.append(tid)
                    self._ev("skip", task=tid, reason="not addressed by the batch patch", request=d.request)
        return [t for t in new_tasks if t.task_id not in done]

    def run(self, tasks: list[DetailedSubtask]) -> LoopReport:
        queue = list(tasks)
        done: set[str] = set()
        self._index_tasks(tasks)
        while True:
            if not queue:
                queue = self._flush_batch(done)          # batch mode: one Global call for this pass's failures
                if not queue:
                    break
            task = queue.pop(0)
            if task.task_id in done:
                continue
            ck = self.gw.call("checkpoint")
            handle = ck.get("handle")
            if handle is None:
                self._stop(f"cannot take a checkpoint: {ck.get('reason', '')}", task=task.task_id)
                break
            self._ev("task_start", task=task.task_id, checkpoint=handle, revision=ck.get("revision"))
            if _TRACE_NETS:
                self._ev("nets", task=task.task_id, groups={str(n): self.gw.adapter.inspect_net(int(n))["pad_groups"] for n in sorted(self.rippable_nets or [])})
            try:
                res = self._run_task_guarded(task)
            except BudgetExhausted as ex:
                ok = self._rollback(handle)
                self.report.tasks.append({"task_id": task.task_id, "net_code": task.net_code, "outcome": "aborted",
                                          "decision": "stop", "pad_groups": None, "calls": None, "steps": [],
                                          "deviations": [], "feedback": None, "rolled_back": ok})
                self._stop(str(ex), task=task.task_id)
                break
            dec: Decision = route_feedback(res.feedback, self.budget, self.stalls)
            self._ev("feedback", task=task.task_id, outcome=res.outcome, decision=dec.kind, reason=dec.reason,
                     request=dec.request, calls=res.calls)
            rec = {"task_id": task.task_id, "net_code": task.net_code, "outcome": res.outcome, "decision": dec.kind,
                   "pad_groups": res.pad_groups, "calls": res.calls, "steps": [s.__dict__ for s in res.steps],
                   "deviations": res.deviations, "feedback": res.feedback.__dict__ if res.feedback else None}
            if dec.kind == "done":
                done.add(task.task_id)
                self._release(handle)
                self.report.tasks.append(rec)
                continue
            if dec.kind == "continue":
                # progress without completion: keep the copper and run the same task again,
                # bounded by local_retry_cap so an executor that only ever "progresses" ends
                n = self._continues[task.task_id] = self._continues.get(task.task_id, 0) + 1
                if n <= self.budget.local_retry_cap:
                    self._release(handle)
                    self.report.tasks.append(rec)
                    queue.insert(0, task)
                    continue
                dec = Decision("replan", f"{n} progress reports without completing {task.task_id}",
                               request="widen", edge_id=task.task_id)
                rec["decision"] = dec.kind
            # retry / replan / stop: back to the stable snapshot first
            rec["rolled_back"] = self._rollback(handle)
            self.report.tasks.append(rec)
            if not rec["rolled_back"]:
                self._stop("rollback failed; board state is not the stable snapshot", task=task.task_id)
                break
            if dec.kind in ("retry", "replan") and self.ripup_cap > 0 and res.feedback is not None:
                victim = self._pick_victim(task, res.feedback)
                if victim is not None:
                    rec2, helped = self._ripup_and_retry(task, victim, queue, done)
                    if rec2 is not None:
                        self.report.tasks.append(rec2)
                        if rec2.get("_stop"):
                            break
                        if helped:
                            continue
            if dec.kind == "retry":
                # the scripted executor already spent its local retries inside run_task; treat
                # a residual 'retry' as one more whole-task attempt on a clean checkpoint
                self.budget.use_retry(task.task_id)
                queue.insert(0, task)
                continue
            if dec.kind == "stop":
                if self.on_unroutable == "skip":
                    self.report.failed.append(task.task_id)
                    self._ev("skip", task=task.task_id, reason=dec.reason, request=dec.request)
                    continue
                self._stop(dec.reason, task=task.task_id)
                break
            if self.replan_fn is None:
                if self.on_unroutable == "skip":
                    self.report.failed.append(task.task_id)
                    self._ev("skip", task=task.task_id, reason=dec.reason, request=dec.request)
                    continue
                self._stop("replan requested but no Global provider", task=task.task_id)
                break
            if self._edge_replans.get(task.task_id, 0) >= self.replans_per_edge or not self.budget.can_replan():
                why = ("replan cap reached" if not self.budget.can_replan()
                       else f"{self.replans_per_edge} replans already spent on {task.task_id}")
                if self.on_unroutable == "skip":
                    self.report.failed.append(task.task_id)
                    self._ev("skip", task=task.task_id, reason=why, request=dec.request)
                    continue
                self._stop(why, task=task.task_id)
                break
            if self.batch_replan:
                # defer: the edge waits for the end-of-pass batch call instead of asking now
                self._pending[task.task_id] = (task, res.feedback, dec)
                self._ev("defer_replan", task=task.task_id, request=dec.request, reason=dec.reason)
                continue
            n = self.budget.use_replan(dec.reason)
            self._edge_replans[task.task_id] = self._edge_replans.get(task.task_id, 0) + 1
            self.report.replans = n
            new_plan = self.replan_fn(self._with_history(task, res.feedback), dec)
            if new_plan is None:
                if self.on_unroutable == "skip":
                    self.report.failed.append(task.task_id)
                    self._ev("skip", task=task.task_id, reason="Global returned no usable plan / patch", request=dec.request)
                    continue
                self._stop("Global provider returned no usable plan / patch")
                break
            new_tasks = self.recompile_fn(new_plan) if self.recompile_fn else []
            self._index_tasks(new_tasks)
            self._ev("replan", n=n, edges=[t.task_id for t in new_tasks])
            ids = {t.task_id for t in new_tasks}
            queue = [t for t in queue if t.task_id not in ids]
            queue = list(new_tasks) + queue
        drc = self.gw.call_recovery("run_drc", target_net_names=self.target_net_names)
        self.report.budget = self.budget.snapshot()
        self.report.failed = list(dict.fromkeys(self.report.failed))
        self.report.final = {"done": sorted(done), "failed": list(self.report.failed), "drc_scoped": drc.get("count"),
                             "drc_errors": drc.get("errors"), "drc_unconnected": drc.get("unconnected"),
                             "violations": drc.get("violations", [])[:50]}
        return self.report
