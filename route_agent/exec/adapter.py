"""Native adapter: the only PCBRouteAgent module that touches ``pcb_world`` engine objects.

* ``LayerMap`` — KiCad layer names <-> the engine's human ids (1 = F.Cu … N = B.Cu). The
  raw PCB_LAYER_ID enum (0 / 2 / 4 …) is a third domain and is only translated here.
* ``LazyEngine`` — constructs ``KiCadEngine`` on first use so contract tests, schema
  tooling and prompt building never load the GPL shared library.
* ``NativeAdapter`` — the facade the tool gateway and the scripted executor call. Every
  method is one "executor facade call" for budget purposes; aggregate getters may fan out
  into several native calls underneath (documented per method).

Routing primitives reuse ``pcb_world.core.action`` verbatim: PCBWorld's contract is that
``make_line`` / ``make_via`` commit the segment and *end* the session (force_finish), so a
Detailed step is always ``start_route(head) -> route_to | route_via -> (session idle)``.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcb_world.core import action as _act
from pcb_world.engine.drc import DRC_SEVERITY_ERROR
from pcb_world.engine.layer_mapping import LayerMapping

from route_agent.exec.waypoints import segment_hits_rect
from route_agent.guide.geometry import normalize

from route_agent.schema.snapshot import NetRule


# --------------------------------------------------------------------------- layers
class LayerMap:
    """KiCad copper-layer *names* on top of PCBWorld's ``LayerMapping``.

    ``LayerMapping`` (``pcb_world.engine.layer_mapping``; every KiCadEngine exposes its own as
    ``engine.layer_map``) owns the board PCB_LAYER_ID <-> human id (1 = top .. N = bottom)
    mapping. This class only adds the names (F.Cu, In1.Cu .., B.Cu) that every PCBRouteAgent
    contract uses, so the KiCad enum encoding lives in exactly one place.
    """

    def __init__(self, copper_layer_count: int, mapping: Any | None = None) -> None:
        n = int(copper_layer_count)
        if n < 2:
            raise ValueError("copper_layer_count must be >= 2")
        self.mapping = mapping if mapping is not None else LayerMapping(n)
        if int(self.mapping.max_layer) != n:
            raise ValueError(f"layer mapping has {self.mapping.max_layer} layers, board reports {n}")
        self.names = ["F.Cu"] + [f"In{i}.Cu" for i in range(1, n - 1)] + ["B.Cu"]
        self._board_to_name = {int(bl): self.names[self.mapping.board_to_human(bl) - 1]
                               for bl in self.mapping.board_layer_order}

    def human(self, name: str) -> int:
        try:
            return self.names.index(name) + 1
        except ValueError:
            raise KeyError(f"unknown copper layer {name!r}; have {self.names}") from None

    def name(self, human: int) -> str:
        if human < 1 or human > len(self.names):
            raise KeyError(f"human layer {human} out of range 1..{len(self.names)}")
        return self.names[human - 1]

    def board(self, name: str) -> int:
        """Name -> KiCad PCB_LAYER_ID through PCBWorld's mapping."""
        return int(self.mapping.human_to_board(self.human(name)))

    def name_from_board(self, board_layer: int) -> str | None:
        """KiCad PCB_LAYER_ID -> name; None for a layer that is not copper on this board."""
        return self._board_to_name.get(int(board_layer))

    def pad_layers(self, board_layer: int) -> list[str]:
        """Pad layer field: -2 marks a pad spanning every copper layer (thru-hole)."""
        if board_layer == -2:
            return list(self.names)
        n = self.name_from_board(board_layer)
        return [n] if n else []


# --------------------------------------------------------------------------- engine
def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_project_path(board_path: str, project_path: str | None = None) -> str:
    """Explicit path, else ``<stem>.kicad_pro``, else the one ``.kicad_pro`` in the folder
    (PCBench prepared boards ship ``processed_v9_guide_v3_unrouted.kicad_pcb`` next to
    ``processed_v9_guide_v3.kicad_pro``)."""
    if project_path:
        return str(project_path)
    bp = Path(board_path)
    sib = bp.with_suffix(".kicad_pro")
    if sib.exists():
        return str(sib)
    stem = bp.stem.replace("_unrouted", "")
    cand = bp.parent / f"{stem}.kicad_pro"
    if cand.exists():
        return str(cand)
    pros = sorted(bp.parent.glob("*.kicad_pro"))
    if len(pros) == 1:
        return str(pros[0])
    return ""


class LazyEngine:
    """Holds the board identity; builds the ``KiCadEngine`` on first ``.engine`` access."""

    def __init__(self, board_path: str, project_path: str | None = None, **engine_kwargs) -> None:
        self.board_path = str(board_path)
        self.project_path = resolve_project_path(self.board_path, project_path)
        self.engine_kwargs = engine_kwargs
        self._engine = None

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    @property
    def engine(self):
        if self._engine is None:
            from pcb_world.engine.kicad_engine import KiCadEngine
            self._engine = KiCadEngine(self.board_path, project_path=self.project_path or None,
                                       **self.engine_kwargs)
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None


# --------------------------------------------------------------------------- adapter
@dataclass
class StateRef:
    """Monotonic revision of the live board; every mutation bumps it. Tool requests carry
    the StateRef they were planned against so a stale request is refused, not executed."""
    revision: int = 0
    checkpoint: int | None = None


@dataclass
class CallLog:
    calls: int = 0
    by_name: dict = field(default_factory=dict)

    def tick(self, name: str) -> None:
        self.calls += 1
        self.by_name[name] = self.by_name.get(name, 0) + 1


CAPABILITIES = (
    # Capability manifest — what the pinned engine binding (PCBWorld-Engine 7a31e0c)
    # actually supports through this adapter. Anything not listed is "proposed", never
    # advertised as live just because the KiCad GUI has it.
    "select_net", "start_route", "route_to", "route_via", "finish", "cancel", "undo_last",
    "flip_posture", "end_net", "inspect_net", "inspect_region", "routing_state", "get_rules",
    "apply_profile", "set_via_dims", "set_corner_mode", "delete_net_routing", "lock_net", "cleanup",
    "checkpoint", "rollback", "run_drc", "save", "connected_points",
)
UNSUPPORTED = ("diff_pair", "length_tune", "skew_tune", "drag_preview", "polygon_corridor")


class NativeAdapter:
    """Facade over a (lazy) ``KiCadEngine`` with human-layer ids replaced by layer names."""

    def __init__(self, engine: Any, layer_map: LayerMap | None = None) -> None:
        self._eng = engine                      # LazyEngine, KiCadEngine or a test double
        self._lm = layer_map
        self.state = StateRef()
        self.log = CallLog()
        self.current_net: int | None = None
        self.routing_mode = 2                   # 2 = walkaround (PCBWorld default), 1 = shove

    # -- plumbing ------------------------------------------------------------
    @property
    def e(self):
        return self._eng.engine if isinstance(self._eng, LazyEngine) else self._eng

    @property
    def layers(self) -> LayerMap:
        if self._lm is None:
            e = self.e
            self._lm = LayerMap(e.get_copper_layer_count(), mapping=getattr(e, "layer_map", None))
        return self._lm

    def _mutated(self) -> None:
        self.state.revision += 1

    def capabilities(self) -> list[str]:
        return list(CAPABILITIES)

    # -- observe -------------------------------------------------------------
    def routing_state(self) -> dict:
        self.log.tick("routing_state")
        s = self.e.get_routing_session_state()
        head = list(s.route_head)
        lay = int(s.current_layer)
        return {
            "is_routing": bool(s.is_routing), "is_dragging": bool(s.is_dragging),
            "is_placing_via": bool(s.is_placing_via),
            "current_layer": self.layers.name(lay) if lay >= 1 else None,
            "head": [float(head[0]), float(head[1])] if s.is_routing else None,
            "current_net": self.current_net, "revision": self.state.revision,
            "mode": self.routing_mode,
        }

    def inspect_net(self, net_code: int) -> dict:
        """Aggregate getter: ratsnest + pad groups + copper of one net (3 native calls)."""
        self.log.tick("inspect_net")
        e = self.e
        e.build_connectivity()
        edges = [[r.x1_mm, r.y1_mm, r.x2_mm, r.y2_mm] for r in e.get_ratsnest() if r.net_code == net_code]
        groups = e.get_pad_groups().get(net_code, 0)
        tracks = [[t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm, self.layers.name_from_board(t.layer)]
                  for t in e.get_tracks() if t.net_code == net_code]
        vias = [[v.x_mm, v.y_mm] for v in e.get_vias() if v.net_code == net_code]
        return {"net_code": net_code, "remaining_edges": edges, "pad_groups": groups,
                "tracks": tracks, "vias": vias, "revision": self.state.revision}

    def inspect_region(self, bbox: list[float], layer: str | None = None) -> dict:
        """Copper + pads inside a bbox (optionally one layer). Pure query."""
        self.log.tick("inspect_region")
        x1, y1, x2, y2 = normalize(bbox)       # the validator accepts reversed corners; so do we
        e = self.e

        def inside(x, y):
            return x1 <= x <= x2 and y1 <= y <= y2
        pads, tracks, vias = [], [], []
        for p in e.get_pads():
            if inside(p.x_mm, p.y_mm) and (layer is None or layer in self.layers.pad_layers(p.layer)):
                pads.append({"id": f"{p.footprint_ref}:{p.pad_name}", "net_code": p.net_code,
                             "x_mm": p.x_mm, "y_mm": p.y_mm, "pad_type": p.pad_type,
                             "w_mm": float(p.width_mm), "h_mm": float(p.height_mm),
                             "layers": self.layers.pad_layers(p.layer)})
        for t in e.get_tracks():
            ln = self.layers.name_from_board(t.layer)
            if (layer is None or ln == layer) and segment_hits_rect(t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm, [x1, y1, x2, y2]):
                tracks.append({"net_code": t.net_code, "layer": ln,
                               "seg": [t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm], "width_mm": t.width_mm})
        for v in e.get_vias():
            if inside(v.x_mm, v.y_mm):
                vias.append({"net_code": v.net_code, "x_mm": v.x_mm, "y_mm": v.y_mm, "diameter_mm": v.diameter_mm})
        return {"bbox": list(bbox), "layer": layer, "pads": pads, "tracks": tracks, "vias": vias}

    def connected_points(self, x: float, y: float, layer: str) -> list[list]:
        self.log.tick("connected_points")
        return [[px, py, self.layers.name(pl)] for px, py, pl in
                self.e.get_connected_points(x, y, self.layers.human(layer))]

    def get_rules(self, net_code: int | None = None) -> dict:
        self.log.tick("get_rules")
        e = self.e
        r = e.get_design_rules()
        out = {"min_track_width_mm": r.min_track_width_mm, "min_clearance_mm": r.min_clearance_mm,
               "min_via_diameter_mm": r.min_via_diameter_mm,
               "copper_edge_clearance_mm": r.copper_edge_clearance_mm,
               "profiles": {nc.name: _nc_dims(nc) for nc in [r.default_netclass, *r.netclasses]}}
        if net_code is not None:
            nc = e.get_netclass_for_net(net_code)
            out["net_class"] = {"name": nc.name, **_nc_dims(nc)}
        return out

    def net_rule(self, net_code: int, net_name: str) -> NetRule:
        nc = self.e.get_netclass_for_net(net_code)
        d = _nc_dims(nc)
        return NetRule(net_code=net_code, net_name=net_name, class_name=nc.name,
                       track_width_mm=d["track_width_mm"], clearance_mm=d["clearance_mm"],
                       via_diameter_mm=d["via_diameter_mm"], via_drill_mm=d["via_drill_mm"],
                       layers=list(self.layers.names))

    # -- route ---------------------------------------------------------------
    def select_net(self, net_code: int) -> dict:
        self.log.tick("select_net")
        ok, info = _act.net_select(self.e, int(net_code))
        if ok:
            self.current_net = int(net_code)
        return {"ok": ok, **info}

    def start_route(self, x: float, y: float, layer: str) -> dict:
        self.log.tick("start_route")
        if self.current_net is None:
            return {"ok": False, "reason": "no net selected"}
        ok, info = _act.start_route(self.e, float(x), float(y), self.layers.human(layer))
        if ok:
            self._mutated()
        info["layer"] = layer
        return {"ok": ok, **info}

    def route_to(self, x: float, y: float, mode: int | None = None) -> dict:
        """make_line: commit a walkaround/shove segment to (x, y) on the current layer; the
        session ends on success (force_finish) and is cancelled on failure."""
        self.log.tick("route_to")
        e = self.e
        if not e.is_routing():
            return {"ok": False, "reason": "no active routing session"}
        ok, info = _act.make_line(e, float(x), float(y), self._mode(mode))
        if not ok and e.is_routing():
            e.cancel_route()
        self._mutated()
        # make_line fails for exactly two reasons: the pre-check rejected the target (reported in
        # ``rejected``) or PNS could not reach it and committed nothing (``stuck``)
        return {"ok": ok, "native_reason": info.get("rejected") or ("" if ok else "stuck"), **info}

    def route_via(self, x: float, y: float, mode: int | None = None) -> dict:
        """make_via: segment to (x, y) + via there, all-or-nothing; the head continues on the
        other layer of the via, the session ends."""
        self.log.tick("route_via")
        e = self.e
        if not e.is_routing():
            return {"ok": False, "reason": "no active routing session"}
        ok, info = _act.make_via(e, float(x), float(y), self._mode(mode))
        if not ok and e.is_routing():
            e.cancel_route()
        self._mutated()
        reason = info.get("rejected") or ("" if ok else
                                         ("via_not_placed" if info.get("routed") and not info.get("via_placed") else "stuck"))
        out = {"ok": ok, "native_reason": reason, **info}
        if ok:
            # PNS commits an arrival within the via radius: report where the via really landed
            mine = [v for v in e.get_vias() if int(v.net_code) == (self.current_net or int(v.net_code))]
            if mine:
                v = min(mine, key=lambda v: (v.x_mm - float(x)) ** 2 + (v.y_mm - float(y)) ** 2)
                out["via_xy"] = [float(v.x_mm), float(v.y_mm)]
        return out

    def finish(self, mode: int | None = None, max_attempts: int = 5) -> dict:
        self.log.tick("finish")
        e = self.e
        if not e.is_routing():
            return {"ok": False, "reason": "no active routing session"}
        ok, info = _act.finish(e, self._mode(mode), max_attempts)
        if e.is_routing():
            e.cancel_route()
        self._mutated()
        return {"ok": ok, **info}

    def cancel(self) -> dict:
        self.log.tick("cancel")
        if self.e.is_routing():
            self.e.cancel_route()
        return {"ok": True}

    def undo_last(self) -> dict:
        self.log.tick("undo_last")
        ok = bool(self.e.undo_last_segment())
        if ok:
            self._mutated()
        return {"ok": ok}

    def flip_posture(self) -> dict:
        self.log.tick("flip_posture")
        self.e.flip_posture()
        return {"ok": True}

    def end_net(self) -> dict:
        self.log.tick("end_net")
        if self.current_net is None:
            return {"ok": False, "reason": "no active net"}
        ok, info = _act.net_end(self.e, self.current_net)
        self.current_net = None
        return {"ok": ok, **info}

    def _mode(self, mode: int | None) -> int:
        if mode is not None:
            self.routing_mode = int(mode)
        return self.routing_mode

    # -- configure -----------------------------------------------------------
    def apply_profile(self, profile: dict, allowed: dict[str, dict] | None = None) -> dict:
        """Set track width / via dims from a *task-allowed* profile. ``allowed`` is the set
        the task exposes (netclass name -> dims); a profile outside it is refused so a policy
        cannot shrink widths to make DRC disappear."""
        self.log.tick("apply_profile")
        name = profile.get("name", "")
        if allowed is not None and name not in allowed:
            return {"ok": False, "reason": f"profile {name!r} not allowed", "allowed": sorted(allowed)}
        dims = allowed[name] if allowed is not None else profile
        e = self.e
        e.set_track_width(float(dims["track_width_mm"]))
        e.set_via_diameter(float(dims["via_diameter_mm"]))
        e.set_via_drill(float(dims["via_drill_mm"]))
        return {"ok": True, "applied": {"name": name, **{k: dims[k] for k in ("track_width_mm", "via_diameter_mm", "via_drill_mm")}}}

    def set_via_dims(self, diameter_mm: float, drill_mm: float, allowed: dict[str, dict] | None = None) -> dict:
        """Explicit via dims, guarded like ``apply_profile``: with ``allowed`` profiles given the via
        may be larger than an allowed profile's via but never smaller in diameter, drill or annular
        ring, so a policy cannot shrink vias below the task's rule set to make DRC disappear."""
        self.log.tick("set_via_dims")
        d, h = float(diameter_mm), float(drill_mm)
        if not (0 < h < d):
            return {"ok": False, "reason": "need 0 < drill_mm < diameter_mm"}
        if allowed is not None:
            floors = [(float(pr.get("via_diameter_mm", 0)), float(pr.get("via_drill_mm", 0))) for pr in allowed.values()]
            if not any(d >= pd and h >= ph and (d - h) >= (pd - ph) for pd, ph in floors):
                return {"ok": False, "reason": "via dims below every allowed profile",
                        "allowed": {k: {"via_diameter_mm": pr.get("via_diameter_mm"), "via_drill_mm": pr.get("via_drill_mm")}
                                    for k, pr in allowed.items()}}
        self.e.set_via_diameter(d)
        self.e.set_via_drill(h)
        return {"ok": True, "applied": {"via_diameter_mm": d, "via_drill_mm": h}}

    def set_corner_mode(self, mode: int) -> dict:
        """PNS corner style: 0 MITERED_45 (KiCad default), 1 ROUNDED_45, 2 MITERED_90, 3 ROUNDED_90."""
        self.log.tick("set_corner_mode")
        m = int(mode)
        if m not in (0, 1, 2, 3):
            return {"ok": False, "reason": "corner mode must be 0..3"}
        self.e.set_corner_mode(m)
        return {"ok": True, "applied": {"corner_mode": m}}

    # -- edit / repair -------------------------------------------------------
    def delete_net_routing(self, net_codes: list[int], frozen: list[int] | None = None) -> dict:
        self.log.tick("delete_net_routing")
        bad = sorted(set(net_codes) & set(frozen or []))
        if bad:
            return {"ok": False, "reason": "frozen scope", "nets": bad}
        n = self.e.delete_routing_of_nets([int(c) for c in net_codes])
        self.e.build_connectivity()
        self._mutated()
        return {"ok": True, "removed": int(n)}

    def lock_net(self, net_code: int, locked: bool = True) -> dict:
        self.log.tick("lock_net")
        n = self.e.lock_net(int(net_code), bool(locked))
        return {"ok": True, "changed": int(n)}

    def cleanup(self, net_codes: list[int] | None = None, dry_run: bool = True, finalize: bool = False) -> dict:
        """KiCad track cleaner. Topology-preserving merge by default; ``finalize`` adds the
        dangling passes and is only legal when no session is open (adapter checks)."""
        self.log.tick("cleanup")
        e = self.e
        if e.is_routing():
            return {"ok": False, "reason": "session open"}
        r = e.cleanup_tracks(dry_run=dry_run, merge_segments=True, clean_vias=finalize, tracks_in_pads=finalize,
                             dangling_tracks=finalize, dangling_vias=finalize, net_codes=net_codes)
        if not dry_run and getattr(r, "changed", False):
            self._mutated()
        return {"ok": bool(r.ran), "reject_reason": r.reject_reason,
                "removed": len(r.removed), "modified": len(r.modified)}

    # -- transaction ---------------------------------------------------------
    def checkpoint(self) -> dict:
        """IDLE-only: a pending route / drag is not a complete restorable snapshot."""
        self.log.tick("checkpoint")
        e = self.e
        if e.is_routing() or e.is_dragging():
            return {"ok": False, "reason": "session open; cancel or finish first"}
        h = int(e.checkpoint())
        self.state.checkpoint = h
        return {"ok": True, "handle": h, "revision": self.state.revision}

    def rollback(self, handle: int) -> dict:
        """Restore board + wrapper state (net selection, session) together."""
        self.log.tick("rollback")
        e = self.e
        if e.is_routing():
            e.cancel_route()
        ok = bool(e.restore(int(handle)))
        if ok:
            e.build_connectivity()
            self.current_net = None
            self._mutated()
        return {"ok": ok, "handle": int(handle), "revision": self.state.revision}

    def release_checkpoint(self, handle: int) -> None:
        self.e.release_checkpoint(int(handle))

    def run_drc(self, target_net_names: list[str] | None = None) -> dict:
        """Scoped DRC: only violations touching a target net are returned (the C++ cache
        stays complete). Error-severity counts are what ``clean_pass`` reads.

        The engine's incremental DRC returns what a full run would: it rechecks clearance only for
        the tracks and vias changed since the last DRC, and that baseline travels with every
        checkpoint, so a rollback is diffed correctly. A full run is the whole board's clearance
        pass, ~20 s on a 245-track board and ~90 s on a 163-net board, and the joint repair asks for a DRC
        twice per strategy tried."""
        self.log.tick("run_drc")
        e = self.e
        helper = getattr(e, "drc_helper", None)
        if helper is not None:
            helper.set_target_net_names(target_net_names)
        viol = e.run_drc_incremental()
        if helper is not None:
            viol = helper.get_violations()
        recs = []
        for v in viol:
            recs.append({"error_code": int(v.error_code), "error_type": v.error_type,
                         "message": v.message, "x_mm": v.x_mm, "y_mm": v.y_mm,
                         "layer": self.layers.name_from_board(int(v.layer)) if int(v.layer) >= 0 else None,
                         "net_names": list(v.net_names), "severity": int(v.severity)})
        # KiCad RPT_SEVERITY_ERROR = 0x20 (warnings 0x10). "Missing connection between items"
        # (DRCE_UNCONNECTED_ITEMS = 1) is the ratsnest, not a rule violation: reported apart.
        unconnected = [r for r in recs if _is_unconnected(r)]
        errors = [r for r in recs if r["severity"] == DRC_SEVERITY_ERROR and not _is_unconnected(r)]
        return {"ok": True, "count": len(recs), "errors": len(errors), "unconnected": len(unconnected), "violations": recs}

    def save(self, output_path: str) -> dict:
        self.log.tick("save")
        if self.e.is_routing():
            return {"ok": False, "reason": "session open"}
        self.e.save(str(output_path))
        pro = os.path.splitext(str(output_path))[0] + ".kicad_pro"
        return {"ok": True, "board": str(output_path), "project": pro if os.path.exists(pro) else ""}


def _is_unconnected(v) -> bool:
    """DRCE_UNCONNECTED_ITEMS (KiCad code 1) or its message / PCBWorld 'unconnected' keyword."""
    code = int(v["error_code"] if isinstance(v, dict) else v.error_code)
    et = str(v["error_type"] if isinstance(v, dict) else v.error_type).lower()
    return code == 1 or "unconnected" in et or "missing connection" in et


def _nc_dims(nc) -> dict:
    def pos(v, default):
        v = float(v) if v is not None else -1.0
        return v if v > 0 else default
    return {"track_width_mm": pos(nc.track_width_mm, 0.2), "clearance_mm": pos(nc.clearance_mm, 0.2),
            "via_diameter_mm": pos(nc.via_diameter_mm, 0.6), "via_drill_mm": pos(nc.via_drill_mm, 0.4)}
