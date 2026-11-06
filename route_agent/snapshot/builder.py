"""Build the structured ``Snapshot`` from a live engine (through ``NativeAdapter``).

Cluster ids are deterministic: within a net, clusters are ordered by their lexicographically
smallest pad id (``RF1:6`` < ``U1:20``), so ``cluster:24:0`` is stable across rebuilds of the
same board state and both roles can refer to it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from route_agent.exec.adapter import LazyEngine, NativeAdapter, file_sha256
from route_agent.paths import pcbworld_root
from route_agent.schema.snapshot import (
    Anchor, BoardRef, Cluster, ExecutionContext, FootprintRec, Geometry, NetConnectivity,
    PadRec, RuleContext, Snapshot, TrackRec, ViaRec, ZoneRec,
)


def _engine_version(repo_root: Path) -> str:
    for cand in (repo_root / "build_rl" / "pcbnew" / "python" / "rl" / "ENGINE_VERSION",):
        if cand.exists():
            return cand.read_text().strip()
    try:
        return subprocess.run(["git", "-C", str(repo_root / "engine"), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _rules_hash(rules: dict) -> str:
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:16]


def _cluster_pads(adapter: NativeAdapter, pads: list[PadRec]) -> list[list[PadRec]]:
    """Union pads of one net by the engine's committed connectivity (``get_connected_points``)."""
    parent = list(range(len(pads)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    key_to_pad: dict[tuple[float, float], int] = {}
    for i, p in enumerate(pads):
        key_to_pad[(round(p.x_mm, 4), round(p.y_mm, 4))] = i
    for i, p in enumerate(pads):
        layer = p.layers[0] if p.layers else adapter.layers.names[0]
        for x, y, _l in adapter.connected_points(p.x_mm, p.y_mm, layer):
            j = key_to_pad.get((round(x, 4), round(y, 4)))
            if j is not None:
                union(i, j)
    groups: dict[int, list[PadRec]] = {}
    for i, p in enumerate(pads):
        groups.setdefault(find(i), []).append(p)
    clusters = [sorted(g, key=lambda p: p.id) for g in groups.values()]
    clusters.sort(key=lambda g: g[0].id)
    return clusters


def build_snapshot(adapter: NativeAdapter, *, board_id: str, board_path: str, project_path: str,
                   work_set_names: list[str] | None = None, work_set_codes: list[int] | None = None,
                   repo_root: str | Path | None = None, preparation: str = "",
                   frozen_scope: list[int] | None = None) -> Snapshot:
    e = adapter.e
    e.build_connectivity()
    lm = adapter.layers
    names = e.get_net_names()                       # net_code -> name

    pads: list[PadRec] = []
    for p in e.get_pads():
        pads.append(PadRec(id=f"{p.footprint_ref}:{p.pad_name}", footprint_ref=p.footprint_ref,
                           pad_name=p.pad_name, x_mm=p.x_mm, y_mm=p.y_mm, width_mm=p.width_mm,
                           height_mm=p.height_mm, layers=lm.pad_layers(p.layer), net_code=p.net_code,
                           net_name=p.net_name, pad_type=p.pad_type, shape=p.shape))
    tracks = [TrackRec(t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm, t.width_mm, lm.name_from_board(t.layer) or "?",
                       t.net_code, getattr(t, "uuid", "")) for t in e.get_tracks()]
    vias = [ViaRec(v.x_mm, v.y_mm, v.diameter_mm, v.drill_mm,
                   [lm.name_from_board(v.top_layer) or "F.Cu", lm.name_from_board(v.bottom_layer) or "B.Cu"],
                   v.net_code, getattr(v, "uuid", "")) for v in e.get_vias()]
    keepouts = [ZoneRec(z.name, lm.name_from_board(z.layer) or "?", [list(pt) for pt in z.pts],
                        z.keepout_tracks, z.keepout_vias) for z in e.get_keepouts()]
    fps = []
    for f in e.get_footprints():
        cb = None
        pts = [pt for contour in (f.courtyard or []) for pt in contour]
        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            cb = [min(xs), min(ys), max(xs), max(ys)]
        fps.append(FootprintRec(f.ref, f.value, f.x_mm, f.y_mm, f.orientation_deg,
                                lm.name_from_board(f.layer) or "F.Cu", cb))
    bb = e.get_board_bbox()
    outline = [[s.x1_mm, s.y1_mm, s.x2_mm, s.y2_mm] for s in e.get_board_outline()]
    geometry = Geometry(layer_names=list(lm.names),
                        outline_bbox=[bb.x_mm, bb.y_mm, bb.x_mm + bb.width_mm, bb.y_mm + bb.height_mm],
                        outline=outline, footprints=fps, pads=pads, tracks=tracks, vias=vias,
                        keepouts=keepouts, protected=[])

    # work set
    work: set[int] = set(work_set_codes or [])
    if work_set_names:
        inv = {v: k for k, v in names.items()}
        for n in work_set_names:
            if n not in inv:
                raise KeyError(f"net {n!r} not on board")
            work.add(inv[n])

    routable = set(e.get_routable_nets())
    ratsnest = e.get_ratsnest()
    groups = e.get_pad_groups()
    conn: list[NetConnectivity] = []
    for code in sorted(routable | work):
        npads = [p for p in pads if p.net_code == code and p.pad_type != "np_thru_hole"]
        clusters = []
        for k, grp in enumerate(_cluster_pads(adapter, npads)):
            clusters.append(Cluster(id=f"cluster:{code}:{k}", net_code=code,
                                    anchors=[Anchor(p.id, p.x_mm, p.y_mm, list(p.layers)) for p in grp]))
        edges = [[r.x1_mm, r.y1_mm, r.x2_mm, r.y2_mm] for r in ratsnest if r.net_code == code]
        conn.append(NetConnectivity(net_code=code, net_name=names.get(code, ""), clusters=clusters,
                                    remaining_edges=edges, pad_groups=int(groups.get(code, len(clusters))),
                                    in_work_set=code in work))

    rules_raw = adapter.get_rules()
    net_rules = [adapter.net_rule(c.net_code, c.net_name) for c in conn]
    rules = RuleContext(net_rules=net_rules, min_track_width_mm=rules_raw["min_track_width_mm"],
                        min_clearance_mm=rules_raw["min_clearance_mm"],
                        min_via_diameter_mm=rules_raw["min_via_diameter_mm"],
                        copper_edge_clearance_mm=rules_raw["copper_edge_clearance_mm"],
                        profiles=rules_raw["profiles"],
                        checker_coverage=["kicad_drc:clearance", "kicad_drc:track_width", "kicad_drc:via",
                                          "kicad_drc:unconnected", "connectivity:pad_groups"],
                        source="kicad_pro" if e.was_project_loaded_from_file() else "engine_default")
    rules_hash = _rules_hash({"profiles": rules.profiles, "min": [rules.min_track_width_mm, rules.min_clearance_mm]})
    input_hash = file_sha256(board_path) if Path(board_path).exists() else ""
    snap_id = hashlib.sha256(f"{input_hash}:{rules_hash}:{len(tracks)}:{len(vias)}:{adapter.state.revision}".encode()).hexdigest()[:12]
    root = Path(repo_root) if repo_root else pcbworld_root()
    board = BoardRef(board_id=board_id, board_path=str(board_path), project_path=str(project_path),
                     snapshot_id=snap_id, revision=adapter.state.revision, input_hash=input_hash,
                     rules_hash=rules_hash, engine_version=_engine_version(root), preparation=preparation,
                     extra={"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "copper_layers": len(lm.names)})
    execution = ExecutionContext(capabilities=adapter.capabilities(), frozen_scope=sorted(frozen_scope or []),
                                 budget_remaining={})
    return Snapshot(board=board, geometry=geometry, connectivity=conn, rules=rules, views=[],
                    execution=execution, work_set=sorted(work))


def open_board(board_path: str, project_path: str | None = None, **kw) -> tuple[LazyEngine, NativeAdapter]:
    lazy = LazyEngine(board_path, project_path, **kw)
    return lazy, NativeAdapter(lazy)
