"""Schema + semantic validation of a RouteGuidePlan against a Snapshot.

Semantic rules (v0, bbox corridors only):
  V1  based_on.snapshot_id matches the snapshot (version guard; rules_hash too when given)
  V2  corridor / via-region ids unique; layers exist; bbox well-formed; inside board bbox (warn)
  V3  every edge: clusters exist and belong to the edge's net, source != target
  V4  stages reference corridors; consecutive same-layer stages intersect
  V5  consecutive different-layer stages intersect AND a via region (in order) covers both
      layers and intersects that common area; via_regions count == layer changes
  V6  first stage covers an anchor of the source cluster on that layer; last stage covers an
      anchor of the target cluster
  V7  profile_ref is a task-allowed profile (netclass name in rules.profiles)
  V8  net_order (if given) only names nets with trees; each tree net is in the snapshot
  V9  portals (if given) lie inside their stage
These guarantee the guide is self-consistent and addressable. Native legality (can PNS
actually route the corridor) is NOT checked here — that is the executor's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jsonschema

from route_agent.guide.geometry import contains, intersect, normalize
from route_agent.schema.common import from_dict
from route_agent.schema.guide import GUIDE_JSON_SCHEMA, RouteGuidePlan
from route_agent.schema.snapshot import Snapshot


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_edges: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "checked_edges": self.checked_edges}


def schema_errors(guide_dict: dict) -> list[str]:
    v = jsonschema.Draft202012Validator(GUIDE_JSON_SCHEMA)
    return [f"schema: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in sorted(v.iter_errors(guide_dict), key=lambda e: list(e.absolute_path))]


def parse_guide(guide_dict: dict) -> tuple[RouteGuidePlan | None, list[str]]:
    errs = schema_errors(guide_dict)
    if errs:
        return None, errs
    return from_dict(RouteGuidePlan, guide_dict), []


def _anchor_in(stage_bbox, layer, cluster) -> bool:
    return any(contains(stage_bbox, a.x_mm, a.y_mm) and layer in a.layers for a in cluster.anchors)


def validate_guide(plan: RouteGuidePlan, snap: Snapshot, *, edges_only: list[str] | None = None,
                   check_version: bool = True) -> ValidationReport:
    rep = ValidationReport()
    E, W = rep.errors.append, rep.warnings.append
    layers = set(snap.geometry.layer_names)
    board_bb = snap.geometry.outline_bbox

    # V1
    if check_version:
        if plan.based_on.snapshot_id != snap.board.snapshot_id:
            E(f"V1 based_on.snapshot_id {plan.based_on.snapshot_id!r} != snapshot {snap.board.snapshot_id!r}")
        if plan.based_on.rules_hash and plan.based_on.rules_hash != snap.board.rules_hash:
            E(f"V1 based_on.rules_hash {plan.based_on.rules_hash!r} != snapshot {snap.board.rules_hash!r}")
    if plan.layer_names and plan.layer_names != snap.geometry.layer_names:
        E(f"V1 layer_names {plan.layer_names} != board {snap.geometry.layer_names}")

    # V2
    seen: set[str] = set()
    for c in plan.corridors:
        if c.id in seen:
            E(f"V2 duplicate corridor id {c.id}")
        seen.add(c.id)
        if c.layer not in layers:
            E(f"V2 corridor {c.id}: layer {c.layer!r} not in {sorted(layers)}")
        if c.polygon:
            E(f"V2 corridor {c.id}: polygon corridors are not supported by the v0 compiler")
        b = normalize(c.bbox)
        if b[2] - b[0] <= 0 or b[3] - b[1] <= 0:
            E(f"V2 corridor {c.id}: degenerate bbox {c.bbox}")
        elif intersect(b, board_bb) is None:
            E(f"V2 corridor {c.id}: bbox {c.bbox} is outside the board bbox {board_bb}")
        elif not (board_bb[0] <= b[0] and b[2] <= board_bb[2] and board_bb[1] <= b[1] and b[3] <= board_bb[3]):
            W(f"V2 corridor {c.id}: bbox extends beyond the board bbox")
    for v in plan.via_regions:
        if v.id in seen:
            E(f"V2 duplicate region id {v.id}")
        seen.add(v.id)
        for l in v.layers:
            if l not in layers:
                E(f"V2 via region {v.id}: layer {l!r} unknown")
        b = normalize(v.bbox)
        if b[2] - b[0] <= 0 or b[3] - b[1] <= 0:
            E(f"V2 via region {v.id}: degenerate bbox")

    # V8
    tree_nets = {t.net_code for t in plan.connection_trees}
    for n in plan.net_order:
        if n not in tree_nets:
            E(f"V8 net_order names net {n} without a connection tree")
    for t in plan.connection_trees:
        if snap.net(t.net_code) is None:
            E(f"V8 tree net {t.net_code} is not on the snapshot")

    # per edge
    edge_ids: set[str] = set()
    for tree, e in plan.edges():
        if edges_only is not None and e.id not in edges_only:
            continue
        rep.checked_edges.append(e.id)
        if e.id in edge_ids:
            E(f"V3 duplicate edge id {e.id}")
        edge_ids.add(e.id)
        src, dst = snap.cluster(e.source_cluster), snap.cluster(e.target_cluster)
        if src is None:
            E(f"V3 {e.id}: source cluster {e.source_cluster} not in snapshot")
        if dst is None:
            E(f"V3 {e.id}: target cluster {e.target_cluster} not in snapshot")
        if src and src.net_code != tree.net_code:
            E(f"V3 {e.id}: source cluster belongs to net {src.net_code}, tree is net {tree.net_code}")
        if dst and dst.net_code != tree.net_code:
            E(f"V3 {e.id}: target cluster belongs to net {dst.net_code}, tree is net {tree.net_code}")
        if e.source_cluster == e.target_cluster:
            E(f"V3 {e.id}: source == target cluster")
        # V7
        if e.profile_ref not in snap.rules.profiles:
            E(f"V7 {e.id}: profile_ref {e.profile_ref!r} not an allowed profile {sorted(snap.rules.profiles)}")
        # V4/V5
        if not e.stages:
            E(f"V4 {e.id}: edge has no stages")
            continue
        stages = [plan.corridor(s) for s in e.stages]
        missing = [s for s, c in zip(e.stages, stages) if c is None]
        if missing:
            E(f"V4 {e.id}: unknown stage corridor(s) {missing}")
            continue
        changes = sum(1 for a, b in zip(stages, stages[1:]) if a.layer != b.layer)
        if changes != len(e.via_regions):
            E(f"V5 {e.id}: {changes} layer change(s) but {len(e.via_regions)} via region(s)")
        vi = 0
        for a, b in zip(stages, stages[1:]):
            common = intersect(a.bbox, b.bbox)
            if common is None:
                E(f"V{'4' if a.layer == b.layer else '5'} {e.id}: stages {a.id}->{b.id} do not intersect")
                if a.layer != b.layer:
                    vi += 1
                continue
            if a.layer != b.layer:
                if vi < len(e.via_regions):
                    vr = plan.via_region(e.via_regions[vi])
                    if vr is None:
                        E(f"V5 {e.id}: unknown via region {e.via_regions[vi]}")
                    else:
                        if set(vr.layers) != {a.layer, b.layer}:
                            E(f"V5 {e.id}: via region {vr.id} layers {vr.layers} != transition {a.layer}->{b.layer}")
                        if intersect(common, vr.bbox) is None:
                            E(f"V5 {e.id}: via region {vr.id} misses the {a.id}∩{b.id} overlap")
                vi += 1
        # V6
        if src and not _anchor_in(stages[0].bbox, stages[0].layer, src):
            E(f"V6 {e.id}: first stage {stages[0].id} ({stages[0].layer}) covers no reachable anchor of {src.id}")
        if dst and not _anchor_in(stages[-1].bbox, stages[-1].layer, dst):
            E(f"V6 {e.id}: last stage {stages[-1].id} ({stages[-1].layer}) covers no reachable anchor of {dst.id}")
        # V9
        if e.entry_portal and not contains(stages[0].bbox, *e.entry_portal):
            E(f"V9 {e.id}: entry_portal outside {stages[0].id}")
        if e.exit_portal and not contains(stages[-1].bbox, *e.exit_portal):
            E(f"V9 {e.id}: exit_portal outside {stages[-1].id}")
    return rep


def validate_guide_dict(guide_dict: dict, snap: Snapshot, **kw) -> tuple[RouteGuidePlan | None, ValidationReport]:
    plan, errs = parse_guide(guide_dict)
    if plan is None:
        return None, ValidationReport(errors=errs)
    return plan, validate_guide(plan, snap, **kw)
