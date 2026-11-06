"""Deterministic straight-corridor Global baseline (no model).

For every work-set net with two or more committed clusters it builds a minimum spanning tree
over the clusters (weight = distance of the nearest anchor pair, plus a penalty when the pair
shares no copper layer) and, per tree edge, one soft corridor around the straight segment
between the two nearest anchors on a shared layer — or two corridors plus a via region at the
segment midpoint when the anchors share no layer. Nets are ordered short-first.

This is the "scripted Detailed without a Global model" condition of the comparison and a
stress source for the executor. It knows nothing about obstacles: a corridor may cover
foreign pads, and the executor is expected to fail inside it and report that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from route_agent.schema.guide import (BasedOn, Budget, ConnectionEdge, ConnectionTree, Corridor,
                                      RouteGuidePlan, ViaRegion)
from route_agent.schema.snapshot import Anchor, Cluster, Snapshot

VIA_PENALTY_MM = 4.0          # a layer change costs about this much extra straight length


@dataclass
class _Pair:
    dist: float
    score: float
    a: Anchor
    b: Anchor
    common: list[str]          # copper layers both anchors occupy, in board order


def _nearest_pair(ca: Cluster, cb: Cluster, layer_names: list[str]) -> _Pair:
    best: _Pair | None = None
    for pa in ca.anchors:
        for pb in cb.anchors:
            d = math.hypot(pa.x_mm - pb.x_mm, pa.y_mm - pb.y_mm)
            common = [ln for ln in layer_names if ln in pa.layers and ln in pb.layers]
            score = d + (0.0 if common else VIA_PENALTY_MM)
            if best is None or score < best.score or (score == best.score and (pa.pad_id, pb.pad_id) < (best.a.pad_id, best.b.pad_id)):
                best = _Pair(d, score, pa, pb, common)
    assert best is not None, "clusters without anchors"
    return best


def _mst(clusters: list[Cluster], layer_names: list[str]) -> list[tuple[int, int, _Pair]]:
    """Prim's algorithm over clusters; deterministic (ties broken by index)."""
    n = len(clusters)
    pairs: dict[tuple[int, int], _Pair] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pairs[(i, j)] = _nearest_pair(clusters[i], clusters[j], layer_names)

    def pair(i, j):
        """The nearest pair oriented from cluster i (``a``) to cluster j (``b``)."""
        p = pairs[(i, j) if i < j else (j, i)]
        return p if i < j else _Pair(p.dist, p.score, p.b, p.a, p.common)

    in_tree = [0]
    out: list[tuple[int, int, _Pair]] = []
    while len(in_tree) < n:
        best = None
        for i in in_tree:
            for j in range(n):
                if j in in_tree:
                    continue
                p = pair(i, j)
                key = (p.score, i, j)
                if best is None or key < best[0]:
                    best = (key, i, j, p)
        _k, i, j, p = best
        in_tree.append(j)
        out.append((i, j, p))
    return out


def _seg_bbox(ax: float, ay: float, bx: float, by: float, margin: float, board: list[float] | None) -> list[float]:
    b = [min(ax, bx) - margin, min(ay, by) - margin, max(ax, bx) + margin, max(ay, by) + margin]
    if board:
        x1, y1, x2, y2 = board
        b = [max(b[0], x1), max(b[1], y1), min(b[2], x2), min(b[3], y2)]
        # Connector pads can overhang the Edge.Cuts bbox. Keep the actual
        # endpoints reachable after clipping; V2 permits this with a warning.
        b = [min(b[0], ax, bx), min(b[1], ay, by),
             max(b[2], ax, bx), max(b[3], ay, by)]
    return [math.floor(b[0] * 1000) / 1000, math.floor(b[1] * 1000) / 1000,
            math.ceil(b[2] * 1000) / 1000, math.ceil(b[3] * 1000) / 1000]


def _profile_for(snap: Snapshot, net_code: int) -> str:
    rule = snap.rule_for(net_code)
    profiles = snap.rules.profiles or {}
    if rule is not None and rule.class_name in profiles:
        return rule.class_name
    return next(iter(profiles)) if profiles else "Default"


def baseline_plan(snap: Snapshot, *, margin_mm: float = 1.5, plan_id: str = "baseline-straight",
                  calls_per_edge: int = 80, local_retry_cap: int = 2) -> RouteGuidePlan:
    """Build the straight-corridor plan for ``snap.work_set``. Always validates by construction
    (V1-V9): every stage contains its anchor on its layer, consecutive stages overlap around
    the midpoint, and each layer change carries a via region covering that overlap."""
    layer_names = list(snap.geometry.layer_names)
    board = snap.geometry.outline_bbox
    corridors: list[Corridor] = []
    via_regions: list[ViaRegion] = []
    trees: list[ConnectionTree] = []
    lengths: dict[int, float] = {}
    n_edges = 0
    for code in snap.work_set:
        net = snap.net(code)
        if net is None or len(net.clusters) < 2:
            continue
        prof = _profile_for(snap, code)
        edges: list[ConnectionEdge] = []
        total = 0.0
        for k, (i, j, p) in enumerate(_mst(net.clusters, layer_names)):
            eid = f"n{code}_e{k}"
            total += p.dist
            src, dst = net.clusters[i], net.clusters[j]
            if p.common:
                layer = p.common[0]
                cid = f"{eid}_c0"
                corridors.append(Corridor(id=cid, layer=layer, bbox=_seg_bbox(p.a.x_mm, p.a.y_mm, p.b.x_mm, p.b.y_mm, margin_mm, board),
                                          note=f"straight {p.a.pad_id}->{p.b.pad_id}"))
                edges.append(ConnectionEdge(id=eid, source_cluster=src.id, target_cluster=dst.id, stages=[cid],
                                            via_regions=[], profile_ref=prof))
            else:
                la = next(ln for ln in layer_names if ln in p.a.layers)
                lb = next(ln for ln in layer_names if ln in p.b.layers)
                mx, my = (p.a.x_mm + p.b.x_mm) / 2, (p.a.y_mm + p.b.y_mm) / 2
                c0, c1, vid = f"{eid}_c0", f"{eid}_c1", f"{eid}_v"
                corridors.append(Corridor(id=c0, layer=la, bbox=_seg_bbox(p.a.x_mm, p.a.y_mm, mx, my, margin_mm, board),
                                          note=f"straight {p.a.pad_id}->via"))
                corridors.append(Corridor(id=c1, layer=lb, bbox=_seg_bbox(mx, my, p.b.x_mm, p.b.y_mm, margin_mm, board),
                                          note=f"straight via->{p.b.pad_id}"))
                via_regions.append(ViaRegion(id=vid, layers=[la, lb], bbox=_seg_bbox(mx, my, mx, my, margin_mm, board)))
                edges.append(ConnectionEdge(id=eid, source_cluster=src.id, target_cluster=dst.id, stages=[c0, c1],
                                            via_regions=[vid], profile_ref=prof))
            n_edges += 1
        trees.append(ConnectionTree(net_code=code, net_name=net.net_name, edges=edges))
        lengths[code] = total
    net_order = sorted(lengths, key=lambda c: (lengths[c], c))
    return RouteGuidePlan(
        plan_id=plan_id,
        based_on=BasedOn(snapshot_id=snap.board.snapshot_id, rules_hash=snap.board.rules_hash, revision=0),
        corridors=corridors, connection_trees=trees, via_regions=via_regions, net_order=net_order,
        budget=Budget(executor_calls=max(400, calls_per_edge * n_edges), replan_cap=3,
                      local_retry_cap=local_retry_cap, stall_threshold=2),
        revision=1, layer_names=layer_names, author="baseline-straight-v0",
        note=f"MST over committed clusters, straight corridors, margin {margin_mm} mm, short nets first",
    )


def baseline_summary(plan: RouteGuidePlan) -> dict:
    edges = [e for _t, e in plan.edges()]
    return {"nets": len(plan.connection_trees), "edges": len(edges), "corridors": len(plan.corridors),
            "via_regions": len(plan.via_regions), "layer_changes": sum(len(e.via_regions) for e in edges),
            "budget": plan.budget.executor_calls}
