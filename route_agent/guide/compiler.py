"""Compile a validated RouteGuidePlan into DetailedSubtasks.

The compiler fixes *addresses* (which clusters, which regions, which transitions) and
*portals* (centres of consecutive-stage overlaps as entry/exit hints). It never picks the
final copper polyline — that is the Detailed role's freedom inside the soft regions.
"""
from __future__ import annotations

from route_agent.guide.geometry import center, intersect, normalize
from route_agent.schema.guide import RouteGuidePlan
from route_agent.schema.snapshot import Snapshot
from route_agent.schema.tasks import DetailedSubtask, StageSpec, TransitionSpec


def compile_plan(plan: RouteGuidePlan, snap: Snapshot, *, capabilities: list[str] | None = None,
                 edges_only: list[str] | None = None) -> list[DetailedSubtask]:
    caps = set(capabilities if capabilities is not None else snap.execution.capabilities)
    order = {n: i for i, n in enumerate(plan.net_order)}
    tasks: list[DetailedSubtask] = []
    for tree, e in plan.edges():
        if edges_only is not None and e.id not in edges_only:
            continue
        net = snap.net(tree.net_code)
        src, dst = snap.cluster(e.source_cluster), snap.cluster(e.target_cluster)
        if src is None or dst is None:
            missing = e.source_cluster if src is None else e.target_cluster
            raise ValueError(f"edge {e.id}: cluster {missing!r} is not in snapshot {snap.board.snapshot_id}; validate the plan first")
        corr = [plan.corridor(s) for s in e.stages]
        if any(c is None for c in corr):
            raise ValueError(f"edge {e.id}: unknown corridor in stages {e.stages}; validate the plan first")
        stages = [StageSpec(c.id, c.layer, normalize(c.bbox), c.soft) for c in corr]
        transitions: list[TransitionSpec] = []
        vi = 0
        for i, (a, b) in enumerate(zip(corr, corr[1:])):
            common = intersect(a.bbox, b.bbox)
            if common is None:
                raise ValueError(f"edge {e.id}: stages {a.id} and {b.id} do not overlap; validate the plan first")
            if a.layer == b.layer:
                portal = center(common)
                stages[i].exit, stages[i + 1].entry = portal, portal
            else:
                vr = plan.via_region(e.via_regions[vi]) if vi < len(e.via_regions) else None
                if vr is None:
                    raise ValueError(f"edge {e.id}: layer change {a.id}->{b.id} has no via region; validate the plan first")
                vi += 1
                tb = normalize(intersect(common, vr.bbox) or common)
                transitions.append(TransitionSpec(a.id, b.id, vr.id, a.layer, b.layer, tb))
                portal = center(tb)
                stages[i].exit, stages[i + 1].entry = portal, portal
        if e.entry_portal:
            stages[0].entry = list(e.entry_portal)
        if e.exit_portal:
            stages[-1].exit = list(e.exit_portal)
        rule = snap.rule_for(tree.net_code)
        prof = dict(snap.rules.profiles.get(e.profile_ref, {}))
        prof["name"] = e.profile_ref
        if rule is not None and e.profile_ref == rule.class_name:
            prof.setdefault("source", "netclass")
        supported, reason = True, ""
        if transitions and "route_via" not in caps:
            supported, reason = False, "layer transition requested but adapter lacks route_via"
        if any(c.polygon for c in corr):
            supported, reason = False, "polygon corridor unsupported in v0"
        regions = [c.id for c in corr] + [t.via_region_id for t in transitions]
        tasks.append(DetailedSubtask(
            task_id=e.id, net_code=tree.net_code, net_name=net.net_name if net else tree.net_name,
            source_cluster=src.id, target_cluster=dst.id,
            source_anchors=[{"pad_id": a.pad_id, "x_mm": a.x_mm, "y_mm": a.y_mm, "layers": a.layers} for a in src.anchors],
            target_anchors=[{"pad_id": a.pad_id, "x_mm": a.x_mm, "y_mm": a.y_mm, "layers": a.layers} for a in dst.anchors],
            stages=stages, transitions=transitions, profile=prof,
            dependencies={"net": tree.net_code, "regions": regions, "rules_hash": snap.board.rules_hash,
                          "snapshot_id": snap.board.snapshot_id, "plan_revision": plan.revision},
            budget={"executor_calls": plan.budget.executor_calls, "local_retry_cap": plan.budget.local_retry_cap,
                    "stall_threshold": plan.budget.stall_threshold},
            order=order.get(tree.net_code, len(order)),
            execution={"status": "pending", "supported": supported, "reason": reason},
        ))
    tasks.sort(key=lambda t: (t.order, t.task_id))
    return tasks


def compile_summary(tasks: list[DetailedSubtask]) -> dict:
    regions = {s.corridor_id for t in tasks for s in t.stages}
    vias = {tr.via_region_id for t in tasks for tr in t.transitions}
    return {"subtasks": len(tasks), "edges": len(tasks), "corridor_regions": len(regions),
            "via_regions": len(vias), "unsupported": [t.task_id for t in tasks if not t.execution["supported"]]}
