"""Provider-neutral prompt bundles for the Global and Detailed roles.

A bundle is ``{"system", "user", "images": [{"view_id", "path"}], "response_schema"}``; the
provider decides how to attach images (data URI, file upload) and how to request JSON.
The structured sidecar is trimmed to the work set so the text stays bounded; the full
snapshot path is referenced for tools that can read it.
"""
from __future__ import annotations

import json

from route_agent.schema.common import to_dict
from route_agent.schema.feedback import PATCH_JSON_SCHEMA

PATCH_EXAMPLE = {
    "patch_id": "p1", "based_on": {"snapshot_id": "<sidecar snapshot_id>", "rules_hash": "<sidecar rules_hash>", "plan_revision": 1},
    "replace_corridors": [{"id": "n7_e0_c0", "layer": "F.Cu", "bbox": [185.0, 90.0, 190.0, 96.0]},
                          {"id": "n7_e0_c1", "layer": "B.Cu", "bbox": [188.0, 94.0, 203.0, 112.0]},
                          {"id": "n7_e0_c2", "layer": "F.Cu", "bbox": [200.0, 110.0, 204.0, 114.0]}],
    "replace_via_regions": [{"id": "n7_e0_v1", "layers": ["F.Cu", "B.Cu"], "bbox": [188.0, 94.0, 190.0, 96.0]},
                            {"id": "n7_e0_v2", "layers": ["B.Cu", "F.Cu"], "bbox": [200.0, 110.0, 203.0, 112.0]}],
    "replace_edges": [{"id": "n7_e0", "source_cluster": "cluster:7:0", "target_cluster": "cluster:7:1",
                       "stages": ["n7_e0_c0", "n7_e0_c1", "n7_e0_c2"], "via_regions": ["n7_e0_v1", "n7_e0_v2"], "profile_ref": "Default"}],
    "preserve": ["n3_e0", "n5_e1"], "budget_revision": None, "note": "F.Cu exit, B.Cu detour around the pin field, F.Cu landing",
}
from route_agent.schema.guide import GUIDE_JSON_SCHEMA
from route_agent.schema.snapshot import Snapshot
from route_agent.schema.tasks import DetailedSubtask

GLOBAL_SYSTEM = """You are the Global routing planner for a printed circuit board.
You receive (1) board-context, per-layer copper-state and connectivity-demand images of the
current committed board in one physical mm frame and (2) a
structured sidecar with exact ids and rules. You output a RouteGuidePlan as JSON only.

What a plan is:
- connection_trees: per net, edges that join two EXISTING connectivity clusters
  (ids like cluster:24:0). Never invent a terminal; Steiner regions are corridors, not clusters.
- corridors: axis-aligned regions per layer (bbox [x1,y1,x2,y2] in mm). They are SOFT: the
  Detailed router works inside them when it can and reports deviations. A region may cover
  obstacles; the Detailed router finds a legal path inside it.
- An edge lists its ordered stages (corridor ids). Consecutive same-layer stages must
  overlap. A layer change needs the two stages to overlap AND a via_region covering that
  overlap, listed in via_regions in order.
- Layer changes are YOUR decision only: the Detailed router never places a via and never
  leaves the layer of the stage it is in. A pad fenced in on its own layer (a dense SMD row,
  a pad walled by neighbouring pads or earlier tracks) cannot be left on that layer: start
  the edge on the other layer when the pad is thru-hole, otherwise put a small via_region
  (about 1 mm) right next to the pad between a short first stage on the pad's layer and the
  next stage on the other layer. Without that the edge is certain to fail.
- The first stage must contain an anchor of the source cluster on that layer; the last
  stage must contain an anchor of the target cluster.
- profile_ref must be one of the allowed profiles.
- net_order is your routing order; reservations are soft preferences.
- based_on.snapshot_id must copy the sidecar's snapshot_id exactly.

Plan only the work-set nets. Prefer short, wide corridors; use layer changes where a pad or
a channel needs one and nowhere else; keep corridors inside the board bbox; use the images
to avoid dense pad fields and existing copper. Do not draw copper. Output JSON matching the
schema, nothing else."""

DETAILED_SYSTEM = """You are the Detailed router. You execute ONE DetailedSubtask through tools.
Contract: every route_to / route_via commits a segment and CLOSES the session, so each step is
start_route(head, layer) then route_to(x,y) or route_via(x,y). route_via switches you to the
other layer at (x,y). Call finish once the head is inside the last stage to let the engine
close the connection. Stay inside the soft corridors when possible; if a commit fails
twice in the same region, stop and report (do not wander). Never change the width or via
profile except through apply_profile with an allowed name. The edge is done when a source
anchor and a target anchor are joined (connected_points), not when the whole net has one pad
group. Reply with tool calls; when done or blocked, reply with a JSON
FeedbackPacket {outcome, last_head, native_reason, region_ref, request}."""


def _sidecar(snap: Snapshot, max_other_nets: int = 40) -> dict:
    g = snap.geometry
    work = set(snap.work_set)
    nets = []
    for n in snap.connectivity:
        if n.in_work_set:
            nets.append(to_dict(n))
    others = [{"net_code": n.net_code, "net_name": n.net_name, "pad_groups": n.pad_groups,
               "remaining_edges": len(n.remaining_edges)} for n in snap.connectivity if not n.in_work_set]
    pads_work = [to_dict(p) for p in g.pads if p.net_code in work]
    return {
        "board": to_dict(snap.board),
        "layer_names": g.layer_names,
        "outline_bbox": g.outline_bbox,
        "work_set": sorted(work),
        "work_set_nets": nets,
        "work_set_pads": pads_work,
        "other_nets_summary": others[:max_other_nets],
        "other_nets_count": len(others),
        "pad_count": len(g.pads), "track_count": len(g.tracks), "via_count": len(g.vias),
        "footprints": [{"ref": f.ref, "x_mm": f.x_mm, "y_mm": f.y_mm, "courtyard_bbox": f.courtyard_bbox} for f in g.footprints],
        "keepouts": [to_dict(z) for z in g.keepouts],
        "protected": list(g.protected),
        "obstacle_geometry": {"other_pads": [to_dict(p) for p in g.pads if p.net_code not in work],
                              "tracks": [to_dict(t) for t in g.tracks],
                              "vias": [to_dict(v) for v in g.vias]},
        "rules": {"profiles": snap.rules.profiles, "work_set_rules": [to_dict(r) for r in snap.rules.net_rules if r.net_code in work],
                  "min_track_width_mm": snap.rules.min_track_width_mm, "min_clearance_mm": snap.rules.min_clearance_mm,
                  "min_via_diameter_mm": snap.rules.min_via_diameter_mm,
                  "copper_edge_clearance_mm": snap.rules.copper_edge_clearance_mm,
                  "source": snap.rules.source,
                  "checker_coverage": snap.rules.checker_coverage},
        "views": [to_dict(v) | {"object_map": {}} for v in snap.views],
        "execution": to_dict(snap.execution),
    }


def build_global_bundle(snap: Snapshot, feedback: dict | None = None, previous_plan: dict | None = None) -> dict:
    side = _sidecar(snap)
    user = ["# Task", "Produce a RouteGuidePlan for the work-set nets of this board.",
            "", "# Images", "One frame, mm coordinates; pixel_to_mm affine per view is in the sidecar.views list."]
    for v in snap.views:
        user.append(f"- {v.view_id}: layers {v.layers}, bbox_mm {v.bbox_mm}, size {v.image_size}")
    user += ["", "# Structured sidecar (JSON)", json.dumps(side, ensure_ascii=False)]
    schema = GUIDE_JSON_SCHEMA
    if previous_plan is not None:
        user += ["", "# Previous plan (revise only the affected items; keep the rest)", json.dumps(previous_plan)]
    if feedback is not None:
        rev = (previous_plan or {}).get("revision", 0)
        user += ["", "# Detailed feedback (the failure you must fix)", json.dumps(feedback),
                 "", "Return a GlobalGuidePatch JSON object: {patch_id, based_on: {snapshot_id, rules_hash, "
                 f"plan_revision: {rev}}}, replace_corridors, replace_via_regions, replace_edges, preserve, "
                 "net_order?, budget_revision?}. Replace only what the feedback requires (a replaced edge or "
                 "corridor is given in full); list every unaffected edge id in preserve. The cluster ids and "
                 "snapshot_id stay those of the sidecar (the board was rolled back to that stable state plus "
                 "the edges already done, see feedback.live_state). A full RouteGuidePlan is accepted instead "
                 "of a patch but must then validate on its own.",
                 "", "# GlobalGuidePatch JSON schema", json.dumps(PATCH_JSON_SCHEMA),
                 "", "# Example patch (shape only; use the real ids, layers and mm coordinates of this board)",
                 json.dumps(PATCH_EXAMPLE),
                 "", "If the feedback carries `packets` (several edges, an end-of-pass batch), every edge in it failed in "
                 "this pass and the board now holds everything else that was routed: return ONE patch that fixes as "
                 "many of them as you can, those whose evidence makes the fix clear first (via_sites, head_is_thru, "
                 "alt_starts), and list every edge you leave unchanged in preserve.",
                 "", "The Detailed never places a via on its own: a layer change exists only where the plan has a via "
                 "region. When failure_evidence.via_sites is non-empty the head is fenced in on its layer; each entry is "
                 "a clearance-checked [x, y] in mm right next to the stuck head where a via fits on both layers. To route "
                 "the edge on the other layer, add a via region (a box about 1 mm wide around one of those points), a "
                 "short first-stage corridor on the head's layer that contains the head and overlaps that box, and the "
                 "next stage on the other layer overlapping it too. blocker_candidates lists the foreign pads / tracks "
                 "fencing the head in (with positions and layers); head_is_thru true means the pad itself spans both "
                 "layers, so a corridor starting on the other layer needs no via; alt_starts are other points of the "
                 "net's copper near the first stage the edge could start from instead.",
                 "", "When a packet carries `history`, the edge has been re-planned before: each entry is the shape the "
                 "Detailed executed in an earlier round (stages with layer and bbox, via regions) and where it died "
                 "(died_at, head, reason). Those shapes have already failed: do not return one of them or a near copy. "
                 "Change what failed: enter the target from its other side or on the other layer, move the via region "
                 "away from died_at, or start the edge from one of alt_starts.",
                 "", "Rules the validator enforces: every corridor has id, layer and bbox; every via region has id, "
                 "layers (the two layers it joins) and bbox; consecutive same-layer stages overlap; a layer change "
                 "needs a via region covering the overlap of the two stages; the first stage contains an anchor of "
                 "source_cluster on its layer and the last stage an anchor of target_cluster on its layer."]
        schema = PATCH_JSON_SCHEMA
    return {"role": "global", "system": GLOBAL_SYSTEM, "user": "\n".join(user),
            "images": [{"view_id": v.view_id, "path": v.path} for v in snap.views],
            "response_schema": schema}


def build_detailed_bundle(task: DetailedSubtask, snap: Snapshot, tools: list[dict], local_state: dict | None = None) -> dict:
    user = ["# DetailedSubtask", json.dumps(to_dict(task)),
            "", "# Local state", json.dumps(local_state or {}),
            "", "# Allowed profiles", json.dumps(snap.rules.profiles),
            "", "# Layer names", json.dumps(snap.geometry.layer_names)]
    return {"role": "detailed", "system": DETAILED_SYSTEM, "user": "\n".join(user), "images": [], "tools": tools}
