"""Deterministic repair of a model-produced RouteGuidePlan dict before validation.

Only unambiguous slips are fixed, the kind a model makes when it writes a hundred corridors by
hand: corridor / via-region bboxes are normalized; a via region that does not cover the
overlap of the two stages it joins is moved onto that overlap (V5); an edge with a layer change
but no via region for it gets one centred on the overlap; a via region's ``layers`` are set to
the two stage layers. Everything else — unknown cluster ids, stages that do not overlap,
unknown profiles, missing anchors — is left for the validator, because fixing it would mean
inventing a plan. Returns ``(repaired_guide, notes)``; ``notes`` is empty when nothing changed.
"""
from __future__ import annotations

import copy

from route_agent.guide.geometry import intersect, normalize


def repair_guide(guide: dict, snap=None) -> tuple[dict, list[str]]:
    g = copy.deepcopy(guide)
    notes: list[str] = []
    corr = {c.get("id"): c for c in g.get("corridors", []) if isinstance(c, dict)}
    vias = {v.get("id"): v for v in g.get("via_regions", []) if isinstance(v, dict)}
    for c in corr.values():
        if isinstance(c.get("bbox"), list) and len(c["bbox"]) == 4:
            c["bbox"] = normalize(c["bbox"])
    for v in vias.values():
        if isinstance(v.get("bbox"), list) and len(v["bbox"]) == 4:
            v["bbox"] = normalize(v["bbox"])
    for tree in g.get("connection_trees", []):
        for e in tree.get("edges", []):
            stages = [corr.get(s) for s in e.get("stages", [])]
            if not stages or any(s is None or not isinstance(s.get("bbox"), list) for s in stages):
                continue
            old_ids = list(e.get("via_regions") or [])
            changes = [(a, b) for a, b in zip(stages, stages[1:]) if a.get("layer") != b.get("layer")]
            new_ids: list[str] = []
            for k, (a, b) in enumerate(changes):
                ov = intersect(a["bbox"], b["bbox"])
                if ov is None:
                    if k < len(old_ids):
                        new_ids.append(old_ids[k])
                    continue                                   # V5 overlap failure: the validator reports it
                vid = old_ids[k] if k < len(old_ids) else None
                vr = vias.get(vid) if vid else None
                if vr is None:
                    vid = f"{e.get('id')}_via{k + 1}_auto"
                    vr = {"id": vid, "layers": [a["layer"], b["layer"]], "bbox": ov}
                    vias[vid] = vr
                    g.setdefault("via_regions", []).append(vr)
                    notes.append(f"{e.get('id')}: added via region {vid} on the {a.get('id')}∩{b.get('id')} overlap")
                else:
                    if intersect(vr.get("bbox", ov), ov) is None:
                        vr["bbox"] = ov
                        notes.append(f"{e.get('id')}: via region {vid} moved onto the {a.get('id')}∩{b.get('id')} overlap")
                    if set(vr.get("layers") or []) != {a["layer"], b["layer"]}:
                        vr["layers"] = [a["layer"], b["layer"]]
                        notes.append(f"{e.get('id')}: via region {vid} layers set to {a['layer']}/{b['layer']}")
                new_ids.append(vid)
            if new_ids != old_ids[:len(new_ids)] or len(old_ids) != len(changes):
                e["via_regions"] = new_ids
                if len(old_ids) != len(changes):
                    notes.append(f"{e.get('id')}: via_regions list resized from {len(old_ids)} to {len(changes)}")
    return g, notes
