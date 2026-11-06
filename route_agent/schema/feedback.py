"""Detailed -> Global feedback and Global -> plan patch contracts."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Attempt:
    kind: str                 # route_to | route_via | finish | start_route
    target: list[float]       # [x,y] (+layer for start)
    mode: int
    ok: bool
    cost: int                 # executor facade calls consumed by this attempt
    native_reason: str = "unknown"
    region_ref: str = ""


@dataclass
class FailureEvidence:
    native_reason: str = "unknown"
    blocker_candidates: list[dict] = field(default_factory=list)   # [{id, kind, net_code, source, x_mm?, y_mm?, layer(s)?}]
    region_ref: str = ""
    # boxed-in head (the head could not leave its spot on its layer); all for the Global to decide
    # a layer change with - the Detailed never places a via of its own:
    via_sites: list[list[float]] = field(default_factory=list)     # [x, y] mm next to the head where a via fits on both layers
    head_is_thru: bool = False                                     # the head is a thru-hole pad: it bridges the layers itself
    alt_starts: list[list[float]] = field(default_factory=list)    # other copper of the source cluster on this layer near the first stage


@dataclass
class FeedbackPacket:
    snapshot_ref: str
    guide_id: str
    edge_id: str
    net_code: int
    outcome: str                                  # progress | blocked | partial | conflict | unsupported | done
    last_head: list[float]                        # [x, y, layer_index]
    attempts: list[Attempt]
    connectivity: dict                            # {clusters:[ids], pad_groups:int, remaining_edges:int}
    failure_evidence: FailureEvidence
    checks: dict = field(default_factory=dict)    # {fresh:bool, drc_scoped:int, scope_ok:bool}
    request: str = ""                             # widen | alternate_layer | reorder | repair | ""
    stage_ref: str = ""
    deviations: list[dict] = field(default_factory=list)
    cost: int = 0
    history: list[dict] = field(default_factory=list)   # this edge's earlier rounds: plan shape executed + where it died


@dataclass
class GlobalGuidePatch:
    patch_id: str
    based_on: dict                                # {snapshot_id, rules_hash, plan_revision}
    replace_edges: list[dict] = field(default_factory=list)       # full ConnectionEdge dicts
    replace_corridors: list[dict] = field(default_factory=list)   # full Corridor dicts
    replace_via_regions: list[dict] = field(default_factory=list)
    net_order: list[int] | None = None
    preserve: list[str] = field(default_factory=list)             # edge ids whose routes are accepted
    budget_revision: dict | None = None                           # explicit; never refunded by rollback
    note: str = ""


PATCH_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GlobalGuidePatch",
    "type": "object",
    "required": ["patch_id", "based_on"],
    "properties": {
        "patch_id": {"type": "string"},
        "based_on": {"type": "object", "required": ["snapshot_id", "plan_revision"],
                     "properties": {"snapshot_id": {"type": "string"}, "rules_hash": {"type": "string"},
                                    "plan_revision": {"type": "integer"}}},
        "replace_edges": {"type": "array", "description": "full ConnectionEdge dicts; a NEW edge must also carry net_code",
                          "items": {"type": "object", "required": ["id", "source_cluster", "target_cluster", "stages"],
                                    "properties": {"id": {"type": "string"}, "source_cluster": {"type": "string"},
                                                   "target_cluster": {"type": "string"},
                                                   "stages": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                                   "via_regions": {"type": "array", "items": {"type": "string"}},
                                                   "profile_ref": {"type": "string"}, "net_code": {"type": "integer"}}}},
        "replace_corridors": {"type": "array", "items": {"type": "object", "required": ["id", "layer", "bbox"],
                                                         "properties": {"id": {"type": "string"}, "layer": {"type": "string"},
                                                                        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}}},
        "replace_via_regions": {"type": "array", "items": {"type": "object", "required": ["id", "bbox"],
                                                           "properties": {"id": {"type": "string"},
                                                                          "layers": {"type": "array", "items": {"type": "string"}, "description": "the two copper layers the via joins, e.g. [\"F.Cu\", \"B.Cu\"]"},
                                                                          "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}}},
        "net_order": {"type": ["array", "null"], "items": {"type": "integer"}},
        "preserve": {"type": "array", "items": {"type": "string"}, "description": "edge ids whose routes are accepted"},
        "budget_revision": {"type": ["object", "null"]},
        "note": {"type": "string"},
    },
}
