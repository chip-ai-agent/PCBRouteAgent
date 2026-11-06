"""RouteGuidePlan: the Global role's output.

Spatial guides (corridors / via regions) and connection topology (connection trees) are
separate fields on purpose: a corridor says *where* copper may go on a layer, an edge says
*which existing clusters* to join through *which ordered regions*. Nothing in the plan is
copper; nothing in it is a hard constraint — hard rules come from the task and the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Corridor:
    id: str
    layer: str
    bbox: list[float]                      # [x1,y1,x2,y2] mm
    soft: bool = True
    polygon: list[list[float]] | None = None   # reserved: v0 compiler supports bbox only
    note: str = ""


@dataclass
class ViaRegion:
    id: str
    layers: list[str]                      # the two layers the transition joins
    bbox: list[float]
    note: str = ""


@dataclass
class ConnectionEdge:
    id: str
    source_cluster: str
    target_cluster: str
    stages: list[str]                      # ordered corridor ids
    via_regions: list[str] = field(default_factory=list)   # one per layer change, in order
    profile_ref: str = "Default"
    entry_portal: list[float] | None = None     # optional [x,y] preference inside stages[0]
    exit_portal: list[float] | None = None
    status: str = "pending_native_execution"
    note: str = ""


@dataclass
class ConnectionTree:
    net_code: int
    net_name: str = ""
    edges: list[ConnectionEdge] = field(default_factory=list)


@dataclass
class Reservation:
    region: str
    nets: list[int]
    priority: int = 0
    note: str = ""


@dataclass
class BasedOn:
    snapshot_id: str
    rules_hash: str = ""
    revision: int = 0


@dataclass
class Dependencies:
    nets: list[int] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    rules_hash: str = ""


@dataclass
class Budget:
    executor_calls: int = 400
    replan_cap: int = 3
    local_retry_cap: int = 3
    stall_threshold: int = 2       # same-corridor failures with no progress before replan


@dataclass
class RouteGuidePlan:
    plan_id: str
    based_on: BasedOn
    corridors: list[Corridor]
    connection_trees: list[ConnectionTree]
    via_regions: list[ViaRegion] = field(default_factory=list)
    net_order: list[int] = field(default_factory=list)
    reservations: list[Reservation] = field(default_factory=list)
    dependencies: Dependencies = field(default_factory=Dependencies)
    budget: Budget = field(default_factory=Budget)
    revision: int = 1
    layer_names: list[str] = field(default_factory=list)
    author: str = ""
    note: str = ""

    def corridor(self, cid: str) -> Corridor | None:
        for c in self.corridors:
            if c.id == cid:
                return c
        return None

    def via_region(self, vid: str) -> ViaRegion | None:
        for v in self.via_regions:
            if v.id == vid:
                return v
        return None

    def edges(self):
        for t in self.connection_trees:
            for e in t.edges:
                yield t, e


_BBOX = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
_POINT = {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 2, "maxItems": 2}

GUIDE_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RouteGuidePlan",
    "type": "object",
    "required": ["plan_id", "based_on", "corridors", "connection_trees"],
    "properties": {
        "plan_id": {"type": "string", "minLength": 1},
        "based_on": {
            "type": "object", "required": ["snapshot_id"],
            "properties": {"snapshot_id": {"type": "string"}, "rules_hash": {"type": "string"},
                           "revision": {"type": "integer"}},
        },
        "revision": {"type": "integer", "minimum": 1},
        "layer_names": {"type": "array", "items": {"type": "string"}},
        "corridors": {
            "type": "array",
            "items": {
                "type": "object", "required": ["id", "layer", "bbox"],
                "properties": {"id": {"type": "string", "minLength": 1}, "layer": {"type": "string"},
                               "bbox": _BBOX, "soft": {"type": "boolean"}, "note": {"type": "string"},
                               "polygon": {"type": ["array", "null"]}},
            },
        },
        "via_regions": {
            "type": "array",
            "items": {
                "type": "object", "required": ["id", "layers", "bbox"],
                "properties": {"id": {"type": "string"}, "bbox": _BBOX,
                               "layers": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                               "note": {"type": "string"}},
            },
        },
        "connection_trees": {
            "type": "array",
            "items": {
                "type": "object", "required": ["net_code", "edges"],
                "properties": {
                    "net_code": {"type": "integer"}, "net_name": {"type": "string"},
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "source_cluster", "target_cluster", "stages"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "source_cluster": {"type": "string", "pattern": "^cluster:\\d+:\\d+$"},
                                "target_cluster": {"type": "string", "pattern": "^cluster:\\d+:\\d+$"},
                                "stages": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "via_regions": {"type": "array", "items": {"type": "string"}},
                                "profile_ref": {"type": "string"},
                                "entry_portal": _POINT, "exit_portal": _POINT,
                                "status": {"type": "string"}, "note": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "net_order": {"type": "array", "items": {"type": "integer"}},
        "reservations": {"type": "array", "items": {
            "type": "object", "required": ["region", "nets"],
            "properties": {"region": {"type": "string"}, "nets": {"type": "array", "items": {"type": "integer"}},
                           "priority": {"type": "integer"}, "note": {"type": "string"}}}},
        "dependencies": {"type": "object"},
        "budget": {"type": "object", "properties": {
            "executor_calls": {"type": "integer", "minimum": 1}, "replan_cap": {"type": "integer", "minimum": 0},
            "local_retry_cap": {"type": "integer", "minimum": 0}, "stall_threshold": {"type": "integer", "minimum": 1}}},
        "author": {"type": "string"}, "note": {"type": "string"},
    },
}
