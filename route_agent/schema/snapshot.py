"""Structured snapshot: the sidecar the Global role reads next to the four images.

Coordinates are KiCad board mm (y down). Layers are KiCad names (``F.Cu``/``B.Cu``/``In1.Cu``);
the runtime LayerMap (``route_agent.exec.adapter.LayerMap``) converts them to the engine's
human 1..N ids — the raw PCB_LAYER_ID enum never appears in a contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BoardRef:
    board_id: str
    board_path: str
    project_path: str
    snapshot_id: str
    revision: int = 0
    input_hash: str = ""
    rules_hash: str = ""
    engine_version: str = ""
    preparation: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class PadRec:
    id: str                    # "REF:pad"
    footprint_ref: str
    pad_name: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    layers: list[str]          # copper layers the pad occupies
    net_code: int
    net_name: str
    pad_type: str              # smd | thru_hole | np_thru_hole
    shape: str


@dataclass
class TrackRec:
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    width_mm: float
    layer: str
    net_code: int
    uuid: str = ""


@dataclass
class ViaRec:
    x_mm: float
    y_mm: float
    diameter_mm: float
    drill_mm: float
    layers: list[str]
    net_code: int
    uuid: str = ""


@dataclass
class ZoneRec:
    name: str
    layer: str
    pts: list[list[float]]
    keepout_tracks: bool = False
    keepout_vias: bool = False


@dataclass
class FootprintRec:
    ref: str
    value: str
    x_mm: float
    y_mm: float
    orientation_deg: float
    layer: str
    courtyard_bbox: list[float] | None = None   # [x1,y1,x2,y2]


@dataclass
class Geometry:
    layer_names: list[str]
    outline_bbox: list[float]                 # [x1,y1,x2,y2] board bounding box
    outline: list[list[float]]                # [[x1,y1,x2,y2], ...] edge segments
    footprints: list[FootprintRec]
    pads: list[PadRec]
    tracks: list[TrackRec]
    vias: list[ViaRec]
    keepouts: list[ZoneRec]
    protected: list[str] = field(default_factory=list)   # object ids that must not change


@dataclass
class Anchor:
    pad_id: str
    x_mm: float
    y_mm: float
    layers: list[str]


@dataclass
class Cluster:
    id: str                    # "cluster:<net_code>:<k>"
    net_code: int
    anchors: list[Anchor]


@dataclass
class NetConnectivity:
    net_code: int
    net_name: str
    clusters: list[Cluster]
    remaining_edges: list[list[float]]   # ratsnest [[x1,y1,x2,y2], ...]
    pad_groups: int
    in_work_set: bool = False


@dataclass
class NetRule:
    net_code: int
    net_name: str
    class_name: str
    track_width_mm: float
    clearance_mm: float
    via_diameter_mm: float
    via_drill_mm: float
    layers: list[str]
    source: str = "kicad_pro"


@dataclass
class RuleContext:
    net_rules: list[NetRule]
    min_track_width_mm: float
    min_clearance_mm: float
    min_via_diameter_mm: float
    copper_edge_clearance_mm: float
    profiles: dict[str, dict]            # netclass name -> dims, the only profiles apply_profile accepts
    checker_coverage: list[str]
    source: str = "kicad_pro"


@dataclass
class ViewRecord:
    view_id: str
    snapshot_id: str
    layers: list[str]
    bbox_mm: list[float]
    image_size: list[int]                # [w, h] px
    pixel_to_mm: list[float]             # affine [a,b,c,d,e,f]: x=a*px+b*py+c ; y=d*px+e*py+f
    mirrored: bool
    path: str
    calibration: str = "exact"           # exact (rendered from snapshot) | landmark | uncalibrated
    object_map: dict[str, list[float]] = field(default_factory=dict)   # id -> pixel bbox


@dataclass
class ExecutionContext:
    completed_targets: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    budget_remaining: dict = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    frozen_scope: list[int] = field(default_factory=list)    # net codes the executor must not modify


@dataclass
class Snapshot:
    board: BoardRef
    geometry: Geometry
    connectivity: list[NetConnectivity]
    rules: RuleContext
    views: list[ViewRecord]
    execution: ExecutionContext
    work_set: list[int]

    # --- lookups -------------------------------------------------------
    def net(self, net_code: int) -> NetConnectivity | None:
        for n in self.connectivity:
            if n.net_code == net_code:
                return n
        return None

    def net_by_name(self, name: str) -> NetConnectivity | None:
        for n in self.connectivity:
            if n.net_name == name:
                return n
        return None

    def cluster(self, cluster_id: str) -> Cluster | None:
        for n in self.connectivity:
            for c in n.clusters:
                if c.id == cluster_id:
                    return c
        return None

    def rule_for(self, net_code: int) -> NetRule | None:
        for r in self.rules.net_rules:
            if r.net_code == net_code:
                return r
        return None

    def pad(self, pad_id: str) -> PadRec | None:
        for p in self.geometry.pads:
            if p.id == pad_id:
                return p
        return None
