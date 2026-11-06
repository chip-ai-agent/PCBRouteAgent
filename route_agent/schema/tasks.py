"""DetailedSubtask: what the compiler hands to the Detailed role for one connection edge."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageSpec:
    corridor_id: str
    layer: str
    bbox: list[float]
    soft: bool = True
    entry: list[float] | None = None    # portal into this stage (intersection centre), None for the first
    exit: list[float] | None = None     # portal out of this stage, None for the last


@dataclass
class TransitionSpec:
    from_stage: str
    to_stage: str
    via_region_id: str
    from_layer: str
    to_layer: str
    bbox: list[float]                   # common intersection: stage_i ∩ stage_{i+1} ∩ via_region


@dataclass
class DetailedSubtask:
    task_id: str
    net_code: int
    net_name: str
    source_cluster: str
    target_cluster: str
    source_anchors: list[dict]          # [{pad_id,x_mm,y_mm,layers}]
    target_anchors: list[dict]
    stages: list[StageSpec]
    transitions: list[TransitionSpec]
    profile: dict                       # resolved NetRule dims
    dependencies: dict
    budget: dict
    order: int = 0
    execution: dict = field(default_factory=lambda: {"status": "pending", "supported": True, "reason": ""})
