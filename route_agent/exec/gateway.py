"""Tool gateway: the agent-facing tool surface over ``NativeAdapter``.

Each tool has a spec (group, inputs, outputs, native mapping, status). ``ToolGateway.call``
enforces: known tool, implemented status, phase preconditions, StateRef freshness,
frozen-scope protection, and budget charging. The executable subset is what
``tools_schema()`` exports.

Groups follow the design: Observe / Route / Configure / Edit / Transaction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from route_agent.exec.adapter import CAPABILITIES, NativeAdapter, UNSUPPORTED
from route_agent.loop.budget import Budget, BudgetExhausted


@dataclass
class ToolSpec:
    name: str
    group: str
    description: str
    inputs: dict                         # JSON schema "properties"
    required: list[str]
    outputs: list[str]
    native: list[str]                    # engine methods behind it
    status: str = "implemented"          # implemented | proposed
    mutating: bool = False
    needs_session: bool | None = None    # True: needs open route; False: needs idle; None: any
    needs_net: bool = False


_P = {
    "x": {"type": "number", "description": "x in board mm"},
    "y": {"type": "number", "description": "y in board mm"},
    "layer": {"type": "string", "description": "copper layer name, e.g. F.Cu / B.Cu"},
    "mode": {"type": "integer", "enum": [1, 2], "description": "1 = shove, 2 = walkaround"},
    "net_code": {"type": "integer"},
    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
    "state_ref": {"type": "integer", "description": "board revision this request was planned on; stale -> refused"},
}

TOOL_SPECS: list[ToolSpec] = [
    # Observe
    ToolSpec("routing_state", "observe", "Session + head + layer + revision.", {}, [], ["is_routing", "head", "current_layer", "revision"], ["get_routing_session_state"]),
    ToolSpec("inspect_net", "observe", "Remaining ratsnest, pad groups, copper of one net.", {"net_code": _P["net_code"]}, ["net_code"], ["remaining_edges", "pad_groups", "tracks", "vias"], ["build_connectivity", "get_ratsnest", "get_pad_groups", "get_tracks", "get_vias"]),
    ToolSpec("inspect_region", "observe", "Pads / tracks / vias inside a bbox, optional layer.", {"bbox": _P["bbox"], "layer": _P["layer"]}, ["bbox"], ["pads", "tracks", "vias"], ["get_pads", "get_tracks", "get_vias"]),
    ToolSpec("connected_points", "observe", "Anchors already joined to the copper at a point.", {"x": _P["x"], "y": _P["y"], "layer": _P["layer"]}, ["x", "y", "layer"], ["points"], ["get_connected_points"]),
    ToolSpec("get_rules", "observe", "Board minimums, allowed profiles, net class.", {"net_code": _P["net_code"]}, [], ["profiles", "net_class"], ["get_design_rules", "get_netclass_for_net"]),
    ToolSpec("run_drc", "observe", "Scoped DRC (target nets only).", {"target_net_names": {"type": "array", "items": {"type": "string"}}}, [], ["count", "errors", "violations"], ["run_drc"], needs_session=False),
    # Route
    ToolSpec("select_net", "route", "Select an unconnected net for routing.", {"net_code": _P["net_code"]}, ["net_code"], ["ok"], ["get_ratsnest"], needs_session=False),
    ToolSpec("start_route", "route", "Open a routing session at a pad / copper anchor on a layer.", {"x": _P["x"], "y": _P["y"], "layer": _P["layer"], "state_ref": _P["state_ref"]}, ["x", "y", "layer"], ["ok"], ["start_route"], mutating=True, needs_session=False, needs_net=True),
    ToolSpec("route_to", "route", "Commit a segment to (x,y) on the current layer; closes the session.", {"x": _P["x"], "y": _P["y"], "mode": _P["mode"], "state_ref": _P["state_ref"]}, ["x", "y"], ["ok", "native_reason"], ["set_routing_mode", "pad_block_reason", "fix_route"], mutating=True, needs_session=True),
    ToolSpec("route_via", "route", "Commit a segment to (x,y) and a via there; head continues on the other layer; closes the session.", {"x": _P["x"], "y": _P["y"], "mode": _P["mode"], "state_ref": _P["state_ref"]}, ["x", "y"], ["ok", "native_reason", "via_placed"], ["toggle_via", "fix_route", "get_via_count"], mutating=True, needs_session=True),
    ToolSpec("finish", "route", "Let PNS close the route to the nearest unconnected pad of the net.", {"mode": _P["mode"], "state_ref": _P["state_ref"]}, [], ["ok"], ["finish"], mutating=True, needs_session=True),
    ToolSpec("cancel", "route", "Drop the open session without committing.", {}, [], ["ok"], ["cancel_route"]),
    ToolSpec("undo_last", "route", "Undo the last committed segment of the open session.", {}, [], ["ok"], ["undo_last_segment"], mutating=True, needs_session=True),
    ToolSpec("flip_posture", "route", "Flip the 45° posture of the pending segment.", {}, [], ["ok"], ["flip_posture"], needs_session=True),
    ToolSpec("end_net", "route", "Deselect the current net.", {}, [], ["ok", "remaining"], ["build_connectivity", "get_ratsnest"], needs_session=False, needs_net=True),
    # Configure
    ToolSpec("apply_profile", "configure", "Apply a task-allowed width/via profile by name.", {"name": {"type": "string"}}, ["name"], ["ok", "applied"], ["set_track_width", "set_via_diameter", "set_via_drill"], needs_session=False),
    ToolSpec("set_via_dims", "configure", "Set via diameter / drill; refused when smaller (diameter, drill or annular ring) than every allowed profile.", {"diameter_mm": {"type": "number"}, "drill_mm": {"type": "number"}}, ["diameter_mm", "drill_mm"], ["ok", "applied"], ["set_via_diameter", "set_via_drill"], needs_session=False),
    ToolSpec("set_corner_mode", "configure", "PNS corner style: 0 mitered 45 (default), 1 rounded 45, 2 mitered 90, 3 rounded 90.", {"mode": {"type": "integer", "enum": [0, 1, 2, 3]}}, ["mode"], ["ok", "applied"], ["set_corner_mode"], needs_session=False),
    # Edit / repair
    ToolSpec("delete_net_routing", "edit", "Rip up all copper of the given nets (frozen scope refused).", {"net_codes": {"type": "array", "items": {"type": "integer"}}, "state_ref": _P["state_ref"]}, ["net_codes"], ["ok", "removed"], ["delete_routing_of_nets"], mutating=True, needs_session=False),
    ToolSpec("lock_net", "edit", "Lock a net so shove walks around it.", {"net_code": _P["net_code"], "locked": {"type": "boolean"}}, ["net_code"], ["ok", "changed"], ["lock_net"], needs_session=False),
    ToolSpec("cleanup", "edit", "KiCad track cleaner (merge; finalize adds dangling passes).", {"net_codes": {"type": "array", "items": {"type": "integer"}}, "dry_run": {"type": "boolean"}, "finalize": {"type": "boolean"}}, [], ["ok", "removed", "modified"], ["cleanup_tracks"], mutating=True, needs_session=False),
    # Transaction
    ToolSpec("checkpoint", "transaction", "IDLE checkpoint of board + session.", {}, [], ["handle", "revision"], ["checkpoint"], needs_session=False),
    ToolSpec("rollback", "transaction", "Restore a checkpoint (board + wrapper state).", {"handle": {"type": "integer"}}, ["handle"], ["ok", "revision"], ["restore", "build_connectivity"], mutating=True),
    ToolSpec("save", "transaction", "Durable save of board + project.", {"output_path": {"type": "string"}}, ["output_path"], ["board", "project"], ["save"], needs_session=False),
    # Proposed (not wired to the pinned binding through this adapter)
    ToolSpec("preview_route", "route", "Dry-run a segment without committing.", {"x": _P["x"], "y": _P["y"]}, ["x", "y"], ["head"], ["move", "get_route_head"], status="proposed", needs_session=True),
    ToolSpec("drag_segment", "edit", "Drag an existing segment / via.", {"x": _P["x"], "y": _P["y"], "layer": _P["layer"]}, ["x", "y", "layer"], ["ok"], ["start_drag", "fix_drag"], status="proposed"),
    ToolSpec("reopen_net", "edit", "Re-select a closed net for repair.", {"net_code": _P["net_code"]}, ["net_code"], ["ok"], ["get_ratsnest"], status="proposed"),
    ToolSpec("failure_evidence", "observe", "Native blocker query for the last failed commit.", {}, [], ["native_reason", "blockers"], [], status="proposed"),
    ToolSpec("diff_pair_route", "route", "Differential pair routing.", {}, [], [], [], status="proposed"),
    ToolSpec("length_tune", "route", "Length / skew tuning.", {}, [], [], [], status="proposed"),
]
SPEC_BY_NAME = {t.name: t for t in TOOL_SPECS}


class ToolError(RuntimeError):
    pass


@dataclass
class ToolGateway:
    adapter: NativeAdapter
    budget: Budget | None = None
    allowed_profiles: dict | None = None
    frozen_scope: list[int] = field(default_factory=list)
    target_net_names: list[str] | None = None
    trace: list[dict] = field(default_factory=list)
    exhausted: str | None = None            # set once the executor budget refused a call; never cleared
    _recovery: bool = field(default=False, repr=False)
    _local: bool = field(default=False, repr=False)

    def implemented(self) -> list[str]:
        return [t.name for t in TOOL_SPECS if t.status == "implemented"]

    def call_recovery(self, tool: str, **args: Any) -> dict:
        """Orchestrator-only path for cancel / rollback / end_net / run_drc after a failure or at
        the end of a run: charged from the recovery reserve (``Budget.charge_recovery``) instead
        of the executor pool, so the stable snapshot can always be restored and checked."""
        self._recovery = True
        try:
            return self.call(tool, **args)
        finally:
            self._recovery = False

    def call_local(self, tool: str, **args: Any) -> dict:
        """Executor-side transaction calls (stage sub-checkpoints and their rollbacks): charged to
        the normal executor pool, so the orchestrator's recovery reserve stays untouched."""
        self._local = True
        try:
            return self.call(tool, **args)
        finally:
            self._local = False

    def call(self, tool: str, **args: Any) -> dict:
        spec = SPEC_BY_NAME.get(tool)
        if spec is None:
            raise ToolError(f"unknown tool {tool!r}")
        if spec.status != "implemented":
            return self._rec(tool, args, {"ok": False, "reason": f"tool {tool} is proposed, not wired to the pinned binding"})
        missing = [r for r in spec.required if r not in args]
        if missing:
            return self._rec(tool, args, {"ok": False, "reason": f"missing args {missing}"})
        # freshness
        sref = args.pop("state_ref", None)
        if sref is not None and int(sref) != self.adapter.state.revision:
            return self._rec(tool, args, {"ok": False, "reason": "stale state_ref",
                                          "expected": self.adapter.state.revision, "got": int(sref)})
        # phase
        if spec.needs_session is not None or spec.needs_net:
            st = self.adapter.e.is_routing()
            if spec.needs_session is True and not st:
                return self._rec(tool, args, {"ok": False, "reason": "needs an open routing session"})
            if spec.needs_session is False and st:
                return self._rec(tool, args, {"ok": False, "reason": "needs an idle session (cancel/finish first)"})
            if spec.needs_net and self.adapter.current_net is None:
                return self._rec(tool, args, {"ok": False, "reason": "no net selected"})
        # scope
        if tool == "delete_net_routing":
            args["frozen"] = list(self.frozen_scope)
        if tool == "apply_profile":
            args = {"profile": {"name": args["name"]}, "allowed": self.allowed_profiles}
        if tool == "set_via_dims":
            args["allowed"] = self.allowed_profiles
        if tool == "run_drc" and "target_net_names" not in args:
            args["target_net_names"] = self.target_net_names
        # budget
        if self.budget is not None:
            try:
                if (spec.group == "transaction" or tool == "cancel" or self._recovery) and not self._local:
                    self.budget.charge_recovery(1)
                else:
                    self.budget.charge(1, tool)
            except BudgetExhausted as ex:
                # structured for an LLM caller; executors raise BudgetExhausted on ``exhausted``
                self.exhausted = self.exhausted or str(ex)
                return self._rec(tool, args, {"ok": False, "reason": str(ex), "budget_exhausted": True,
                                              "budget": self.budget.snapshot()})
        fn: Callable = getattr(self.adapter, tool)
        try:
            out = fn(**args)
        except Exception as ex:          # native exceptions become structured failures
            out = {"ok": False, "reason": f"{type(ex).__name__}: {ex}"}
        if isinstance(out, list):
            out = {"ok": True, "points": out}
        out.setdefault("revision", self.adapter.state.revision)
        return self._rec(tool, args, out)

    def _rec(self, name, args, out) -> dict:
        self.trace.append({"tool": name, "args": {k: v for k, v in args.items() if k not in ("allowed", "frozen")},
                           "ok": bool(out.get("ok", True)), "reason": out.get("reason", out.get("native_reason", ""))})
        return out


def tools_schema(include_proposed: bool = False) -> dict:
    tools = []
    for t in TOOL_SPECS:
        if t.status != "implemented" and not include_proposed:
            continue
        tools.append({"name": t.name, "group": t.group, "status": t.status, "description": t.description,
                      "parameters": {"type": "object", "properties": t.inputs, "required": t.required},
                      "outputs": t.outputs, "native_mapping": t.native, "mutating": t.mutating,
                      "needs_session": t.needs_session, "needs_net": t.needs_net})
    return {"version": 1, "capabilities": list(CAPABILITIES), "unsupported": list(UNSUPPORTED), "tools": tools}


def openai_tools(include_proposed: bool = False) -> list[dict]:
    """Function-calling format for the LLM Detailed role."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["parameters"]}}
            for t in tools_schema(include_proposed)["tools"]]


def write_tools_schema(path: str) -> None:
    with open(path, "w") as f:
        json.dump(tools_schema(include_proposed=True), f, indent=2)
        f.write("\n")
