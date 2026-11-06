"""Apply a GlobalGuidePatch to a plan with version / budget guards.

Rules
  * ``based_on.snapshot_id`` (and rules_hash when given) must equal the stable snapshot the
    patch was produced on — a patch against a stale state is refused, not merged.
  * ``based_on.plan_revision`` must equal the plan's current revision (no lost updates).
  * Only the listed edges / corridors / via regions are replaced; everything else is kept.
  * ``preserve`` edges are marked accepted and may not be replaced by the same patch.
  * Budget changes are explicit (``budget_revision``); a rollback never refunds spent calls.
  * Returns the affected edge ids so the caller can re-validate incrementally.
"""
from __future__ import annotations

import copy

from route_agent.schema.common import from_dict, to_dict
from route_agent.schema.feedback import GlobalGuidePatch
from route_agent.schema.guide import ConnectionEdge, Corridor, RouteGuidePlan, ViaRegion
from route_agent.schema.snapshot import Snapshot


class PatchRefused(ValueError):
    pass


def patch_schema_errors(d: dict) -> list[str]:
    """JSON-schema errors of a GlobalGuidePatch dict (PATCH_JSON_SCHEMA), as strings."""
    from route_agent.schema.feedback import PATCH_JSON_SCHEMA
    try:
        import jsonschema
    except ImportError:            # pragma: no cover - jsonschema is in the env
        return []
    v = jsonschema.Draft202012Validator(PATCH_JSON_SCHEMA)
    return [f"{'/'.join(str(x) for x in e.path) or '$'}: {e.message}" for e in sorted(v.iter_errors(d), key=lambda e: list(e.path))]


def apply_patch(plan: RouteGuidePlan, patch: GlobalGuidePatch, snap: Snapshot) -> tuple[RouteGuidePlan, list[str]]:
    b = patch.based_on or {}
    if not isinstance(b, dict):
        raise PatchRefused("patch based_on must be an object")
    if patch.budget_revision is not None and not isinstance(patch.budget_revision, dict):
        raise PatchRefused("patch budget_revision must be an object of budget fields")
    for name in ("replace_edges", "replace_corridors", "replace_via_regions", "preserve"):
        if not isinstance(getattr(patch, name), list):
            raise PatchRefused(f"patch {name} must be a list")
    if b.get("snapshot_id") != snap.board.snapshot_id:
        raise PatchRefused(f"patch based_on.snapshot_id {b.get('snapshot_id')!r} != stable snapshot {snap.board.snapshot_id!r}")
    if b.get("rules_hash") and b["rules_hash"] != snap.board.rules_hash:
        raise PatchRefused("patch rules_hash does not match the snapshot")
    if "plan_revision" in b and int(b["plan_revision"]) != plan.revision:
        raise PatchRefused(f"patch targets plan revision {b['plan_revision']} but plan is at {plan.revision}")
    preserved = set(patch.preserve)
    for ed in patch.replace_edges:
        if ed.get("id") in preserved:
            raise PatchRefused(f"edge {ed['id']} is both preserved and replaced")

    new = copy.deepcopy(plan)
    affected: list[str] = []
    for cd in patch.replace_corridors:
        c = from_dict(Corridor, cd)
        new.corridors = [x for x in new.corridors if x.id != c.id] + [c]
        affected += [e.id for _t, e in new.edges() if c.id in e.stages]
    for vd in patch.replace_via_regions:
        v = from_dict(ViaRegion, vd)
        new.via_regions = [x for x in new.via_regions if x.id != v.id] + [v]
        affected += [e.id for _t, e in new.edges() if v.id in e.via_regions]
    for ed in patch.replace_edges:
        e = from_dict(ConnectionEdge, ed)
        placed = False
        for t in new.connection_trees:
            for i, old in enumerate(t.edges):
                if old.id == e.id:
                    t.edges[i] = e
                    placed = True
        if not placed:
            net = int(ed.get("net_code", 0))
            tree = next((t for t in new.connection_trees if t.net_code == net), None)
            if tree is None:
                raise PatchRefused(f"edge {e.id} is new and names no existing tree (net_code {net})")
            tree.edges.append(e)
        affected.append(e.id)
    for _t, e in new.edges():
        if e.id in preserved:
            e.status = "accepted"
    if patch.net_order is not None:
        new.net_order = list(patch.net_order)
    if patch.budget_revision:
        for k, v in patch.budget_revision.items():
            if hasattr(new.budget, k):
                setattr(new.budget, k, int(v))
    new.revision = plan.revision + 1
    new.based_on.revision = plan.revision
    return new, sorted(set(affected))


def plan_to_dict(plan: RouteGuidePlan) -> dict:
    return to_dict(plan)


def is_patch(d: dict) -> bool:
    return isinstance(d, dict) and ("patch_id" in d or any(k in d for k in ("replace_edges", "replace_corridors", "replace_via_regions")))


def merge_global_response(response: dict, plan: RouteGuidePlan, snap: Snapshot, budget=None):
    """Turn a Global replan answer into the next plan.

    The answer may be a ``GlobalGuidePatch`` (preferred: version-guarded, only the listed items
    change, ``budget_revision`` is applied to ``budget`` explicitly) or a full ``RouteGuidePlan``
    (accepted; it must validate on its own against the stable snapshot). Returns
    ``(new_plan | None, report, affected_edge_ids | None, kind)``; ``affected`` is ``None`` for a
    full plan, meaning "recompile every edge that is not done".
    """
    from route_agent.guide.validate import ValidationReport, validate_guide, validate_guide_dict
    notes: list[str] = []
    if is_patch(response):
        if "budget_revision" in response and not isinstance(response["budget_revision"], (dict, type(None))):
            notes.append(f"budget_revision {response['budget_revision']!r} ignored (not an object)")
            response = dict(response, budget_revision=None)
        # a via region written without `layers` (the commonest model slip) is completed from the
        # stages it joins by repair_guide after the merge; let it through the dataclass first
        vrs = []
        for v in response.get("replace_via_regions", []) or []:
            if isinstance(v, dict) and "layers" not in v:
                notes.append(f"via region {v.get('id')}: layers taken from its stages")
                v = dict(v, layers=[])
            vrs.append(v)
        if vrs:
            response = dict(response, replace_via_regions=vrs)
        errs = patch_schema_errors(response)
        if errs:
            return None, ValidationReport(errors=[f"patch schema: {e}" for e in errs[:8]]), None, "patch"
        try:
            patch = from_dict(GlobalGuidePatch, response)
            new_plan, affected = apply_patch(plan, patch, snap)
        except (PatchRefused, KeyError, TypeError, ValueError, AttributeError) as ex:
            return None, ValidationReport(errors=[f"patch refused: {ex}"]), None, "patch"
        # deterministic repair of the merged plan (via regions onto stage overlaps, their layers,
        # normalized boxes) before the semantic checks — the same step `global --repair` applies
        from route_agent.guide.repair import repair_guide
        from route_agent.guide.validate import parse_guide
        repaired, rnotes = repair_guide(to_dict(new_plan), snap)
        if rnotes:
            notes.extend(rnotes)
            fixed, errs2 = parse_guide(repaired)
            if fixed is None:
                return None, ValidationReport(errors=[f"patch repair: {e}" for e in errs2[:5]]), affected, "patch"
            new_plan = fixed
        rep = validate_guide(new_plan, snap)
        rep.warnings.extend(notes)
        if not rep.ok:
            return None, rep, affected, "patch"
        if budget is not None and patch.budget_revision:
            budget.revise(**patch.budget_revision)
        return new_plan, rep, affected, "patch"
    new_plan, rep = validate_guide_dict(response, snap)
    return (new_plan if rep.ok else None), rep, None, "plan"


def overlay_patch(prev: dict, fix: dict) -> dict:
    """The repair round's answer laid over the patch it corrects.

    Items are merged by id and the fix wins. A Global that answers the validation errors with
    only the broken items — the measured case: the corridor it forgot to define and the two
    edges that name it — thus keeps the rest of its first patch, instead of having two edges
    checked against a plan that never saw the other 25 corridors. When either side is a full
    plan rather than a patch the fix is returned as it is.
    """
    if not (is_patch(prev) and is_patch(fix)):
        return fix
    out = dict(prev)
    for key in ("replace_corridors", "replace_via_regions", "replace_edges"):
        by_id: dict = {}
        odd: list = []
        for item in list(prev.get(key) or []) + list(fix.get(key) or []):
            if isinstance(item, dict) and item.get("id") is not None:
                by_id[item["id"]] = item
            else:
                odd.append(item)                  # malformed: kept, the schema check names it
        out[key] = list(by_id.values()) + odd
    pv = prev.get("preserve") if isinstance(prev.get("preserve"), list) else []
    fv = fix.get("preserve") if isinstance(fix.get("preserve"), list) else []
    out["preserve"] = sorted(set(pv) | set(fv), key=str)
    for key in ("patch_id", "based_on", "note", "net_order", "budget_revision"):
        if fix.get(key) is not None:
            out[key] = fix[key]
    return out


def merge_repair_response(prev: dict, fix: dict, plan: RouteGuidePlan, snap: Snapshot, budget=None):
    """``merge_global_response`` for a repair round.

    Tries the fix laid over the previous answer first (``overlay_patch``), then the fix on its
    own — a Global that re-sent a complete corrected patch and dropped an item the checks
    refused. Returns the usual ``(new_plan, report, affected, kind)`` plus ``overlaid``: whether
    the merged result was the overlay. When both fail the overlay's report is returned, it being
    the fuller attempt.
    """
    overlay = overlay_patch(prev, fix)
    if overlay is fix:
        return merge_global_response(fix, plan, snap, budget) + (False,)
    r = merge_global_response(overlay, plan, snap, budget)
    if r[0] is not None:
        return r + (True,)
    r2 = merge_global_response(fix, plan, snap, budget)
    if r2[0] is not None:
        return r2 + (False,)
    return r + (True,)
