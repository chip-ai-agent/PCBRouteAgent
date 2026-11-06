"""Official PCBWorld scoring of a saved board — the only numbers PCBRouteAgent reports.

``official_score`` wraps ``eval.metrics.evaluate_one`` (the paper's post-hoc / Stage-2 scoring
entry point: load the saved ``.kicad_pcb`` in a fresh engine, run DRC, reset destructively to
recover the routability baseline) and trims its result to the fields we tabulate. Every field
is the evaluator's own except the two marked *derived*, which only split the evaluator's
``drv_errors_only_count`` into "Missing connection between items" (KiCad DRCE_UNCONNECTED_ITEMS,
the ratsnest) and everything else; the official ``clean_pass`` (= all connected and zero
errors-only DRVs) is reported unchanged.

Note the evaluator scores the WHOLE board: with a partial work set the other nets count as
unconnected, so ``success`` / ``clean_pass`` are only meaningful for whole-board runs.
"""
from __future__ import annotations

import os
from typing import Callable

DEFAULT_CHECK_ANGLE = 45


def _pro_for(pcb: str) -> str | None:
    pro = os.path.splitext(str(pcb))[0] + ".kicad_pro"
    return pro if os.path.exists(pro) else None


def _is_unconnected_bucket(b: dict) -> bool:
    return int(b.get("error_code", 0)) == 1 or "missing connection" in str(b.get("error_type", "")).lower() \
        or "unconnected" in str(b.get("error_type", "")).lower()


def trim_metrics(m: dict, *, check_angle: int = DEFAULT_CHECK_ANGLE, keep_violations: int = 20) -> dict:
    """Pure: evaluator result dict -> the tabulated subset (testable without an engine)."""
    ex = m["extras"]
    by_type = list(m["drv_breakdown"]["errors_only_by_type"])
    unconnected = sum(int(b["count"]) for b in by_type if _is_unconnected_bucket(b))
    total = int(ex["initial_unrouted_edges"])
    remaining = int(ex["unrouted_edges_remaining"])
    return {
        "source": "eval.metrics.evaluate_one",
        "reward_config": m.get("reward_config"),
        "check_angle": int(check_angle),
        "success": bool(m["success"]),
        "clean_pass": bool(m["clean_pass"]),
        "routability": float(m["routability"]) if m.get("routability") is not None else None,
        "potential_gain": float(m["potential_gain"]) if m.get("potential_gain") is not None else None,
        "final_potential": float(m["final_potential"]) if m.get("final_potential") is not None else None,
        "edges_total": total,
        "edges_routed": total - remaining,
        "edges_remaining": remaining,
        "pad_groups_initial": int(ex["pad_groups_initial"]),
        "pad_groups_remaining": int(ex["pad_groups_remaining"]),
        "drv_errors_only": int(m["drv_errors_only_count"]),
        "drv_errors_and_promoted": int(m["drv_errors_and_promoted_count"]),
        "drv_unconnected": int(unconnected),                                   # derived split
        "drv_rule_errors": int(m["drv_errors_only_count"]) - int(unconnected),  # derived split
        "track_angle_drv": int(m["track_angle_drv"]["count"]),
        "total_drv": int(m["total_drv_count"]),
        "wirelength_mm": round(float(m["wirelength_mm"]), 3),
        "via_count": int(m["via_count"]),
        "track_count": m.get("track_count"),
        "errors_by_type": by_type,
        "track_angle_violations": list(m["track_angle_drv"]["violations"])[:keep_violations],
        "per_net": ex.get("per_net"),
    }


PAPER_REWARD_CONFIG = "drc_dense_errors_only_eval"   # the KDD paper's evaluation potential (Table 3 Pot.)


def official_score(routed_pcb: str, pro_path: str | None = None, *, check_angle: int = DEFAULT_CHECK_ANGLE,
                   keep_violations: int = 20, reward_config: str | None = None, _evaluate: Callable | None = None) -> dict:
    """Score a saved board with the official evaluator. ``reward_config`` selects the potential
    function (``None`` = the repo default; ``PAPER_REWARD_CONFIG`` = the paper's tables); CP, Rout.,
    DRV, WL and Via do not depend on it. ``_evaluate`` is injectable for tests."""
    if _evaluate is None:
        from eval.metrics import evaluate_one as _evaluate
    pro = pro_path or _pro_for(routed_pcb)
    kw = {"check_angle": check_angle}
    if reward_config:
        kw["reward_config_name"] = reward_config
    m = _evaluate(str(routed_pcb), str(pro) if pro else None, **kw)
    out = trim_metrics(m, check_angle=check_angle, keep_violations=keep_violations)
    out["board"] = str(routed_pcb)
    out["project"] = str(pro) if pro else None
    return out


def headline(o: dict) -> str:
    """One line for logs / READMEs."""
    return (f"official: edges {o['edges_routed']}/{o['edges_total']}, pad groups {o['pad_groups_initial']}->"
            f"{o['pad_groups_remaining']}, DRV errors {o['drv_errors_only']} (unconnected {o['drv_unconnected']}, "
            f"rule {o['drv_rule_errors']}), track-angle {o['track_angle_drv']}, wirelength {o['wirelength_mm']} mm, "
            f"vias {o['via_count']}, routability {o.get('routability')}, potential gain {o.get('potential_gain')}, "
            f"success={o['success']}, clean_pass={o['clean_pass']}")
