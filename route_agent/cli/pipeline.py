"""PCBRouteAgent pipeline CLI.

    python -m route_agent.cli.pipeline snapshot BOARD --nets /CLK /LE /DATA --out DIR [--pro PRO] [--native]
    python -m route_agent.cli.pipeline validate-guide SNAPSHOT GUIDE
    python -m route_agent.cli.pipeline compile SNAPSHOT GUIDE --output TASKS
    python -m route_agent.cli.pipeline execute SNAPSHOT GUIDE --out DIR [--save routed.kicad_pcb]
    python -m route_agent.cli.pipeline global SNAPSHOT --provider openai|anthropic --out GUIDE
    python -m route_agent.cli.pipeline bench BOARD --out DIR [--provider anthropic]
    python -m route_agent.cli.pipeline benchset --list LIST --out DIR [--provider anthropic]
    python -m route_agent.cli.pipeline score ROUTED.kicad_pcb
    python -m route_agent.cli.pipeline status [--repo PATH]
"""
from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path

from route_agent.paths import data_root, ensure_pcbworld_on_path, pcbworld_root
from route_agent.schema.common import dump_json, from_dict, load_json, to_dict
from route_agent.schema.snapshot import Snapshot


def _load_snapshot(path: str) -> Snapshot:
    return from_dict(Snapshot, load_json(path))


def _fill_guide_placeholders(guide: dict, snap: Snapshot) -> dict:
    """The design-example guide carries __SNAPSHOT_ID__ / __RULES_HASH__ placeholders so it
    binds to whatever snapshot is rebuilt from the same board state."""
    b = guide.get("based_on", {})
    if b.get("snapshot_id") == "__SNAPSHOT_ID__":
        b["snapshot_id"] = snap.board.snapshot_id
    if b.get("rules_hash") == "__RULES_HASH__":
        b["rules_hash"] = snap.board.rules_hash
    return guide


def cmd_snapshot(a) -> int:
    from route_agent.snapshot.builder import build_snapshot, open_board
    from route_agent.snapshot.views import export_native_layer_svgs, native_view_records, render_views
    lazy, adapter = open_board(a.board, a.pro)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        snap = build_snapshot(adapter, board_id=a.board_id or Path(a.board).parent.name, board_path=a.board,
                              project_path=lazy.project_path, work_set_names=a.nets, repo_root=pcbworld_root(),
                              preparation=a.preparation)
        views = render_views(snap, out / "views", px_per_mm=a.px_per_mm)
        native = export_native_layer_svgs(a.board, out / "views", snap.geometry.layer_names, pcbworld_root()) if a.native else []
        views += native_view_records(snap, native)
        snap.board.extra["native_exports"] = native
        dump_json(snap, out / "snapshot.json")
        dump_json(views, out / "view_records.json")
    finally:
        lazy.close()
    print(json.dumps({"snapshot_id": snap.board.snapshot_id, "rules_hash": snap.board.rules_hash,
                      "work_set": snap.work_set, "pads": len(snap.geometry.pads), "tracks": len(snap.geometry.tracks),
                      "vias": len(snap.geometry.vias), "nets": len(snap.connectivity),
                      "views": [v.view_id for v in views], "native": [n["layer"] for n in native]}, indent=2))
    return 0


def cmd_validate(a) -> int:
    from route_agent.guide.validate import validate_guide_dict
    snap = _load_snapshot(a.snapshot)
    guide = _fill_guide_placeholders(load_json(a.guide), snap)
    plan, rep = validate_guide_dict(guide, snap)
    print(json.dumps(rep.to_dict(), indent=2))
    if a.output:
        dump_json(rep.to_dict(), a.output)
    return 0 if rep.ok else 1


def cmd_compile(a) -> int:
    from route_agent.guide.compiler import compile_plan, compile_summary
    from route_agent.guide.validate import validate_guide_dict
    snap = _load_snapshot(a.snapshot)
    guide = _fill_guide_placeholders(load_json(a.guide), snap)
    plan, rep = validate_guide_dict(guide, snap)
    if not rep.ok:
        print(json.dumps(rep.to_dict(), indent=2))
        return 1
    tasks = compile_plan(plan, snap)
    summary = compile_summary(tasks)
    if a.output:
        dump_json({"summary": summary, "plan_id": plan.plan_id, "tasks": tasks}, a.output)
    print(json.dumps(summary, indent=2))
    for t in tasks:
        print(f"DetailedSubtask {t.task_id}  net {t.net_code} {t.net_name}  {t.source_cluster} -> {t.target_cluster}")
        for s in t.stages:
            print(f"    {s.corridor_id:10s} {s.layer:5s} {s.bbox}")
        for tr in t.transitions:
            print(f"    transition {tr.from_stage}->{tr.to_stage} via {tr.via_region_id} {tr.bbox}")
        print(f"    execution: {t.execution}")
    return 0


def _run_execute(snap: Snapshot, guide: dict, out: Path, *, save: str | None, provider=None,
                 mode: int = 2, max_global_rounds: int = 12, on_unroutable: str = "stop",
                 finish_first: bool = True, shove_fallback: bool = True, obstacle_aware: bool = True,
                 corner_mode: int | None = 0, final_cleanup: bool = True, backtrack: int = 2, ripup_cap: int = 1,
                 replans_per_edge: int = 4, reward_config: str | None = None, repair_rounds: int = 1,
                 maze: int = 1, alt_anchors: int = 2, batch_replan: bool = True, batch_rounds: int = 5,
                 edge_history: bool = True, angle_repair: bool = True) -> dict:
    from route_agent.exec.executor import Orchestrator, ScriptedDetailed
    from route_agent.exec.gateway import ToolGateway
    from route_agent.guide.compiler import compile_plan, compile_summary
    from route_agent.guide.patch import merge_global_response, merge_repair_response
    from route_agent.guide.validate import validate_guide_dict
    from route_agent.loop.budget import Budget
    from route_agent.prompts.bundle import build_global_bundle
    from route_agent.snapshot.builder import open_board

    plan, rep = validate_guide_dict(guide, snap)
    if not rep.ok:
        return {"ok": False, "validation": rep.to_dict()}
    lazy, adapter = open_board(snap.board.board_path, snap.board.project_path)
    try:
        eng = adapter.e
        eng.set_target_nets(snap.work_set)
        names = [n.net_name for n in snap.connectivity if n.in_work_set]
        budget = Budget.from_plan(plan.budget)
        gw = ToolGateway(adapter, budget=budget, allowed_profiles=snap.rules.profiles,
                         frozen_scope=snap.execution.frozen_scope, target_net_names=names)
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id, mode=mode, finish_first=finish_first,
                               shove_fallback=shove_fallback, obstacle_aware=obstacle_aware, backtrack=backtrack, maze=maze,
                               alt_anchors=alt_anchors)
        for code in snap.execution.frozen_scope:          # shove must walk around frozen nets, never move them
            gw.call("lock_net", net_code=int(code), locked=True)
        if corner_mode is not None:
            gw.call("set_corner_mode", mode=int(corner_mode))
        state = {"plan": plan, "rounds": 0, "usage": []}

        def live_state() -> dict:
            """What changed since the stable snapshot: the board is rolled back to it plus the edges
            already accepted, so the plan keeps addressing the original cluster ids while the
            Global sees the current connectivity of every work-set net."""
            done = [t["task_id"] for t in state["orch"].report.tasks if t["outcome"] == "done"] if state.get("orch") else []
            nets = {}
            for code in snap.work_set:
                info = adapter.inspect_net(int(code))
                nets[str(code)] = {"pad_groups": info["pad_groups"], "remaining_edges": len(info["remaining_edges"]),
                                   "tracks": len(info["tracks"]), "vias": len(info["vias"])}
            return {"done_edges": done, "nets": nets, "revision": adapter.state.revision}

        def replan_fn(packet, decision):
            if provider is None or state["rounds"] >= max_global_rounds:
                return None
            state["rounds"] += 1
            if isinstance(packet, list):                 # end-of-pass batch: every failed edge's packet at once
                feedback = {"packets": [to_dict(p) for p in packet], "decision": decision.__dict__, "live_state": live_state()}
            else:
                feedback = {"packet": to_dict(packet), "decision": decision.__dict__, "live_state": live_state()}
            bundle = build_global_bundle(snap, feedback=feedback, previous_plan=to_dict(state["plan"]))
            response, usage = provider.replan(bundle)
            state["usage"].append(usage)
            response = _fill_guide_placeholders(response, snap)
            new_plan, rep2, affected, kind = merge_global_response(response, state["plan"], snap, budget)
            rounds = [{"kind": kind, "response": response, "validation": rep2.to_dict(), "affected": affected,
                       "usage": {k: v for k, v in usage.items() if k != "raw"}}]
            if new_plan is None and repair_rounds > 0:
                # the merged patch failed validation: hand the errors back once, same feedback, and merge again
                fb2 = dict(feedback, previous_response=response, validation_errors=rep2.errors[:20],
                           instruction="Your previous patch failed the checks listed in validation_errors. Return a corrected "
                                       "GlobalGuidePatch for the same edge(s); keep everything else unchanged.")
                bundle2 = build_global_bundle(snap, feedback=fb2, previous_plan=to_dict(state["plan"]))
                response2, usage2 = provider.replan(bundle2)
                state["usage"].append(usage2)
                response2 = _fill_guide_placeholders(response2, snap)
                # the fix is laid over the first answer before the checks (a Global usually answers
                # with only the items it got wrong), and tried alone when the overlay still fails
                new_plan, rep2, affected, kind, overlaid = merge_repair_response(response, response2, state["plan"], snap, budget)
                rounds.append({"kind": kind, "response": response2, "validation": rep2.to_dict(), "affected": affected,
                               "usage": {k: v for k, v in usage2.items() if k != "raw"}, "repair_round": True,
                               "overlaid_on_previous": overlaid})
            dump_json(rounds[-1] | {"rounds": rounds}, out / f"replan_{state['rounds']}.json")
            if new_plan is None:
                return None
            state["plan"], state["affected"] = new_plan, affected
            return new_plan

        def recompile_fn(new_plan):
            return compile_plan(new_plan, snap, edges_only=state.get("affected"))

        tasks = compile_plan(plan, snap)
        orch = Orchestrator(gw, det, budget, stall_threshold=plan.budget.stall_threshold,
                            replan_fn=replan_fn if provider else None, recompile_fn=recompile_fn, target_net_names=names,
                            on_unroutable=on_unroutable, ripup_cap=ripup_cap, rippable_nets=set(int(c) for c in snap.work_set),
                            replans_per_edge=replans_per_edge, batch_replan=batch_replan, batch_rounds=batch_rounds,
                            edge_history=edge_history)
        state["orch"] = orch
        t0 = time.time()
        report = orch.run(tasks)
        angles = None
        if angle_repair:
            # joints between two committed hops that are not 45/90 degrees (the evaluator's
            # track-angle marks, acute ones included) are re-routed as one PNS move; each repair
            # is kept only if connectivity, scoped DRC and the net's copper length hold
            from route_agent.exec.angles import guarded_repair_passes
            angles = guarded_repair_passes(adapter, [int(c) for c in snap.work_set],
                                          {int(c): (snap.net(int(c)).net_name if snap.net(int(c)) else None) for c in snap.work_set},
                                          mode=mode, final_cleanup=final_cleanup)
        cleanup = None
        if final_cleanup:
            # KiCad's own track cleaner on the work-set nets: merge collinear segments, drop dangling
            # stubs / vias. Topology-preserving; what a designer runs before saving.
            cleanup = gw.call_recovery("cleanup", net_codes=[int(c) for c in snap.work_set], dry_run=False, finalize=True)
        final_nets = {}
        for code in snap.work_set:
            info = adapter.inspect_net(int(code))
            final_nets[str(code)] = {"net_name": (snap.net(int(code)).net_name if snap.net(int(code)) else ""),
                                     "pad_groups": info["pad_groups"], "remaining_edges": len(info["remaining_edges"]),
                                     "tracks": len(info["tracks"]), "vias": len(info["vias"])}
        result = {"ok": True, "plan_id": plan.plan_id, "compile": compile_summary(tasks),
                  "tasks": report.tasks, "events": [e.__dict__ for e in report.events], "replans": report.replans,
                  "stopped": report.stopped, "budget": report.budget, "final": report.final, "final_nets": final_nets,
                  "cleanup": cleanup, "angle_repair": angles, "corner_mode": corner_mode,
                  "wall_s": round(time.time() - t0, 3),
                  "adapter_calls": adapter.log.by_name, "trace_len": len(gw.trace), "global_usage": state["usage"]}
        if save:
            r = gw.call("save", output_path=str(save))
            result["saved"] = r
            # the ONLY reported score: PCBWorld's official evaluator on the saved board
            # (fresh engine, DRC, routability baseline; whole board, not just the work set)
            lazy.close()
            from route_agent.eval_official import official_score
            result["official"] = official_score(str(save), r.get("project") or None, reward_config=reward_config)
        return result
    finally:
        lazy.close()


def cmd_execute(a) -> int:
    snap = _load_snapshot(a.snapshot)
    guide = _fill_guide_placeholders(load_json(a.guide), snap)
    if a.budget_calls:
        guide.setdefault("budget", {})["executor_calls"] = int(a.budget_calls)   # explicit override, visible in the report
    if a.replan_cap is not None:
        guide.setdefault("budget", {})["replan_cap"] = int(a.replan_cap)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    provider = None
    if a.provider and a.provider != "none":
        from route_agent.providers import make_global_provider
        provider = make_global_provider(a.provider, model=a.model, patch_path=a.patch)
    save = a.save or str(out / "routed.kicad_pcb")
    result = _run_execute(snap, guide, out, save=save, provider=provider, mode=a.mode, finish_first=not a.no_finish_first,
                          shove_fallback=not a.no_shove_fallback, obstacle_aware=not a.no_obstacle_aware,
                          corner_mode=a.corner_mode, final_cleanup=not a.no_final_cleanup, backtrack=a.backtrack, ripup_cap=a.ripup_cap,
                          on_unroutable=a.on_unroutable, max_global_rounds=a.max_global_rounds, replans_per_edge=a.replans_per_edge,
                          maze=a.maze, alt_anchors=a.alt_anchors, batch_replan=a.batch_replan, batch_rounds=a.batch_rounds, edge_history=a.edge_history,
                          angle_repair=a.angle_repair)
    dump_json(result, out / "execution_report.json")
    brief = {k: result.get(k) for k in ("ok", "plan_id", "compile", "replans", "stopped", "budget", "final", "saved")}
    if result.get("official"):
        from route_agent.eval_official import headline
        brief["official"] = {k: v for k, v in result["official"].items() if k not in ("errors_by_type", "track_angle_violations", "per_net")}
        print(headline(result["official"]))
    brief["tasks"] = [{k: t[k] for k in ("task_id", "outcome", "pad_groups", "calls")} for t in result.get("tasks", [])]
    print(json.dumps(brief, indent=2, default=str))
    return 0 if result.get("ok") else 1


def cmd_global(a) -> int:
    from route_agent.global_planner.planner import GlobalPlanner
    from route_agent.providers import make_global_provider
    snap = _load_snapshot(a.snapshot)
    prov = None if a.provider == "none" else make_global_provider(
        a.provider, model=a.model, guide_path=a.guide)
    planner = GlobalPlanner(prov, asset_root=a.asset_root or Path(a.snapshot).resolve().parent)
    bundle = planner.build_bundle(snap)
    if a.dump_bundle:
        dump_json(bundle, a.dump_bundle)
    if a.provider == "none":
        print("bundle only")
        return 0
    if a.repair:
        return _global_with_repair(a, snap, planner, prov)
    guide, usage = planner.plan(snap)
    rep = usage["validation"]
    dump_json(guide, a.out)
    dump_json({"usage": {k: v for k, v in usage.items() if k != "raw"}, "validation": rep}, str(a.out) + ".meta.json")
    print(json.dumps({"usage": {k: v for k, v in usage.items() if k != "raw"}, "validation": rep}, indent=2))
    return 0


def _global_with_repair(a, snap, planner, prov) -> int:
    """Experiment path beside the strict GlobalPlanner boundary: the provider's plan is first
    repaired deterministically (guide/repair.py: via regions onto stage overlaps, normalized
    boxes), then validated; while it still fails and rounds remain, the validation errors go back
    to the model with the previous plan and its answer (patch or full plan) is merged. Every
    round, note and verdict is written next to the guide."""
    from route_agent.guide.patch import is_patch, merge_global_response
    from route_agent.guide.repair import repair_guide
    from route_agent.guide.validate import parse_guide, validate_guide_dict
    bundle = planner.build_bundle(snap)
    guide, usage = prov.plan(bundle)
    usages = [{k: v for k, v in usage.items() if k != "raw"}]
    rounds = []
    rep = None
    for r in range(int(a.repair) + 1):
        guide = _fill_guide_placeholders(guide, snap)
        guide, notes = repair_guide(guide, snap)
        plan, rep = validate_guide_dict(guide, snap)
        rounds.append({"round": r, "repair_notes": notes, "validation": rep.to_dict()})
        if rep.ok or r == int(a.repair):
            break
        fb = {"snapshot_ref": snap.board.snapshot_id, "validation_errors": rep.errors[:30],
              "instruction": "The plan above failed these checks. Return a GlobalGuidePatch that fixes exactly these "
                             "items (or the full corrected RouteGuidePlan). Keep everything else unchanged."}
        bundle2 = planner.build_bundle(snap, feedback=fb, previous_plan=guide)
        resp, usage2 = prov.replan(bundle2)
        usages.append({k: v for k, v in usage2.items() if k != "raw"})
        if is_patch(resp):
            prev, errs = parse_guide(guide)
            merged = merge_global_response(resp, prev, snap)[0] if prev is not None else None
            if merged is not None:
                guide = to_dict(merged)
            else:
                rounds[-1]["patch_refused"] = True
        else:
            guide = resp
    dump_json(guide, a.out)
    meta = {"usages": usages, "rounds": rounds, "validation": rep.to_dict() if rep else None}
    dump_json(meta, str(a.out) + ".meta.json")
    print(json.dumps(meta, indent=2)[:3000])
    return 0 if rep is not None and rep.ok else 1


def cmd_bench(a) -> int:
    """No-model baseline over many nets: straight-corridor guide + scripted Detailed, skipping
    edges the executor cannot route (``on_unroutable=skip``). Writes bench_report.json."""
    from route_agent.guide.baseline import baseline_plan, baseline_summary
    from route_agent.guide.validate import validate_guide
    from route_agent.snapshot.builder import build_snapshot, open_board
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    board_id = a.board_id or Path(a.board).resolve().parent.name
    lazy, adapter = open_board(a.board, a.pro)
    try:
        probe = build_snapshot(adapter, board_id=board_id, board_path=a.board, project_path=lazy.project_path,
                               work_set_codes=[], repo_root=pcbworld_root())
        cands = [n for n in probe.connectivity if len(n.clusters) >= 2]
        if a.nets:
            cands = [n for n in cands if n.net_name in a.nets or str(n.net_code) in a.nets]
        if a.skip_nets:
            cands = [n for n in cands if n.net_name not in a.skip_nets]
        if a.max_clusters:
            cands = [n for n in cands if len(n.clusters) <= a.max_clusters]
        cands.sort(key=lambda n: (len(n.clusters), n.net_code))
        if a.max_nets:
            cands = cands[:a.max_nets]
        codes = [n.net_code for n in cands]
        snap = build_snapshot(adapter, board_id=board_id, board_path=a.board, project_path=lazy.project_path,
                              work_set_codes=codes, repo_root=pcbworld_root())
    finally:
        lazy.close()
    provider = None
    if getattr(a, "provider", "none") and a.provider != "none":
        from route_agent.providers import make_global_provider
        from route_agent.snapshot.views import render_views
        provider = make_global_provider(a.provider, model=getattr(a, "model", None))
        snap.views = render_views(snap, out / "views")          # the four views the Global receives
    dump_json(snap, out / "snapshot.json")
    plan = baseline_plan(snap, margin_mm=a.margin)
    if getattr(a, "replan_cap", None) is not None:
        plan.budget.replan_cap = int(a.replan_cap)
    rep = validate_guide(plan, snap)
    guide = to_dict(plan)
    dump_json(guide, out / "guide_baseline.json")
    dump_json(rep.to_dict(), out / "validation.json")
    print("baseline:", json.dumps(baseline_summary(plan)), "validation ok" if rep.ok else f"validation FAILED: {rep.errors[:5]}")
    if not rep.ok:
        return 1
    result = _run_execute(snap, guide, out, save=a.save or str(out / "routed.kicad_pcb"), provider=provider, mode=a.mode,
                          on_unroutable="skip", finish_first=not a.no_finish_first, shove_fallback=not a.no_shove_fallback,
                          obstacle_aware=not a.no_obstacle_aware, corner_mode=a.corner_mode, final_cleanup=not a.no_final_cleanup,
                          backtrack=a.backtrack, ripup_cap=a.ripup_cap, reward_config=getattr(a, "reward_config", None),
                          max_global_rounds=getattr(a, "max_global_rounds", 12), replans_per_edge=getattr(a, "replans_per_edge", 4),
                          maze=getattr(a, "maze", 1), alt_anchors=getattr(a, "alt_anchors", 2),
                          batch_replan=getattr(a, "batch_replan", True), batch_rounds=getattr(a, "batch_rounds", 5),
                          edge_history=getattr(a, "edge_history", True), angle_repair=getattr(a, "angle_repair", True))
    nets = result["final_nets"]
    by_net: dict[str, dict] = {}
    for t in result["tasks"]:
        d = by_net.setdefault(str(t["net_code"]), {"edges": set(), "done": set(), "calls": 0})
        d["edges"].add(t["task_id"]); d["calls"] += t["calls"] or 0
        if t["outcome"] == "done":
            d["done"].add(t["task_id"])
    per_net = []
    for n in cands:
        d = by_net.get(str(n.net_code), {"edges": set(), "done": set(), "calls": 0})
        fin = nets[str(n.net_code)]
        per_net.append({"net_code": n.net_code, "net_name": n.net_name, "clusters": len(n.clusters), "edges": len(d["edges"]),
                        "edges_done": len(d["done"]), "pad_groups_final": fin["pad_groups"], "calls": d["calls"]})
    off = result.get("official") or {}
    gu = result.get("global_usage") or []
    summary = {"board": board_id, "nets": len(codes),
               "official": {k: v for k, v in off.items() if k not in ("errors_by_type", "track_angle_violations", "per_net")},
               "loop": {"edges": result["compile"]["subtasks"], "edges_done": len(result["final"]["done"]),
                        "edges_failed": len(result["final"].get("failed", [])), "facade_calls": result["budget"]["spent_calls"],
                        "wall_s": result["wall_s"], "stopped": result["stopped"], "replans": result.get("replans", 0),
                        "global_calls": len(gu), "global_prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in gu),
                        "global_completion_tokens": sum(int(u.get("completion_tokens") or 0) for u in gu)}}
    dump_json({"summary": summary, "per_net": per_net, "baseline": baseline_summary(plan), "execution": result},
              out / "bench_report.json")
    from route_agent.eval_official import headline
    print(headline(off) if off else "no official score (board not saved)")
    print("loop:", json.dumps(summary["loop"], default=str))
    if result.get("cleanup"):
        print("cleanup:", json.dumps({k: result["cleanup"].get(k) for k in ("ok", "removed", "modified", "reject_reason")}))
    print(f"{'net':<28}{'clusters':>9}{'edges':>7}{'done':>6}{'groups':>8}{'calls':>7}")
    for r in per_net:
        print(f"{r['net_name'][:27]:<28}{r['clusters']:>9}{r['edges']:>7}{r['edges_done']:>6}{r['pad_groups_final']:>8}{r['calls']:>7}")
    return 0


def _board_list(path: str) -> list[tuple[str, str, str]]:
    """Board list (``configs/board_lists/*.txt``: one prepared .kicad_pcb per line, with
    ``${PCBWORLD_DATA_ROOT}``) -> [(board_id, unrouted .kicad_pcb, .kicad_pro)]."""
    root = str(data_root())
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line.replace("${PCBWORLD_DATA_ROOT}", root))
        stem = p.name[:-len(".kicad_pcb")]
        unrouted = p.with_name(stem + "_unrouted.kicad_pcb")
        # D3 boards live in a per-board directory; synthetic D2 boards are flat
        # files named board_NNNNN.kicad_pcb under the shared test directory.
        board_id = stem if stem.startswith("board_") else p.parent.name
        out.append((board_id, str(unrouted if unrouted.exists() else p), str(p.with_name(stem + ".kicad_pro"))))
    return out


def cmd_benchset(a) -> int:
    """Run `bench` over a PCBWorld board list and aggregate the official per-board scores the way
    the paper's tables do (deterministic baseline: @1 == @5): CP = fraction of boards with
    clean_pass, Pot. = mean potential gain, Rout. = mean routability, DRV = mean error-level
    violations, WL / Via means, Time = mean wall seconds of the loop."""
    import argparse as _ap
    boards = _board_list(a.list)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bid, pcb, pro in boards:
        if a.only and bid not in a.only:
            continue
        bdir = out / bid
        ns = _ap.Namespace(board=pcb, pro=pro if Path(pro).exists() else None, out=str(bdir), board_id=bid, nets=None,
                           skip_nets=a.skip_nets, max_nets=None, max_clusters=None, margin=a.margin, mode=a.mode, save=None,
                           no_finish_first=False, no_shove_fallback=False, no_obstacle_aware=False, corner_mode=0,
                           no_final_cleanup=False, backtrack=a.backtrack, ripup_cap=a.ripup_cap, reward_config=a.reward_config,
                           provider=a.provider, model=a.model, max_global_rounds=a.max_global_rounds, replan_cap=a.replan_cap,
                           replans_per_edge=a.replans_per_edge, maze=a.maze, alt_anchors=a.alt_anchors,
                           batch_replan=a.batch_replan, batch_rounds=a.batch_rounds, edge_history=a.edge_history,
                          angle_repair=a.angle_repair)
        t0 = time.time()
        try:
            rc = cmd_bench(ns)
            rep = load_json(str(bdir / "bench_report.json"))
            off = rep["summary"]["official"]
            loop = rep["summary"]["loop"]
            rows.append({"board": bid, "ok": rc == 0, "nets": rep["summary"]["nets"], **{k: off.get(k) for k in (
                "clean_pass", "success", "routability", "potential_gain", "edges_routed", "edges_total", "drv_errors_only",
                "drv_rule_errors", "drv_unconnected", "track_angle_drv", "wirelength_mm", "via_count")},
                "facade_calls": loop["facade_calls"], "loop_wall_s": loop["wall_s"], "wall_s": round(time.time() - t0, 1),
                "replans": loop.get("replans", 0), "global_calls": loop.get("global_calls", 0),
                "global_prompt_tokens": loop.get("global_prompt_tokens", 0), "global_completion_tokens": loop.get("global_completion_tokens", 0)})
        except Exception as ex:  # noqa: BLE001 - one bad board must not end the set
            rows.append({"board": bid, "ok": False, "error": f"{type(ex).__name__}: {ex}", "wall_s": round(time.time() - t0, 1)})
        dump_json({"list": a.list, "per_board": rows}, out / "progress.json")
        print(f"[{len(rows)}/{len(boards)}] {bid}: {rows[-1].get('edges_routed')}/{rows[-1].get('edges_total')} edges, "
              f"CP={rows[-1].get('clean_pass')}, rout={rows[-1].get('routability')}, pot={rows[-1].get('potential_gain')}", flush=True)
    good = [r for r in rows if r.get("ok")]
    def mean(k):
        vals = [float(r[k]) for r in good if r.get(k) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    # paper columns: DRV = error-level violations incl. "Missing connection" (drv_errors_only);
    # DRV_rule (derived) leaves the unconnected items out; Time = the routing loop's wall seconds
    agg = {"boards": len(rows), "scored": len(good),
           "CP": round(sum(1 for r in good if r.get("clean_pass")) / len(good), 3) if good else None,
           "Pot": mean("potential_gain"), "Rout": mean("routability"), "DRV": mean("drv_errors_only"),
           "DRV_rule": mean("drv_rule_errors"), "WL_mm": mean("wirelength_mm"), "Via": mean("via_count"),
           "Time_s": mean("loop_wall_s"), "Time_total_s": mean("wall_s"),
           "global_calls": sum(int(r.get("global_calls") or 0) for r in good),
           "global_prompt_tokens": sum(int(r.get("global_prompt_tokens") or 0) for r in good),
           "global_completion_tokens": sum(int(r.get("global_completion_tokens") or 0) for r in good),
           "edges": f"{sum(int(r['edges_routed']) for r in good)}/{sum(int(r['edges_total']) for r in good)}" if good else None}
    dump_json({"list": a.list, "aggregate": agg, "per_board": rows, "settings": {"backtrack": a.backtrack, "ripup_cap": a.ripup_cap,
               "margin": a.margin, "mode": a.mode, "skip_nets": a.skip_nets, "reward_config": a.reward_config,
               "provider": a.provider, "model": a.model, "max_global_rounds": a.max_global_rounds, "replan_cap": a.replan_cap,
               "replans_per_edge": a.replans_per_edge, "maze": a.maze, "alt_anchors": a.alt_anchors,
               "batch_replan": a.batch_replan, "batch_rounds": a.batch_rounds, "edge_history": a.edge_history, "angle_repair": a.angle_repair}}, out / "summary.json")
    print(json.dumps(agg, indent=2))
    return 0 if good and len(good) == len(rows) else 1


def cmd_score(a) -> int:
    """Official PCBWorld score of any saved board (eval.metrics.evaluate_one)."""
    from route_agent.eval_official import headline, official_score
    o = official_score(a.board, a.pro, check_angle=a.check_angle, reward_config=a.reward_config)
    if a.out:
        dump_json(o, a.out)
    print(headline(o))
    print(json.dumps({k: v for k, v in o.items() if k not in ("per_net", "errors_by_type", "track_angle_violations")}, indent=2))
    if a.violations:
        print(json.dumps({"errors_by_type": o["errors_by_type"], "track_angle_violations": o["track_angle_violations"]}, indent=2))
    return 0


def cmd_repair_angles(a) -> int:
    """Offline joint repair of an already routed board: open it in the engine, repair the work-set
    nets (exec/angles.py), KiCad cleanup, save next to ``--out`` and score both boards officially.
    Connectivity and scoped DRC cannot get worse by construction, so a finished run's numbers keep;
    only the corners change."""
    from route_agent.eval_official import official_score
    from route_agent.exec.angles import guarded_repair_passes
    from route_agent.snapshot.builder import open_board
    snapd = load_json(a.snapshot) if a.snapshot else {}
    nets = [int(c) for c in (a.nets or snapd.get("work_set") or [])]
    if not nets:
        raise SystemExit("repair-angles: give --nets or --snapshot with a work_set")
    names = {int(n["net_code"]): n.get("net_name") for n in snapd.get("connectivity", [])}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pro = a.pro or (os.path.splitext(a.board)[0] + ".kicad_pro" if os.path.exists(os.path.splitext(a.board)[0] + ".kicad_pro") else None)
    lazy, adapter = open_board(a.board, pro)
    try:
        adapter.e.set_target_nets(nets)
        adapter.set_corner_mode(0)
        rep = guarded_repair_passes(adapter, nets, names, mode=a.mode, final_cleanup=True)
        cleanup = adapter.cleanup(net_codes=nets, dry_run=False, finalize=True)
        saved = adapter.save(str(out / "routed.kicad_pcb"))
    finally:
        lazy.close()
    if pro and not os.path.exists(out / "routed.kicad_pro"):
        import shutil
        shutil.copy(pro, out / "routed.kicad_pro")
    before = official_score(a.board, pro, reward_config=a.reward_config)
    after = official_score(str(out / "routed.kicad_pcb"), str(out / "routed.kicad_pro") if os.path.exists(out / "routed.kicad_pro") else None,
                           reward_config=a.reward_config)
    res = {"board": a.board, "nets": nets, "angle_repair": rep, "cleanup": cleanup, "saved": saved, "official_before": before, "official_after": after}
    dump_json(res, out / "angle_repair.json")
    keys = ("edges_routed", "clean_pass", "drv_errors_only", "track_angle_drv", "wirelength_mm", "via_count")
    print("before:", {k: before.get(k) for k in keys})
    print("after: ", {k: after.get(k) for k in keys}, "| repaired", rep["repaired"], "failed", rep["failed"], "acute", rep["acute_before"], "->", rep["acute_after"])
    return 0


def cmd_status(a) -> int:
    repo = Path(a.repo or pcbworld_root())
    from route_agent.snapshot.views import find_kicad_cli
    st = {
        "pcbworld_root": str(repo),
        "data_root": str(data_root()), "data_root_exists": data_root().is_dir(),
        "engine_so": (repo / "build_rl/pcbnew/python/rl/kicad_rl_router.so").exists(),
        "engine_version": (repo / "build_rl/pcbnew/python/rl/ENGINE_VERSION").read_text().strip() if (repo / "build_rl/pcbnew/python/rl/ENGINE_VERSION").exists() else None,
        "kicad_cli": find_kicad_cli(repo),
        "python": sys.executable,
        "env_file": ENV_INFO,
        "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai_model": os.environ.get("ROUTEAGENT_OPENAI_MODEL", "gpt-4o (default)"),
        "anthropic_model": os.environ.get("ROUTEAGENT_ANTHROPIC_MODEL", "claude-opus-5-5 (default)"),
    }
    for mod in ("pcb_world", "matplotlib", "cairosvg", "jsonschema", "openai", "anthropic", "yaml"):
        try:
            __import__(mod)
            st[f"mod:{mod}"] = True
        except Exception:
            st[f"mod:{mod}"] = False
    if a.probe:
        st["openai_key_valid"] = _probe_openai()
        st["anthropic_key_valid"] = _probe_anthropic()
    print(json.dumps(st, indent=2))
    return 0


def _probe_openai() -> str:
    """One free ``models.list`` call: distinguishes a missing, invalid and working key."""
    if not os.environ.get("OPENAI_API_KEY"):
        return "missing"
    try:
        from openai import OpenAI
        OpenAI().models.list()
        return "ok"
    except Exception as e:  # noqa: BLE001 - report the class, never the key
        return f"error: {type(e).__name__}: {getattr(e, 'status_code', '')}".rstrip(": ")


def _probe_anthropic() -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "missing"
    try:
        import anthropic
        anthropic.Anthropic().models.list(limit=1)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {getattr(e, 'status_code', '')}".rstrip(": ")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="route_agent.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot"); s.add_argument("board"); s.add_argument("--pro"); s.add_argument("--nets", nargs="*", default=[])
    s.add_argument("--out", required=True); s.add_argument("--board-id"); s.add_argument("--px-per-mm", type=float, default=20.0)
    s.add_argument("--native", action="store_true"); s.add_argument("--preparation", default=""); s.set_defaults(fn=cmd_snapshot)
    s = sub.add_parser("validate-guide"); s.add_argument("snapshot"); s.add_argument("guide"); s.add_argument("--output"); s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("compile"); s.add_argument("snapshot"); s.add_argument("guide"); s.add_argument("--output"); s.set_defaults(fn=cmd_compile)
    s = sub.add_parser("execute"); s.add_argument("snapshot"); s.add_argument("guide"); s.add_argument("--out", required=True)
    s.add_argument("--save"); s.add_argument("--provider", default="none", help="replan provider: none|file|openai|anthropic")
    s.add_argument("--model"); s.add_argument("--mode", type=int, default=2)
    s.add_argument("--patch", help="GlobalGuidePatch JSON replayed by --provider file on replan")
    s.add_argument("--on-unroutable", choices=["stop", "skip"], default="stop", help="when no (more) Global help: stop the run or skip the edge")
    s.add_argument("--max-global-rounds", type=int, default=12, help="Global replan calls per run (default: the measured D3-B configuration)")
    s.add_argument("--budget-calls", type=int, help="override the guide's budget.executor_calls")
    s.add_argument("--replan-cap", type=int, help="override the guide's budget.replan_cap")
    s.add_argument("--replans-per-edge", type=int, default=4, help="Global replans spent on one edge before it is skipped")
    s.add_argument("--no-finish-first", action="store_true", help="v0 Detailed: finish only in the last stage")
    s.add_argument("--no-shove-fallback", action="store_true", help="do not retry a stuck commit in shove mode")
    s.add_argument("--no-obstacle-aware", action="store_true", help="blind centre + diamond waypoints (v0)")
    s.add_argument("--corner-mode", type=int, default=0, choices=[0, 1, 2, 3], help="PNS corner style set once per run (0 = mitered 45)")
    s.add_argument("--no-final-cleanup", action="store_true", help="skip KiCad's track cleaner on the work-set nets at the end")
    s.add_argument("--backtrack", type=int, default=2, help="stage-level backtracks per edge (0 = off)")
    s.add_argument("--ripup-cap", type=int, default=1, help="rip-ups per victim net (0 = off)")
    s.add_argument("--maze", type=int, default=1, help="same-layer A* detours per edge when a straight hop is stuck (0 = off)")
    s.add_argument("--alt-anchors", type=int, default=2, help="restarts from other copper of the source cluster near the first stage when the start pad is boxed in (0 = off)")
    s.add_argument("--batch-replan", action=argparse.BooleanOptionalAction, default=True,
                   help="defer replans: one Global call per pass carrying every failed edge's evidence (default on; --no-batch-replan = one call per failed edge)")
    s.add_argument("--batch-rounds", type=int, default=5, help="end-of-pass batch replan calls at most")
    s.add_argument("--edge-history", action=argparse.BooleanOptionalAction, default=True, help="replan feedback carries this edge's earlier rounds: plan shape executed and where it died (default on)")
    s.add_argument("--angle-repair", action=argparse.BooleanOptionalAction, default=True, help="after routing, re-route the two segments around every non-45-degree joint as one PNS move (kept only if connectivity, DRC and length hold)")
    s.set_defaults(fn=cmd_execute)
    s = sub.add_parser("global"); s.add_argument("snapshot"); s.add_argument("--provider", default="none"); s.add_argument("--model")
    s.add_argument("--asset-root"); s.add_argument("--guide", help="concrete JSON guide for --provider file replay")
    s.add_argument("--repair", type=int, default=0, help="experiment path: deterministic repair + N model repair rounds on validation errors")
    s.add_argument("--out", default="guide_model.json"); s.add_argument("--dump-bundle"); s.set_defaults(fn=cmd_global)
    s = sub.add_parser("bench", help="no-model baseline: straight corridors + scripted Detailed over many nets")
    s.add_argument("board"); s.add_argument("--pro"); s.add_argument("--out", required=True); s.add_argument("--board-id")
    s.add_argument("--nets", nargs="*", help="net names/codes to include (default: every net with >= 2 clusters)")
    s.add_argument("--skip-nets", nargs="*", default=[]); s.add_argument("--max-nets", type=int); s.add_argument("--max-clusters", type=int)
    s.add_argument("--margin", type=float, default=1.5, help="corridor half-width around the straight segment, mm")
    s.add_argument("--mode", type=int, default=2); s.add_argument("--save")
    s.add_argument("--reward-config", help="potential for Pot.; the paper's tables use drc_dense_errors_only_eval")
    s.add_argument("--provider", default="none", help="Global replan provider for failed edges: none|anthropic|openai")
    s.add_argument("--model"); s.add_argument("--max-global-rounds", type=int, default=12); s.add_argument("--replan-cap", type=int, default=12)
    s.add_argument("--replans-per-edge", type=int, default=4)
    s.add_argument("--no-finish-first", action="store_true", help="v0 Detailed: finish only in the last stage")
    s.add_argument("--no-shove-fallback", action="store_true", help="do not retry a stuck commit in shove mode")
    s.add_argument("--no-obstacle-aware", action="store_true", help="blind centre + diamond waypoints (v0)")
    s.add_argument("--corner-mode", type=int, default=0, choices=[0, 1, 2, 3], help="PNS corner style set once per run (0 = mitered 45)")
    s.add_argument("--no-final-cleanup", action="store_true", help="skip KiCad's track cleaner on the work-set nets at the end")
    s.add_argument("--backtrack", type=int, default=2, help="stage-level backtracks per edge (0 = off)")
    s.add_argument("--ripup-cap", type=int, default=1, help="rip-ups per victim net (0 = off)")
    s.add_argument("--maze", type=int, default=1, help="same-layer A* detours per edge when a straight hop is stuck (0 = off)")
    s.add_argument("--alt-anchors", type=int, default=2, help="restarts from other copper of the source cluster near the first stage when the start pad is boxed in (0 = off)")
    s.add_argument("--batch-replan", action=argparse.BooleanOptionalAction, default=True,
                   help="defer replans: one Global call per pass carrying every failed edge's evidence (default on; --no-batch-replan = one call per failed edge)")
    s.add_argument("--batch-rounds", type=int, default=5, help="end-of-pass batch replan calls at most")
    s.add_argument("--edge-history", action=argparse.BooleanOptionalAction, default=True, help="replan feedback carries this edge's earlier rounds: plan shape executed and where it died (default on)")
    s.add_argument("--angle-repair", action=argparse.BooleanOptionalAction, default=True, help="after routing, re-route the two segments around every non-45-degree joint as one PNS move (kept only if connectivity, DRC and length hold)")
    s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("benchset", help="bench over a PCBWorld board list, aggregated like the paper's tables")
    s.add_argument("--list", required=True, help="e.g. configs/board_lists/d3b_test.txt"); s.add_argument("--out", required=True)
    s.add_argument("--only", nargs="*"); s.add_argument("--skip-nets", nargs="*", default=[])
    s.add_argument("--margin", type=float, default=1.5); s.add_argument("--mode", type=int, default=2)
    s.add_argument("--backtrack", type=int, default=2); s.add_argument("--ripup-cap", type=int, default=1)
    s.add_argument("--maze", type=int, default=1, help="same-layer A* detours per edge (0 = off)")
    s.add_argument("--alt-anchors", type=int, default=2, help="restarts from nearby copper of the source cluster (0 = off)")
    s.add_argument("--batch-replan", action=argparse.BooleanOptionalAction, default=True, help="one Global call per pass with every failed edge's evidence (default on)")
    s.add_argument("--batch-rounds", type=int, default=5)
    s.add_argument("--edge-history", action=argparse.BooleanOptionalAction, default=True, help="replan feedback carries this edge's earlier rounds: plan shape executed and where it died (default on)")
    s.add_argument("--angle-repair", action=argparse.BooleanOptionalAction, default=True, help="after routing, re-route the two segments around every non-45-degree joint as one PNS move (kept only if connectivity, DRC and length hold)")
    s.add_argument("--reward-config", default="drc_dense_errors_only_eval", help="potential for Pot. (paper tables: drc_dense_errors_only_eval)")
    s.add_argument("--provider", default="none", help="Global replan provider for failed edges: none|anthropic|openai")
    s.add_argument("--model"); s.add_argument("--max-global-rounds", type=int, default=12); s.add_argument("--replan-cap", type=int, default=12)
    s.add_argument("--replans-per-edge", type=int, default=4)
    s.set_defaults(fn=cmd_benchset)
    s = sub.add_parser("score", help="official PCBWorld score of a saved .kicad_pcb (eval.metrics.evaluate_one)")
    s.add_argument("board"); s.add_argument("--pro"); s.add_argument("--check-angle", type=int, default=45)
    s.add_argument("--out"); s.add_argument("--violations", action="store_true")
    s.add_argument("--reward-config", help="potential for Pot.; the paper's tables use drc_dense_errors_only_eval"); s.set_defaults(fn=cmd_score)
    s = sub.add_parser("repair-angles", help="offline: re-route non-45-degree joints of a routed board's work-set nets, save and score")
    s.add_argument("board"); s.add_argument("--pro"); s.add_argument("--snapshot", help="snapshot.json of the run (work_set + net names)")
    s.add_argument("--nets", nargs="*", type=int); s.add_argument("--out", required=True); s.add_argument("--mode", type=int, default=2)
    s.add_argument("--reward-config", default="drc_dense_errors_only_eval"); s.set_defaults(fn=cmd_repair_angles)
    s = sub.add_parser("status"); s.add_argument("--repo")
    s.add_argument("--probe", action="store_true", help="also verify OPENAI/ANTHROPIC keys with one free list call")
    s.set_defaults(fn=cmd_status)
    return p


ENV_INFO: dict = {"path": "", "loaded": False, "keys": []}


def main(argv=None) -> int:
    global ENV_INFO
    from route_agent.env import load_env_file
    ENV_INFO = load_env_file()                 # ~/.config/routeagent/env (or $ROUTEAGENT_ENV) wins over the shell
    ensure_pcbworld_on_path()                  # PCBWORLD_ROOT may come from that file
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
