"""Contract tests on explicit test doubles. They prove the interfaces are self-consistent;
they are NOT routing or performance results (no engine, no model).

    python -m unittest discover -s route_agent/tests -p 'test_*.py'
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS

from route_agent.exec.adapter import LayerMap, NativeAdapter
from route_agent.exec.doubles import DRC_SEVERITY_WARNING, FakeEngine, violation
from route_agent.exec.executor import Orchestrator, ScriptedDetailed, candidate_points
from route_agent.exec.gateway import SPEC_BY_NAME, ToolGateway, openai_tools, tools_schema
from route_agent.guide.baseline import baseline_plan, baseline_summary
from route_agent.guide.compiler import compile_plan, compile_summary
from route_agent.guide.patch import PatchRefused, apply_patch, merge_global_response
from route_agent.guide.validate import parse_guide, schema_errors, validate_guide, validate_guide_dict
from route_agent.loop.budget import Budget, BudgetExhausted
from route_agent.loop.router import StallTracker, route_feedback
from route_agent.prompts.bundle import build_detailed_bundle, build_global_bundle
from route_agent.schema.common import from_dict, to_dict
from route_agent.schema.feedback import Attempt, FailureEvidence, FeedbackPacket, GlobalGuidePatch
from route_agent.schema.guide import RouteGuidePlan
from route_agent.schema.snapshot import Snapshot
from route_agent.snapshot.builder import build_snapshot

# --------------------------------------------------------------------------- fixtures
PADS = [
    {"ref": "A", "pad": "1", "x": 2.0, "y": 2.0, "net": 1, "name": "N1"},
    {"ref": "B", "pad": "1", "x": 18.0, "y": 18.0, "net": 1, "name": "N1"},
    {"ref": "C", "pad": "1", "x": 2.0, "y": 18.0, "net": 2, "name": "N2"},
    {"ref": "D", "pad": "1", "x": 18.0, "y": 2.0, "net": 2, "name": "N2"},
    {"ref": "E", "pad": "1", "x": 10.0, "y": 10.0, "net": 3, "name": "N3"},   # single pad net
]


def make_snapshot(engine=None, work=(1, 2)):
    eng = engine or FakeEngine(PADS, bbox=[0, 0, 20, 20])
    ad = NativeAdapter(eng)
    snap = build_snapshot(ad, board_id="fake", board_path="fake.kicad_pcb", project_path="fake.kicad_pro",
                          work_set_codes=list(work), repo_root=tempfile.gettempdir())
    return eng, ad, snap


def guide_for(snap: Snapshot, two_layer=True) -> dict:
    g = {
        "plan_id": "t", "based_on": {"snapshot_id": snap.board.snapshot_id, "rules_hash": snap.board.rules_hash},
        "revision": 1, "layer_names": ["F.Cu", "B.Cu"],
        "corridors": [
            {"id": "c1", "layer": "F.Cu", "bbox": [0.5, 0.5, 6.0, 6.0]},
            {"id": "c2", "layer": "B.Cu", "bbox": [4.0, 4.0, 16.0, 16.0]},
            {"id": "c3", "layer": "F.Cu", "bbox": [14.0, 14.0, 19.5, 19.5]},
            {"id": "d1", "layer": "F.Cu", "bbox": [0.5, 14.0, 19.5, 19.5]},
            {"id": "d2", "layer": "F.Cu", "bbox": [14.0, 0.5, 19.5, 19.5]},
        ],
        "via_regions": [
            {"id": "v1", "layers": ["F.Cu", "B.Cu"], "bbox": [4.0, 4.0, 6.0, 6.0]},
            {"id": "v2", "layers": ["F.Cu", "B.Cu"], "bbox": [14.0, 14.0, 16.0, 16.0]},
        ],
        "connection_trees": [
            {"net_code": 1, "net_name": "N1", "edges": [
                {"id": "e1", "source_cluster": "cluster:1:0", "target_cluster": "cluster:1:1",
                 "stages": ["c1", "c2", "c3"], "via_regions": ["v1", "v2"], "profile_ref": "Default"}]},
            {"net_code": 2, "net_name": "N2", "edges": [
                {"id": "e2", "source_cluster": "cluster:2:0", "target_cluster": "cluster:2:1",
                 "stages": ["d1", "d2"], "via_regions": [], "profile_ref": "Default"}]},
        ],
        "net_order": [1, 2],
        "budget": {"executor_calls": 120, "replan_cap": 2, "local_retry_cap": 2, "stall_threshold": 2},
    }
    return g


# --------------------------------------------------------------------------- tests
class TestLayerMap(unittest.TestCase):
    def test_two_layer(self):
        lm = LayerMap(2)
        self.assertEqual(lm.names, ["F.Cu", "B.Cu"])
        self.assertEqual(lm.human("B.Cu"), 2)
        self.assertEqual(lm.name_from_board(2), "B.Cu")
        self.assertEqual(lm.pad_layers(-2), ["F.Cu", "B.Cu"])

    def test_four_layer(self):
        lm = LayerMap(4)
        self.assertEqual(lm.names, ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"])
        self.assertEqual(lm.name_from_board(4), "In1.Cu")
        with self.assertRaises(KeyError):
            lm.human("In9.Cu")

    def test_board_ids_come_from_pcbworld_mapping(self):
        from pcb_world.engine.layer_mapping import LayerMapping
        lm = LayerMap(4)
        self.assertEqual([lm.board(n) for n in lm.names], LayerMapping(4).board_layer_order)   # [0, 4, 6, 2]
        self.assertEqual((lm.name_from_board(6), lm.name_from_board(2), lm.name_from_board(31)), ("In2.Cu", "B.Cu", None))
        with self.assertRaises(ValueError):
            LayerMap(1)
        with self.assertRaises(ValueError):
            LayerMap(4, mapping=LayerMapping(2))


class TestSnapshot(unittest.TestCase):
    def test_clusters_and_roundtrip(self):
        _e, _a, snap = make_snapshot()
        n1 = snap.net(1)
        self.assertEqual([c.id for c in n1.clusters], ["cluster:1:0", "cluster:1:1"])
        self.assertEqual(n1.clusters[0].anchors[0].pad_id, "A:1")      # deterministic order
        self.assertEqual(n1.pad_groups, 2)
        self.assertIsNone(snap.net(3))                                   # single-pad net is not routable
        d = to_dict(snap)
        back = from_dict(Snapshot, json.loads(json.dumps(d)))
        self.assertEqual(back.board.snapshot_id, snap.board.snapshot_id)
        self.assertEqual(back.cluster("cluster:2:1").anchors[0].pad_id, "D:1")

    def test_work_set_by_name(self):
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20])
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_names=["N2"])
        self.assertEqual(snap.work_set, [2])
        with self.assertRaises(KeyError):
            build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_names=["NOPE"])


class TestGuideValidation(unittest.TestCase):
    def setUp(self):
        self.eng, self.ad, self.snap = make_snapshot()
        self.guide = guide_for(self.snap)

    def test_valid(self):
        plan, rep = validate_guide_dict(self.guide, self.snap)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(sorted(rep.checked_edges), ["e1", "e2"])

    def test_schema_rejects_bad_bbox(self):
        g = copy.deepcopy(self.guide)
        g["corridors"][0]["bbox"] = [1, 2, 3]
        self.assertTrue(any("schema" in e for e in schema_errors(g)))

    def test_version_guard(self):
        g = copy.deepcopy(self.guide)
        g["based_on"]["snapshot_id"] = "stale"
        _p, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(any(e.startswith("V1") for e in rep.errors))

    def test_same_layer_stages_must_intersect(self):
        g = copy.deepcopy(self.guide)
        g["corridors"][3]["bbox"] = [0.5, 14.0, 10.0, 19.5]     # d1 no longer reaches d2
        _p, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(any("V4 e2" in e for e in rep.errors))

    def test_layer_change_needs_via_region(self):
        g = copy.deepcopy(self.guide)
        g["connection_trees"][0]["edges"][0]["via_regions"] = ["v1"]
        _p, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(any("V5 e1" in e and "layer change" in e for e in rep.errors))

    def test_via_region_must_cover_overlap(self):
        g = copy.deepcopy(self.guide)
        g["via_regions"][0]["bbox"] = [10.0, 10.0, 12.0, 12.0]  # misses c1∩c2
        _p, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(any("misses" in e for e in rep.errors))

    def test_anchor_coverage(self):
        g = copy.deepcopy(self.guide)
        g["corridors"][0]["bbox"] = [3.0, 3.0, 6.0, 6.0]        # A:1 at (2,2) uncovered
        _p, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(any("V6 e1" in e for e in rep.errors))

    def test_unknown_cluster_and_profile(self):
        g = copy.deepcopy(self.guide)
        g["connection_trees"][0]["edges"][0]["target_cluster"] = "cluster:1:9"
        g["connection_trees"][0]["edges"][0]["profile_ref"] = "Wide"
        _p, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(any("V3" in e for e in rep.errors))
        self.assertTrue(any("V7" in e for e in rep.errors))

    def test_cluster_net_mismatch(self):
        g = copy.deepcopy(self.guide)
        g["connection_trees"][0]["edges"][0]["target_cluster"] = "cluster:2:1"
        _p, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(any("belongs to net 2" in e for e in rep.errors))

    def test_edge_without_stages_is_an_error_not_a_crash(self):
        g = guide_for(self.snap)
        g["connection_trees"][0]["edges"][0]["stages"] = []
        g["connection_trees"][0]["edges"][0]["via_regions"] = []
        plan, rep = validate_guide_dict(g, self.snap)
        self.assertFalse(rep.ok); self.assertTrue(any("stages" in e for e in rep.errors), rep.errors)   # schema or V4, never a crash

    def test_polygon_unsupported(self):
        g = copy.deepcopy(self.guide)
        g["corridors"][0]["polygon"] = [[0, 0], [1, 0], [1, 1]]
        _p, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(any("polygon" in e for e in rep.errors))


class TestCompiler(unittest.TestCase):
    def setUp(self):
        self.eng, self.ad, self.snap = make_snapshot()
        self.plan, rep = validate_guide_dict(guide_for(self.snap), self.snap)
        self.assertTrue(rep.ok, rep.errors)

    def test_tasks_and_transitions(self):
        tasks = compile_plan(self.plan, self.snap)
        self.assertEqual([t.task_id for t in tasks], ["e1", "e2"])
        e1 = tasks[0]
        self.assertEqual(len(e1.transitions), 2)
        self.assertEqual(e1.transitions[0].bbox, [4.0, 4.0, 6.0, 6.0])
        self.assertEqual(e1.stages[0].exit, [5.0, 5.0])
        self.assertEqual(e1.stages[1].entry, [5.0, 5.0])
        self.assertEqual(e1.source_anchors[0]["pad_id"], "A:1")
        self.assertTrue(e1.execution["supported"])
        s = compile_summary(tasks)
        self.assertEqual((s["corridor_regions"], s["via_regions"]), (5, 2))

    def test_unsupported_without_via_capability(self):
        tasks = compile_plan(self.plan, self.snap, capabilities=["route_to", "start_route"])
        self.assertFalse(tasks[0].execution["supported"])
        self.assertTrue(tasks[1].execution["supported"])

    def test_unknown_cluster_is_a_clear_error(self):
        g = guide_for(self.snap)
        g["connection_trees"][0]["edges"][0]["source_cluster"] = "cluster:1:7"
        plan, errs = parse_guide(g)                      # schema-valid, semantically wrong
        self.assertEqual(errs, [])
        with self.assertRaisesRegex(ValueError, "cluster:1:7"):
            compile_plan(plan, self.snap)

    def test_reversed_bbox_is_normalized_for_execution(self):
        g = guide_for(self.snap)
        g["corridors"][1]["bbox"] = [16.0, 16.0, 4.0, 4.0]
        plan, rep = validate_guide_dict(g, self.snap)
        self.assertTrue(rep.ok, rep.errors)
        t = compile_plan(plan, self.snap)[0]
        self.assertEqual(t.stages[1].bbox, [4.0, 4.0, 16.0, 16.0])
        self.assertEqual(t.transitions[0].bbox, [4.0, 4.0, 6.0, 6.0])

    def test_order_follows_net_order(self):
        self.plan.net_order = [2, 1]
        tasks = compile_plan(self.plan, self.snap)
        self.assertEqual([t.task_id for t in tasks], ["e2", "e1"])


class TestPatch(unittest.TestCase):
    def setUp(self):
        self.eng, self.ad, self.snap = make_snapshot()
        self.plan, _ = validate_guide_dict(guide_for(self.snap), self.snap)

    def test_patch_replaces_only_affected(self):
        patch = GlobalGuidePatch(patch_id="p1", based_on={"snapshot_id": self.snap.board.snapshot_id, "plan_revision": 1},
                                 replace_corridors=[{"id": "c2", "layer": "B.Cu", "bbox": [3.0, 3.0, 17.0, 17.0]}],
                                 preserve=["e2"], budget_revision={"executor_calls": 150})
        new, affected = apply_patch(self.plan, patch, self.snap)
        self.assertEqual(affected, ["e1"])
        self.assertEqual(new.revision, 2)
        self.assertEqual(new.corridor("c2").bbox, [3.0, 3.0, 17.0, 17.0])
        self.assertEqual(new.corridor("d1").bbox, self.plan.corridor("d1").bbox)
        self.assertEqual(new.budget.executor_calls, 150)
        self.assertEqual(next(e for _t, e in new.edges() if e.id == "e2").status, "accepted")
        rep = validate_guide(new, self.snap, edges_only=affected)
        self.assertTrue(rep.ok, rep.errors)

    def test_patch_version_guards(self):
        bad = GlobalGuidePatch(patch_id="p", based_on={"snapshot_id": "other"})
        with self.assertRaises(PatchRefused):
            apply_patch(self.plan, bad, self.snap)
        bad2 = GlobalGuidePatch(patch_id="p", based_on={"snapshot_id": self.snap.board.snapshot_id, "plan_revision": 7})
        with self.assertRaises(PatchRefused):
            apply_patch(self.plan, bad2, self.snap)
        bad3 = GlobalGuidePatch(patch_id="p", based_on={"snapshot_id": self.snap.board.snapshot_id},
                                replace_edges=[{"id": "e2", "source_cluster": "cluster:2:0", "target_cluster": "cluster:2:1", "stages": ["d1"]}],
                                preserve=["e2"])
        with self.assertRaises(PatchRefused):
            apply_patch(self.plan, bad3, self.snap)


    def test_merge_global_response_patch_or_plan(self):
        eng, ad, snap = make_snapshot()
        plan, _ = validate_guide_dict(guide_for(snap), snap)
        b = Budget.from_plan(plan.budget)
        patch = {"patch_id": "p", "based_on": {"snapshot_id": snap.board.snapshot_id, "plan_revision": plan.revision},
                 "replace_corridors": [{"id": "c2", "layer": "B.Cu", "bbox": [3.0, 3.0, 17.0, 17.0]}],
                 "preserve": ["e2"], "budget_revision": {"executor_calls": 200}}
        new, rep, affected, kind = merge_global_response(patch, plan, snap, b)
        self.assertEqual((kind, affected, new.revision, b.executor_calls), ("patch", ["e1"], plan.revision + 1, 200))
        self.assertEqual(next(e for _t, e in new.edges() if e.id == "e2").status, "accepted")
        stale = dict(patch, based_on={"snapshot_id": snap.board.snapshot_id, "plan_revision": 99})
        new2, rep2, _a, kind2 = merge_global_response(stale, plan, snap, b)
        self.assertIsNone(new2); self.assertIn("patch refused", rep2.errors[0])
        full = guide_for(snap); full["revision"] = 2
        new3, rep3, aff3, kind3 = merge_global_response(full, plan, snap, b)
        self.assertEqual((kind3, aff3, new3.revision), ("plan", None, 2))
        bad = guide_for(snap); bad["corridors"][0]["bbox"] = [0.5, 0.5, 1.0, 1.0]      # anchor coverage fails
        self.assertIsNone(merge_global_response(bad, plan, snap, b)[0])
        shape = dict(patch, budget_revision=2)                                           # a model wrote an int
        b2 = Budget.from_plan(plan.budget)
        new4, rep4, aff4, kind4 = merge_global_response(shape, plan, snap, b2)
        self.assertIsNotNone(new4); self.assertEqual((kind4, aff4), ("patch", ["e1"]))     # tolerated: patch applied ...
        self.assertTrue(any("ignored" in w for w in rep4.warnings))
        self.assertEqual(b2.executor_calls, plan.budget.executor_calls)                    # ... without a budget change
        # a via region without `layers` and off its overlap (the commonest model slips) is repaired, not refused
        p3 = {"patch_id": "p3", "based_on": {"snapshot_id": snap.board.snapshot_id, "plan_revision": plan.revision},
              "replace_via_regions": [{"id": "v1", "bbox": [0.5, 0.5, 1.5, 1.5]}]}
        new6, rep6, aff6, _k = merge_global_response(p3, plan, snap, Budget.from_plan(plan.budget))
        self.assertIsNotNone(new6, rep6.errors); self.assertEqual(aff6, ["e1"])
        v1 = new6.via_region("v1"); self.assertEqual((v1.layers, v1.bbox), (["F.Cu", "B.Cu"], [4.0, 4.0, 6.0, 6.0]))
        self.assertTrue(any("layers" in w for w in rep6.warnings))
        broken = dict(patch, replace_edges="not-a-list")
        new5, rep5, _a, _k = merge_global_response(broken, plan, snap, b2)
        self.assertIsNone(new5); self.assertTrue(rep5.errors and "patch" in rep5.errors[0])

    def test_repair_round_is_laid_over_the_patch_it_corrects(self):
        # measured on mavbridge: the first batch patch defined 27 corridors but two edges named one
        # it forgot; the repair round answered with that corridor and those two edges only. Checked
        # alone it fails on the corridors it does not repeat; laid over the first answer it passes.
        from route_agent.guide.patch import merge_repair_response, overlay_patch
        snap, plan = self.snap, self.plan
        based = {"snapshot_id": snap.board.snapshot_id, "plan_revision": plan.revision}
        e1 = {"id": "e1", "source_cluster": "cluster:1:0", "target_cluster": "cluster:1:1",
              "stages": ["c1", "c2b", "c3b"], "via_regions": ["v1", "v2"], "profile_ref": "Default"}
        first = {"patch_id": "p1", "based_on": based, "replace_edges": [e1],
                 "replace_corridors": [{"id": "c2b", "layer": "B.Cu", "bbox": [4.0, 4.0, 16.0, 16.0]}]}   # c3b never defined
        new0, rep0, _a, _k = merge_global_response(first, plan, snap)
        self.assertIsNone(new0); self.assertTrue(any("c3b" in e for e in rep0.errors), rep0.errors)
        fix = {"patch_id": "p1-fix", "based_on": based, "replace_edges": [e1],
               "replace_corridors": [{"id": "c3b", "layer": "F.Cu", "bbox": [14.0, 14.0, 19.5, 19.5]}]}
        self.assertIsNone(merge_global_response(fix, plan, snap)[0])                      # alone: c2b unknown
        new, rep, affected, kind, overlaid = merge_repair_response(first, fix, plan, snap, Budget.from_plan(plan.budget))
        self.assertIsNotNone(new, rep.errors)
        self.assertEqual((kind, affected, overlaid), ("patch", ["e1"], True))
        self.assertEqual(sorted(c.id for c in new.corridors if c.id.endswith("b")), ["c2b", "c3b"])
        self.assertEqual(next(e for _t, e in new.edges() if e.id == "e1").stages, ["c1", "c2b", "c3b"])
        ov = overlay_patch(first, fix)
        self.assertEqual((ov["patch_id"], sorted(c["id"] for c in ov["replace_corridors"]), len(ov["replace_edges"])),
                         ("p1-fix", ["c2b", "c3b"], 1))
        # a complete corrected patch that DROPS a refused item is taken on its own: the first answer
        # also carried an edge to a cluster that does not exist, the fix leaves it out
        ghost = {"id": "e9", "source_cluster": "cluster:1:0", "target_cluster": "cluster:1:7", "stages": ["c1"],
                 "via_regions": [], "profile_ref": "Default", "net_code": 1}
        first2 = dict(first, replace_edges=[e1, ghost])
        fix2 = dict(fix, replace_corridors=first["replace_corridors"] + fix["replace_corridors"])
        new2, rep2, aff2, kind2, overlaid2 = merge_repair_response(first2, fix2, plan, snap)
        self.assertIsNotNone(new2, rep2.errors)
        self.assertEqual((aff2, overlaid2), (["e1"], False))
        self.assertFalse(any(e.id == "e9" for _t, e in new2.edges()))
        # a full plan as the fix is used as it is
        full = guide_for(snap); full["revision"] = 2
        new3, _r, aff3, kind3, overlaid3 = merge_repair_response(first, full, plan, snap)
        self.assertEqual((kind3, aff3, overlaid3, new3.revision), ("plan", None, False, 2))


class TestGuideRepair(unittest.TestCase):
    def test_via_region_moved_onto_overlap_and_missing_one_added(self):
        from route_agent.guide.repair import repair_guide
        eng, ad, snap = make_snapshot()
        g = guide_for(snap)
        g["via_regions"][0]["bbox"] = [0.5, 0.5, 2.0, 2.0]                   # v1 no longer covers c1∩c2 = [4,4,6,6]
        g["connection_trees"][0]["edges"][0]["via_regions"] = ["v1"]          # second layer change has no via region
        self.assertFalse(validate_guide_dict(g, snap)[1].ok)
        fixed, notes = repair_guide(g, snap)
        plan, rep = validate_guide_dict(fixed, snap)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(len(notes), 3, notes)
        v1 = next(v for v in fixed["via_regions"] if v["id"] == "v1")
        self.assertEqual(v1["bbox"], [4.0, 4.0, 6.0, 6.0])
        self.assertEqual(fixed["connection_trees"][0]["edges"][0]["via_regions"], ["v1", "e1_via2_auto"])
        same, notes2 = repair_guide(guide_for(snap), snap)
        self.assertEqual(notes2, [])                                            # a valid guide is left alone
        from route_agent.cli.pipeline import build_parser
        self.assertEqual(build_parser().parse_args(["global", "s.json", "--repair", "1"]).repair, 1)


class TestBudgetAndRouter(unittest.TestCase):
    def test_budget_caps(self):
        b = Budget(executor_calls=10, replan_cap=1, local_retry_cap=1, reserve_calls=2)
        for _ in range(8):
            b.charge(1, "x")
        with self.assertRaises(BudgetExhausted):
            b.charge(1, "x")
        b.charge_recovery(2)
        with self.assertRaises(BudgetExhausted):
            b.charge_recovery(1)
        self.assertTrue(b.can_replan()); b.use_replan("r")
        with self.assertRaises(BudgetExhausted):
            b.use_replan("r")
        b.revise(executor_calls=20)
        self.assertEqual(b.spent_calls, 10)        # never refunded

    def _pkt(self, outcome, reason="unknown", region="c2", request=""):
        return FeedbackPacket(snapshot_ref="s", guide_id="g", edge_id="e1", net_code=1, outcome=outcome,
                              last_head=[0, 0, 1], attempts=[], connectivity={},
                              failure_evidence=FailureEvidence(native_reason=reason, region_ref=region), request=request)

    def test_router_table(self):
        b = Budget(replan_cap=2, local_retry_cap=2)
        st = StallTracker(2)
        self.assertEqual(route_feedback(self._pkt("done"), b, st).kind, "done")
        self.assertEqual(route_feedback(self._pkt("progress"), b, st).kind, "continue")
        d1 = route_feedback(self._pkt("blocked"), b, st)
        self.assertEqual(d1.kind, "retry")
        d2 = route_feedback(self._pkt("blocked"), b, st)          # second consecutive failure in c2 -> stalled
        self.assertEqual((d2.kind, d2.request), ("replan", "widen"))
        d3 = route_feedback(self._pkt("blocked", reason="via_on_thru_pad", region="v1"), b, st)
        self.assertEqual((d3.kind, d3.request), ("replan", "alternate_layer"))
        d4 = route_feedback(self._pkt("conflict", request="reorder"), b, st)
        self.assertEqual((d4.kind, d4.request), ("replan", "reorder"))
        b.replans = 2
        self.assertEqual(route_feedback(self._pkt("unsupported"), b, st).kind, "stop")

    def test_progress_resets_the_counter_failures_use(self):
        b = Budget(replan_cap=2, local_retry_cap=5)
        st = StallTracker(2)
        def pkt(outcome):
            p = self._pkt(outcome, region="v2")
            p.stage_ref = "c2"                      # both refs set and different
            return p
        self.assertEqual(route_feedback(pkt("blocked"), b, st).kind, "retry")
        self.assertEqual(route_feedback(pkt("progress"), b, st).kind, "continue")
        self.assertEqual(route_feedback(pkt("blocked"), b, st).kind, "retry")     # not stalled: progress intervened
        self.assertEqual(route_feedback(pkt("blocked"), b, st).kind, "replan")


class TestGatewayContracts(unittest.TestCase):
    def setUp(self):
        self.eng, self.ad, self.snap = make_snapshot()
        self.b = Budget(executor_calls=60)
        self.gw = ToolGateway(self.ad, budget=self.b, allowed_profiles=self.snap.rules.profiles, frozen_scope=[2],
                              target_net_names=["N1", "N2"])

    def test_phase_and_net_preconditions(self):
        self.assertFalse(self.gw.call("route_to", x=1, y=1)["ok"])             # no session
        self.assertFalse(self.gw.call("start_route", x=2, y=2, layer="F.Cu")["ok"])   # no net
        self.assertTrue(self.gw.call("select_net", net_code=1)["ok"])
        self.assertTrue(self.gw.call("start_route", x=2, y=2, layer="F.Cu")["ok"])
        self.assertFalse(self.gw.call("checkpoint")["ok"])                     # not idle
        self.assertFalse(self.gw.call("select_net", net_code=2)["ok"])         # session open
        self.assertTrue(self.gw.call("cancel")["ok"])

    def test_stale_state_ref_refused(self):
        self.gw.call("select_net", net_code=1)
        rev = self.ad.state.revision
        self.assertTrue(self.gw.call("start_route", x=2, y=2, layer="F.Cu", state_ref=rev)["ok"])
        self.assertTrue(self.gw.call("route_to", x=5, y=5)["ok"])
        r = self.gw.call("start_route", x=5, y=5, layer="F.Cu", state_ref=rev)   # revision moved on
        self.assertEqual(r["reason"], "stale state_ref")

    def test_frozen_scope_and_profiles(self):
        r = self.gw.call("delete_net_routing", net_codes=[2])
        self.assertEqual(r["reason"], "frozen scope")
        self.assertFalse(self.gw.call("apply_profile", name="Thin")["ok"])
        self.assertTrue(self.gw.call("apply_profile", name="Default")["ok"])

    def test_proposed_tools_not_executed(self):
        r = self.gw.call("drag_segment", x=1, y=1, layer="F.Cu")
        self.assertFalse(r["ok"]); self.assertIn("proposed", r["reason"])
        self.assertNotIn("drag_segment", [t["name"] for t in tools_schema()["tools"]])
        self.assertTrue(all(t["type"] == "function" for t in openai_tools()))

    def test_rollback_restores_board_and_wrapper(self):
        ck = self.gw.call("checkpoint"); h = ck["handle"]
        self.gw.call("select_net", net_code=1)
        self.gw.call("start_route", x=2, y=2, layer="F.Cu")
        self.gw.call("route_via", x=5, y=5)
        self.assertEqual(len(self.eng.vias), 1)
        self.gw.call("start_route", x=5, y=5, layer="B.Cu")      # session re-opened at via, then rollback mid-session
        r = self.gw.call("rollback", handle=h)
        self.assertTrue(r["ok"])
        self.assertEqual((len(self.eng.tracks), len(self.eng.vias)), (0, 0))
        self.assertFalse(self.eng.is_routing())
        self.assertIsNone(self.ad.current_net)

    def test_budget_exhaustion_is_structured(self):
        b = Budget(executor_calls=3, reserve_calls=1)
        gw = ToolGateway(self.ad, budget=b)
        gw.call("routing_state"); gw.call("routing_state")
        r = gw.call("routing_state")
        self.assertFalse(r["ok"]); self.assertIn("budget", r["reason"])
        self.assertTrue(gw.call("cancel")["ok"])                  # recovery still allowed from reserve

    def test_scoped_drc(self):
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], drc=[violation("clearance", ["N1"]), violation("clearance", ["N9"]),
                                                         violation("silk_overlap", ["N1"], severity=DRC_SEVERITY_WARNING),
                                                         violation("Missing connection between items", ["N1"], error_code=1)])
        gw = ToolGateway(NativeAdapter(eng), target_net_names=["N1"])
        r = gw.call("run_drc")
        # warning reported but not an error; the unconnected item is the ratsnest, counted apart
        self.assertEqual((r["count"], r["errors"], r["unconnected"]), (3, 1, 1))

    def test_set_via_dims_cannot_shrink_below_allowed_profiles(self):
        prof = self.snap.rules.profiles["Default"]
        d, h = prof["via_diameter_mm"], prof["via_drill_mm"]
        self.assertFalse(self.gw.call("set_via_dims", diameter_mm=d / 2, drill_mm=h / 2)["ok"])
        self.assertFalse(self.gw.call("set_via_dims", diameter_mm=d, drill_mm=d - 0.05)["ok"])    # annular ring shrinks
        self.assertFalse(self.gw.call("set_via_dims", diameter_mm=h, drill_mm=d)["ok"])           # drill >= diameter
        r = self.gw.call("set_via_dims", diameter_mm=d + 0.2, drill_mm=h)
        self.assertTrue(r["ok"]); self.assertEqual(r["applied"]["via_diameter_mm"], d + 0.2)

    def test_call_local_charges_the_executor_pool_not_the_reserve(self):
        b = Budget(executor_calls=10, reserve_calls=4)                       # 6 executor calls, 4 reserve
        gw = ToolGateway(self.ad, budget=b)
        for _ in range(6):
            self.assertTrue(gw.call_local("checkpoint")["ok"])
        self.assertFalse(gw.call_local("checkpoint")["ok"])                  # pool spent
        self.assertTrue(gw.call("checkpoint")["ok"])                         # the reserve is still there
        self.assertEqual(b.spent_calls, 7)

    def test_corner_mode_tool(self):
        self.assertEqual(self.gw.call("set_corner_mode", mode=0)["applied"], {"corner_mode": 0})
        self.assertFalse(self.gw.call("set_corner_mode", mode=7)["ok"])
        from route_agent.cli.pipeline import build_parser
        a = build_parser().parse_args(["bench", "b.kicad_pcb", "--out", "o", "--corner-mode", "2", "--no-final-cleanup"])
        self.assertEqual((a.corner_mode, a.no_final_cleanup), (2, True))

    def test_inspect_region_accepts_reversed_corners(self):
        self.assertEqual(len(self.ad.inspect_region([19.0, 19.0, 1.0, 1.0])["pads"]), len(PADS))
        self.assertEqual(len(self.ad.inspect_region([1.0, 1.0, 19.0, 19.0], layer="F.Cu")["pads"]), len(PADS))

    def test_every_spec_maps_to_adapter(self):
        for name, spec in SPEC_BY_NAME.items():
            if spec.status == "implemented":
                self.assertTrue(hasattr(NativeAdapter, name), name)


class TestScriptedExecution(unittest.TestCase):
    # a small F.Cu obstacle on the A->B / C->D diagonal: a direct PNS finish from a pad is blocked,
    # but off-centre candidate points and the B.Cu trunk of guide_for stay reachable
    DIAG_BLOCK = [[9.6, 9.6, 10.4, 10.4]]

    def _run(self, eng, guide_mut=None, replan=None, on_unroutable="stop", finish_first=True, obstacle_aware=True, maze=2, ripup_cap=1,
             alt_anchors=2):
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
        g = guide_for(snap)
        if guide_mut:
            guide_mut(g)
        plan, rep = validate_guide_dict(g, snap)
        self.assertTrue(rep.ok, rep.errors)
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id, finish_first=finish_first, obstacle_aware=obstacle_aware, maze=maze,
                               alt_anchors=alt_anchors)
        orch = Orchestrator(gw, det, b, stall_threshold=plan.budget.stall_threshold, replan_fn=replan,
                            recompile_fn=lambda p: compile_plan(p, snap), target_net_names=["N1", "N2"],
                            on_unroutable=on_unroutable, ripup_cap=ripup_cap)
        return orch.run(compile_plan(plan, snap)), eng, gw, snap, plan

    def test_skip_policy_goes_on_after_an_unroutable_edge(self):
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: self.DIAG_BLOCK, 2: [[9.0, 0.0, 11.0, 20.0]]})   # B.Cu trunk of e1 blocked
        rep, eng, gw, _s, _p = self._run(eng, on_unroutable="skip")
        self.assertEqual([(t["task_id"], t["outcome"]) for t in rep.tasks], [("e1", "blocked"), ("e2", "done")])
        self.assertEqual((rep.failed, rep.final["failed"], rep.stopped), (["e1"], ["e1"], ""))
        self.assertEqual(eng.get_pad_groups()[2], 1)
        self.assertEqual(eng._ckpts, {})

    def test_happy_path_connects_both_nets(self):
        rep, eng, gw, _s, _p = self._run(FakeEngine(PADS, bbox=[0, 0, 20, 20]))
        self.assertEqual([t["outcome"] for t in rep.tasks], ["done", "done"])
        self.assertEqual(eng.get_pad_groups()[1], 1)
        self.assertEqual(len(eng.vias), 0)                                   # finish-first: PNS closes directly, no via needed
        self.assertEqual(rep.final["drc_errors"], 0)
        self.assertEqual(rep.stopped, "")
        self.assertGreater(rep.budget["spent_calls"], 0)
        self.assertEqual(eng._ckpts, {})                                     # every checkpoint released
        rep2, eng2, _g, _s, _p = self._run(FakeEngine(PADS, bbox=[0, 0, 20, 20]), finish_first=False)
        self.assertEqual([t["outcome"] for t in rep2.tasks], ["done", "done"])
        self.assertEqual(len(eng2.vias), 2)                                  # v0 walk follows the guide's B.Cu trunk

    def test_shove_fallback_pushes_through_a_soft_block(self):
        # the diagonal is blocked for walkaround only: finish (mode 2) fails, the shove retry (mode 1) connects
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], shovable={1: self.DIAG_BLOCK})
        rep, eng, gw, _s, _p = self._run(eng)
        self.assertEqual([t["outcome"] for t in rep.tasks], ["done", "done"])
        self.assertEqual(len(eng.vias), 0)
        fin = [(a.ok, a.mode) for a in rep.tasks[0]["feedback"]["attempts"] if a.kind == "finish"]
        self.assertEqual(fin, [(False, 2), (True, 1)])
        self.assertEqual([(s["mode"], s["ok"]) for s in rep.tasks[0]["steps"]], [(2, False), (1, True)])
        # without the fallback the walk takes the guide's B.Cu trunk instead (2 vias)
        eng2 = FakeEngine(PADS, bbox=[0, 0, 20, 20], shovable={1: self.DIAG_BLOCK})
        ad = NativeAdapter(eng2)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
        plan, _ = validate_guide_dict(guide_for(snap), snap)
        b = Budget.from_plan(plan.budget)
        gw2 = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
        det = ScriptedDetailed(gw2, snap.board.snapshot_id, plan.plan_id, shove_fallback=False)
        rep2 = Orchestrator(gw2, det, b, target_net_names=["N1", "N2"]).run(compile_plan(plan, snap))
        self.assertEqual([t["outcome"] for t in rep2.tasks], ["done", "done"])
        self.assertEqual(len(eng2.vias), 1)                                  # via at v1, then finish-first on B.Cu closes
        self.assertTrue(all(a.mode == 2 for a in rep2.tasks[0]["feedback"]["attempts"]))

    def test_rejected_commits_are_not_retried_in_shove(self):
        # via target inside a thru-hole pad: make_via's pre-check rejects it; no shove retry, next candidate instead
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: self.DIAG_BLOCK}, via_blocked=[[4.9, 4.9, 5.1, 5.1]])
        rep, eng, gw, _s, _p = self._run(eng)
        self.assertEqual(rep.tasks[0]["outcome"], "done")
        vias = [a for a in rep.tasks[0]["feedback"]["attempts"] if a.kind == "route_via"]
        self.assertEqual((vias[0].ok, vias[0].native_reason, vias[0].mode), (False, "via_on_thru_pad", 2))
        self.assertTrue(vias[1].ok and vias[1].mode == 2)                   # straight to the next candidate, still walkaround

    def test_finish_to_another_cluster_is_progress_not_rollback(self):
        # net 1: A(2,2) source, B(18,2) target, C(6,2) a third cluster nearer to A than B is.
        # PNS finish from A reaches C first: legal copper -> progress, kept; the re-run then reaches B.
        pads = [{"ref": "A", "pad": "1", "x": 2.0, "y": 2.0, "net": 1, "name": "N1"},
                {"ref": "B", "pad": "1", "x": 18.0, "y": 2.0, "net": 1, "name": "N1"},
                {"ref": "C", "pad": "1", "x": 6.0, "y": 2.0, "net": 1, "name": "N1"}]
        eng = FakeEngine(pads, bbox=[0, 0, 20, 20])
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1])
        g = {"plan_id": "t", "based_on": {"snapshot_id": snap.board.snapshot_id, "rules_hash": snap.board.rules_hash},
             "revision": 1, "layer_names": ["F.Cu", "B.Cu"],
             "corridors": [{"id": "c1", "layer": "F.Cu", "bbox": [0.5, 0.5, 19.5, 3.5]}], "via_regions": [],
             "connection_trees": [{"net_code": 1, "net_name": "N1", "edges": [
                 {"id": "ab", "source_cluster": "cluster:1:0", "target_cluster": "cluster:1:1", "stages": ["c1"], "via_regions": [], "profile_ref": "Default"}]}],
             "net_order": [1], "budget": {"executor_calls": 120, "replan_cap": 2, "local_retry_cap": 2, "stall_threshold": 2}}
        plan, rep0 = validate_guide_dict(g, snap)
        self.assertTrue(rep0.ok, rep0.errors)
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1"])
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id)
        rep = Orchestrator(gw, det, b, target_net_names=["N1"], on_unroutable="skip").run(compile_plan(plan, snap))
        self.assertEqual([(t["task_id"], t["outcome"], t["decision"]) for t in rep.tasks], [("ab", "progress", "continue"), ("ab", "done", "done")])
        self.assertEqual(rep.tasks[0]["feedback"]["connectivity"]["merged_points"], [[6.0, 2.0]])
        self.assertEqual(rep.tasks[0]["feedback"]["failure_evidence"].native_reason, "finish_joined_other_cluster")
        self.assertFalse(any(e.kind == "rollback" for e in rep.events))
        self.assertEqual((len(eng.tracks), eng.get_pad_groups()[1]), (2, 1))   # A-C kept, then A-B
        self.assertEqual(rep.final["done"], ["ab"])

    def test_via_head_uses_the_real_via_position(self):
        # the fake lands every via 0.5 mm right of the request (> snap 0.3): the next start_route must
        # begin on the real via copper, otherwise the walk cannot continue
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: self.DIAG_BLOCK}, via_offset=(0.5, 0.0))
        rep, eng, gw, _s, _p = self._run(eng, finish_first=False)
        self.assertEqual([t["outcome"] for t in rep.tasks], ["done", "done"])
        self.assertEqual(len(eng.vias), 2)
        self.assertFalse(any(a.kind == "start_route" and not a.ok for a in rep.tasks[0]["feedback"]["attempts"]))
        r = gw.trace
        self.assertTrue(any(t["tool"] == "start_route" and abs(t["args"]["x"] - 5.5) < 1e-9 for t in r))   # resumed at the landed via

    def test_feedback_checks_are_real(self):
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20])
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
        plan, _ = validate_guide_dict(guide_for(snap), snap)
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, frozen_scope=[2], target_net_names=["N1", "N2"])
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id)
        tasks = compile_plan(plan, snap)
        res = det.run_task(tasks[0], b)
        self.assertEqual(res.outcome, "done")
        self.assertEqual((res.feedback.checks["fresh"], res.feedback.checks["scope_ok"]), (True, True))
        self.assertEqual(res.feedback.checks["revision"], ad.state.revision)
        # a frozen net that gained copper during the task is reported, not assumed fine
        det._scope_before = det._frozen_signature()
        eng.tracks.append(NS(x1_mm=2.0, y1_mm=18.0, x2_mm=18.0, y2_mm=2.0, width_mm=0.2, layer=0, net_code=2, net_name="N2", uuid=""))
        pkt = det._packet(tasks[0], "done", [0, 0, 1], [], "", "c3", "")
        self.assertFalse(pkt.checks["scope_ok"])

    def test_obstacle_aware_candidates_unit(self):
        from route_agent.exec.waypoints import clearance_ok, foreign_obstacles, grid_points, obstacle_aware_candidates
        region = {"pads": [{"net_code": 3, "x_mm": 5.0, "y_mm": 5.0, "w_mm": 0.8, "h_mm": 0.8, "layers": ["F.Cu", "B.Cu"]},
                           {"net_code": 1, "x_mm": 3.5, "y_mm": 3.5, "w_mm": 1.0, "h_mm": 1.0, "layers": ["F.Cu"]}],
                  "tracks": [{"net_code": 4, "layer": "B.Cu", "seg": [3.0, 6.0, 7.0, 6.0], "width_mm": 0.2}],
                  "vias": []}
        obst = foreign_obstacles(region, net_code=1)
        self.assertEqual((len(obst["pads"]), len(obst["tracks"])), (1, 1))       # own-net pad dropped
        self.assertFalse(clearance_ok(5.0, 5.0, 0.5, obst, ["F.Cu"]))               # on the foreign pad
        self.assertTrue(clearance_ok(5.0, 6.3, 0.2, obst, ["F.Cu"]))                # B.Cu track irrelevant on F.Cu
        self.assertFalse(clearance_ok(5.0, 6.3, 0.2, obst, ["F.Cu", "B.Cu"]))       # a via must clear both faces
        pts = grid_points([3.0, 3.0, 7.0, 7.0], 0.5)
        self.assertEqual(pts[0], [5.0, 5.0]); self.assertEqual(len(pts), 25)
        cands = obstacle_aware_candidates([3.0, 3.0, 7.0, 7.0], 0.5, obst, ["F.Cu", "B.Cu"], head=[2.0, 2.0], hint=[18.0, 18.0])
        self.assertTrue(cands and all(clearance_ok(x, y, 0.5, obst, ["F.Cu", "B.Cu"]) for x, y in cands))
        self.assertNotIn([5.0, 5.0], cands)
        self.assertNotIn([6.5, 6.5], cands)                                         # 0.5 mm from the B.Cu track: too close for a via
        self.assertEqual(cands[0], [3.5, 3.5])                                      # on the A->B line, the nearest clear one
        on_f = obstacle_aware_candidates([3.0, 3.0, 7.0, 7.0], 0.5, obst, ["F.Cu"], head=[2.0, 2.0], hint=[18.0, 18.0])
        self.assertEqual(on_f[0], [6.5, 6.5])                                       # a track-only F.Cu waypoint may sit there
        self.assertEqual(obstacle_aware_candidates([4.6, 4.6, 5.4, 5.4], 0.5, obst, ["F.Cu"], [2, 2], None), [])

    def test_obstacle_aware_candidates_route_around_a_blocked_portal(self):
        # a foreign pad (net 3) sits on the via-region centre, hard-blocked for PNS; blind candidates all
        # fall inside the block, the obstacle-aware grid steps aside
        pads = PADS + [{"ref": "F", "pad": "1", "x": 5.0, "y": 5.0, "net": 3, "name": "N3", "w": 1.6, "h": 1.6}]
        def wide(g):
            g["corridors"][0]["bbox"] = [0.5, 0.5, 8.0, 8.0]              # c1
            g["corridors"][1]["bbox"] = [3.0, 3.0, 16.0, 16.0]            # c2
            g["via_regions"][0]["bbox"] = [3.0, 3.0, 7.0, 7.0]            # v1
        for aware, want in ((True, "done"), (False, "blocked")):
            eng = FakeEngine(pads, bbox=[0, 0, 20, 20], blocked={1: self.DIAG_BLOCK + [[4.0, 4.0, 6.0, 6.0]]})
            ad = NativeAdapter(eng)
            snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
            g = guide_for(snap); wide(g)
            plan, rep0 = validate_guide_dict(g, snap)
            self.assertTrue(rep0.ok, rep0.errors)
            b = Budget.from_plan(plan.budget)
            gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
            det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id, obstacle_aware=aware)
            rep = Orchestrator(gw, det, b, target_net_names=["N1", "N2"], on_unroutable="skip").run(compile_plan(plan, snap))
            self.assertEqual(rep.tasks[0]["outcome"], want, (aware, [(s["kind"], s["target"], s["ok"]) for s in rep.tasks[0]["steps"]]))
            if aware:
                via = next(s for s in rep.tasks[0]["steps"] if s["kind"] == "route_via" and s["ok"])
                self.assertNotEqual(via["target"], [5.0, 5.0])
                self.assertTrue(any(t["tool"] == "inspect_region" for t in gw.trace))

    def test_backtracking_recovers_from_a_dead_end(self):
        # three F.Cu stages; a wall x in [9,10], y <= 9.5. From the first two portal candidates every
        # second-boundary segment crosses the wall; from the third one (5.15, 6) a candidate clears it.
        def three(g):
            g["corridors"][0]["bbox"] = [0.5, 0.5, 8.0, 8.0]
            g["corridors"][1] = {"id": "c2", "layer": "F.Cu", "bbox": [4.0, 4.0, 16.0, 16.0]}
            g["corridors"][2]["bbox"] = [12.0, 12.0, 19.5, 19.5]
            g["connection_trees"][0]["edges"][0]["via_regions"] = []
            g["budget"]["local_retry_cap"] = 20
        for bt, want in ((3, "done"), (0, "blocked")):
            eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: [[9.0, 0.0, 10.0, 9.5]]})
            ad = NativeAdapter(eng)
            snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
            g = guide_for(snap); three(g)
            plan, rep0 = validate_guide_dict(g, snap)
            self.assertTrue(rep0.ok, rep0.errors)
            b = Budget.from_plan(plan.budget)
            gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
            det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id, finish_first=False, obstacle_aware=False, backtrack=bt, maze=0)
            rep = Orchestrator(gw, det, b, target_net_names=["N1", "N2"], on_unroutable="skip", ripup_cap=0).run(compile_plan(plan, snap))
            e1 = rep.tasks[0]
            self.assertEqual(e1["outcome"], want, [(s["kind"], s["target"], s["ok"]) for s in e1["steps"]])
            if bt:
                self.assertGreaterEqual(e1["feedback"]["checks"]["backtracks"], 1)
                self.assertEqual(eng._ckpts, {})                               # sub-checkpoints released
                self.assertTrue(any(t["tool"] == "rollback" for t in gw.trace))
            else:
                self.assertEqual(e1["feedback"]["failure_evidence"].region_ref, "c2->c3")
                self.assertEqual(e1["feedback"]["checks"]["backtracks"], 0)

    def test_ripup_frees_a_blocked_edge_and_requeues_the_victim(self):
        # tracks of other nets are hard obstacles: N1 routed first across the board blocks N2's only
        # F.Cu path; rip-up removes N1, N2 routes, N1 is re-queued and re-routes through its via trunk
        for cap, want_done in ((1, {"e1", "e2"}), (0, {"e1"})):
            eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], tracks_block=True)
            ad = NativeAdapter(eng)
            snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
            plan, _ = validate_guide_dict(guide_for(snap), snap)
            b = Budget.from_plan(plan.budget)
            gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
            det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id, maze=0)          # the maze would detour instead
            rep = Orchestrator(gw, det, b, target_net_names=["N1", "N2"], on_unroutable="skip", ripup_cap=cap,
                               rippable_nets={1, 2}).run(compile_plan(plan, snap))
            self.assertEqual(set(rep.final["done"]), want_done, [(t["task_id"], t["outcome"], t.get("ripup")) for t in rep.tasks])
            kinds = [e.kind for e in rep.events]
            if cap:
                self.assertIn("ripup", kinds); self.assertIn("ripup_requeue", kinds)
                rip = next(t for t in rep.tasks if t.get("ripup"))
                self.assertEqual((rip["task_id"], rip["ripup"]["victim"], rip["ripup"]["helped"]), ("e2", 1, True))
                self.assertEqual((eng.get_pad_groups()[1], eng.get_pad_groups()[2]), (1, 1))
                self.assertEqual(len(eng.vias), 1)                                # N1 re-routed via B.Cu
            else:
                self.assertNotIn("ripup", kinds)
                self.assertEqual(rep.failed, ["e2"])

    def test_frozen_nets_are_never_ripped(self):
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], tracks_block=True)
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
        plan, _ = validate_guide_dict(guide_for(snap), snap)
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, frozen_scope=[1], target_net_names=["N1", "N2"])
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id, maze=0)              # the maze would detour instead
        rep = Orchestrator(gw, det, b, target_net_names=["N1", "N2"], on_unroutable="skip", ripup_cap=2).run(compile_plan(plan, snap))
        self.assertNotIn("ripup", [e.kind for e in rep.events])
        self.assertEqual(rep.failed, ["e2"])

    def test_finish_first_beats_a_blocked_portal(self):
        def same_layer(g):
            g["corridors"][0]["bbox"] = [0.5, 0.5, 6.0, 10.0]               # c1 (F.Cu)
            g["corridors"][1] = {"id": "c2", "layer": "F.Cu", "bbox": [4.0, 6.0, 16.0, 16.0]}
            g["connection_trees"][0]["edges"][0]["via_regions"] = []
        # the whole c1∩c2 portal [4,6]x[6,10] is blocked, the A->B diagonal is free
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: [[4.2, 7.0, 5.8, 9.5]]})
        rep, eng, gw, _s, _p = self._run(eng, guide_mut=same_layer)
        self.assertEqual([t["outcome"] for t in rep.tasks], ["done", "done"])
        self.assertEqual([s["kind"] for s in rep.tasks[0]["steps"]], ["finish"])
        eng0 = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: [[4.2, 7.0, 5.8, 9.5]]})
        rep0, eng0, _g, _s, _p = self._run(eng0, guide_mut=same_layer, finish_first=False, obstacle_aware=False, on_unroutable="skip")
        self.assertEqual(rep0.tasks[0]["outcome"], "blocked")
        self.assertEqual(rep0.tasks[0]["feedback"]["failure_evidence"].region_ref, "c1->c2")

    WALLS = [[3.5, 0.0, 4.0, 4.5], [0.0, 3.5, 4.5, 4.0]]       # F.Cu walls boxing pad A (2, 2) in
    SMD = [dict(p, layer=0) for p in PADS]                       # the same pads on F.Cu only

    def test_boxed_in_pad_reports_evidence_and_a_global_via_region_routes_it(self):
        # nothing from A gets past the walls on F.Cu: finish, portal and via-region candidates all die
        # stuck at c1 (the signature of 80-92% of the D3-B / D3-C failures). The Detailed places no
        # via of its own: it reports where one would fit, what fences the head in, and asks for a
        # layer change. The Global answers with a via region on one of those points and the normal
        # layer-change walk routes the edge. The via is the Global's decision.
        eng = FakeEngine(self.SMD, bbox=[0, 0, 20, 20], blocked={1: self.WALLS})
        rep0, eng0, _g, _s, _p = self._run(eng, on_unroutable="skip")
        e1 = [t for t in rep0.tasks if t["task_id"] == "e1"][-1]
        fb = e1["feedback"]
        self.assertEqual((e1["outcome"], fb["request"]), ("blocked", "alternate_layer"))
        ev = fb["failure_evidence"]
        self.assertTrue(ev.via_sites and all(math.hypot(x - 2.0, y - 2.0) <= 2.6 for x, y in ev.via_sites), ev.via_sites)
        self.assertFalse(ev.head_is_thru)
        self.assertEqual((len(eng0.vias), ev.alt_starts), (0, []))
        # a thru-hole head says so (the Global may then plan the other layer with no via at all)
        rep1, eng1, _g, _s, _p = self._run(FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: self.WALLS}), on_unroutable="skip")
        self.assertTrue([t for t in rep1.tasks if t["task_id"] == "e1"][-1]["feedback"]["failure_evidence"].head_is_thru)
        seen = []

        def replan(packet, decision):
            seen.append((packet.edge_id, decision.request))
            self.assertEqual(packet.history, [])                                       # first round of this edge
            sx, sy = packet.failure_evidence.via_sites[0]
            box = [round(sx - 0.5, 3), round(sy - 0.5, 3), round(sx + 0.5, 3), round(sy + 0.5, 3)]
            patch = GlobalGuidePatch(patch_id="p-via", based_on={"snapshot_id": packet.snapshot_ref, "plan_revision": self._plan.revision},
                                     replace_corridors=[{"id": "c1", "layer": "F.Cu", "bbox": [0.5, 0.5, 3.4, 3.4]},
                                                        {"id": "c2", "layer": "B.Cu", "bbox": [0.5, 0.5, 16.0, 16.0]}],
                                     replace_via_regions=[{"id": "v1", "layers": ["F.Cu", "B.Cu"], "bbox": box}],
                                     replace_edges=[{"id": "e1", "source_cluster": "cluster:1:0", "target_cluster": "cluster:1:1",
                                                     "stages": ["c1", "c2", "c3"], "via_regions": ["v1", "v2"], "profile_ref": "Default"}])
            new, _aff = apply_patch(self._plan, patch, self._snap)
            self._plan = new
            return new

        eng = FakeEngine(self.SMD, bbox=[0, 0, 20, 20], blocked={1: self.WALLS})
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
        self._snap = snap
        plan, rep2 = validate_guide_dict(guide_for(snap), snap)
        self.assertTrue(rep2.ok, rep2.errors)
        self._plan = plan
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id)
        orch = Orchestrator(gw, det, b, replan_fn=replan, recompile_fn=lambda p: compile_plan(p, snap), target_net_names=["N1", "N2"],
                            on_unroutable="skip")
        rep = orch.run(compile_plan(plan, snap))
        self.assertEqual(seen, [("e1", "alternate_layer")])
        self.assertEqual([t["outcome"] for t in rep.tasks if t["task_id"] == "e1"], ["blocked", "done"])
        self.assertEqual(eng.get_pad_groups()[1], 1)
        via = next(s for t in rep.tasks for s in t["steps"] if s["kind"] == "route_via" and s["ok"])     # the PLANNED via
        self.assertLess(math.hypot(via["target"][0] - 2.0, via["target"][1] - 2.0), 1.6)

    def test_boxed_in_start_restarts_from_other_copper_of_its_cluster(self):
        # net 1's source cluster has two SMD pads joined by an existing track: A (2, 2) boxed in by the
        # walls and A:2 (0.6, 7.0) in the open but farther from the portal, so A is tried first. The
        # walk restarts from A:2 (copper already joined to the source) and PNS finish closes the
        # edge: no via, no new copper for the switch itself. Off: the edge stays unrouted.
        pads = self.SMD + [{"ref": "A", "pad": "2", "x": 0.6, "y": 7.0, "net": 1, "name": "N1", "layer": 0}]
        for alt, want in ((2, ("done",)), (0, ("blocked", "partial"))):
            eng = FakeEngine(pads, bbox=[0, 0, 20, 20], blocked={1: self.WALLS})
            eng.tracks.append(NS(x1_mm=2.0, y1_mm=2.0, x2_mm=0.6, y2_mm=7.0, width_mm=0.2, layer=0, net_code=1, net_name="N1", uuid=""))
            rep, eng, gw, snap, _p = self._run(eng, on_unroutable="skip", alt_anchors=alt, ripup_cap=0)
            self.assertEqual(len(snap.connectivity[0].clusters), 2)                         # A + A:2 are one cluster
            e1 = [t for t in rep.tasks if t["task_id"] == "e1"][-1]
            self.assertIn(e1["outcome"], want, (alt, [(s["kind"], s["target"], s["ok"], s["reason"]) for s in e1["steps"]]))
            self.assertEqual(len(eng.vias), 0)
            self.assertEqual(e1["feedback"]["checks"]["anchor_switches"], 1 if alt else 0)
            if alt:
                sw = next(a for a in e1["feedback"]["attempts"] if a.kind == "switch_anchor")
                self.assertEqual(sw.target, [0.6, 7.0])
                self.assertEqual(eng.get_pad_groups()[1], 1)
            else:
                self.assertEqual(e1["feedback"]["failure_evidence"].alt_starts, [[0.6, 7.0]])    # reported to the Global instead

    def test_maze_detour_gets_around_a_same_layer_track(self):
        # one F.Cu plan (no via region); a foreign track is a wall at x = 10 from y = 0.5 to 14 and
        # other nets' tracks are hard obstacles. Every straight attempt from the portal to B crosses
        # it; the maze finds the gap above y = 14 and feeds PNS the corners. No via, no layer change.
        def one_layer(g):
            g["corridors"][0]["bbox"] = [0.5, 0.5, 6.0, 10.0]                      # c1 F.Cu
            g["corridors"][1] = {"id": "c2", "layer": "F.Cu", "bbox": [4.0, 6.0, 19.5, 19.5]}
            g["connection_trees"][0]["edges"][0]["stages"] = ["c1", "c2"]
            g["connection_trees"][0]["edges"][0]["via_regions"] = []
        for mz, want in ((2, ("done",)), (0, ("blocked", "partial"))):
            eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], tracks_block=True)
            eng.tracks.append(NS(x1_mm=10.0, y1_mm=0.5, x2_mm=10.0, y2_mm=14.0, width_mm=0.2, layer=0, net_code=2, net_name="N2", uuid=""))
            rep, eng, gw, _s, _p = self._run(eng, guide_mut=one_layer, on_unroutable="skip", maze=mz, ripup_cap=0)   # no rip-up: the wall stays
            e1 = [t for t in rep.tasks if t["task_id"] == "e1"][-1]
            self.assertIn(e1["outcome"], want, (mz, [(s["kind"], s["target"], s["ok"], s["reason"]) for s in e1["steps"]]))
            self.assertEqual(len(eng.vias), 0)
            if mz:
                hops = [a for a in e1["feedback"]["attempts"] if a.region_ref.endswith(":maze") and a.ok]
                self.assertGreaterEqual(len(hops), 2)                                    # a real detour, not one straight hop
                self.assertTrue(all(a.target[0] < 10.0 or a.target[1] > 14.0 or a.target == [18.0, 18.0] for a in hops), hops)
                self.assertEqual(e1["feedback"]["checks"]["mazes"], 1)
                self.assertEqual((eng.get_pad_groups()[1], eng._ckpts), (1, {}))
            else:
                self.assertEqual(e1["feedback"]["checks"]["mazes"], 0)

    def test_maze_waypoints_unit(self):
        from route_agent.exec.maze import maze_waypoints
        wall = {"pads": [], "tracks": [{"net_code": 9, "layer": "F.Cu", "seg": [10.0, 0.0, 10.0, 14.0], "width_mm": 0.2}], "vias": []}
        pts = maze_waypoints([0, 0, 20, 20], 0.5, wall, "F.Cu", 0.3, [5.0, 8.0], [18.0, 18.0])
        self.assertTrue(pts and pts[-1] == [18.0, 18.0] and len(pts) >= 2, pts)
        self.assertTrue(all(p[1] > 14.0 for p in pts if 9.0 <= p[0] <= 11.0), pts)      # crosses the wall line only above its end
        clear = {"pads": [], "tracks": [], "vias": []}
        self.assertEqual(maze_waypoints([0, 0, 20, 20], 0.5, clear, "F.Cu", 0.3, [5.0, 8.0], [18.0, 18.0]), [[18.0, 18.0]])
        boxed = {"pads": [], "tracks": [{"net_code": 9, "layer": "F.Cu", "seg": [x1, y1, x2, y2], "width_mm": 0.2}
                                         for x1, y1, x2, y2 in ((3, 3, 3, 13), (3, 13, 13, 13), (13, 13, 13, 3), (13, 3, 3, 3))], "vias": []}
        self.assertIsNone(maze_waypoints([0, 0, 20, 20], 0.5, boxed, "F.Cu", 0.3, [5.0, 8.0], [18.0, 18.0]))

    def test_batch_replan_asks_once_for_every_failed_edge_of_the_pass(self):
        # F.Cu wall x in [9, 11] up to y = 19 and B.Cu wall x in [9, 11] from y = 3 down: both edges of the
        # default plan fail in the first pass. Batch mode defers them and makes ONE Global call with
        # both packets; the Global answers with a two-edge patch through the open strips (e1: via at A,
        # B.Cu along y < 3, via up to B's column; e2: via at C, B.Cu down the left column and along the
        # bottom to the thru-hole D). Both route, one replan spent.
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: [[9.0, 0.0, 11.0, 19.0]], 2: [[9.0, 3.0, 11.0, 20.0]]})
        calls = []

        def replan(packets, decision):
            calls.append((sorted(p.edge_id for p in packets), decision.request))
            patch = GlobalGuidePatch(patch_id="p-batch", based_on={"snapshot_id": packets[0].snapshot_ref, "plan_revision": self._plan.revision},
                                     replace_corridors=[{"id": "c1", "layer": "F.Cu", "bbox": [0.5, 0.5, 6.0, 6.0]},
                                                        {"id": "c2", "layer": "B.Cu", "bbox": [0.5, 0.5, 19.5, 2.9]},
                                                        {"id": "c3", "layer": "F.Cu", "bbox": [16.0, 0.5, 19.5, 19.5]},
                                                        {"id": "d1", "layer": "F.Cu", "bbox": [0.5, 14.0, 6.0, 19.5]},
                                                        {"id": "d2a", "layer": "B.Cu", "bbox": [0.5, 0.5, 6.0, 19.5]},
                                                        {"id": "d2b", "layer": "B.Cu", "bbox": [0.5, 0.5, 19.5, 2.9]}],
                                     replace_via_regions=[{"id": "v1", "layers": ["F.Cu", "B.Cu"], "bbox": [1.0, 1.0, 2.8, 2.8]},
                                                          {"id": "v2", "layers": ["B.Cu", "F.Cu"], "bbox": [17.0, 1.0, 19.0, 2.9]},
                                                          {"id": "w1", "layers": ["F.Cu", "B.Cu"], "bbox": [1.0, 15.0, 3.0, 17.0]}],
                                     replace_edges=[{"id": "e1", "source_cluster": "cluster:1:0", "target_cluster": "cluster:1:1",
                                                     "stages": ["c1", "c2", "c3"], "via_regions": ["v1", "v2"], "profile_ref": "Default"},
                                                    {"id": "e2", "source_cluster": "cluster:2:0", "target_cluster": "cluster:2:1",
                                                     "stages": ["d1", "d2a", "d2b"], "via_regions": ["w1"], "profile_ref": "Default"}])
            new, _aff = apply_patch(self._plan, patch, self._snap)
            self._plan = new
            return new

        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
        self._snap = snap
        g = guide_for(snap)
        g["budget"]["executor_calls"] = 400                                                    # two passes need more than 120 calls
        plan, rep0 = validate_guide_dict(g, snap)
        self.assertTrue(rep0.ok, rep0.errors)
        self._plan = plan
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id)
        orch = Orchestrator(gw, det, b, replan_fn=replan, recompile_fn=lambda p: compile_plan(p, snap), target_net_names=["N1", "N2"],
                            on_unroutable="skip", batch_replan=True)
        rep = orch.run(compile_plan(plan, snap))
        self.assertEqual(calls, [(["e1", "e2"], "batch")])                                     # one call, both edges
        self.assertEqual(rep.replans, 1)
        kinds = [e.kind for e in rep.events]
        self.assertEqual((kinds.count("defer_replan"), kinds.count("batch_replan")), (2, 1))
        self.assertEqual(rep.failed, [], [(t["task_id"], t["outcome"]) for t in rep.tasks])
        self.assertEqual(sorted(rep.final["done"]), ["e1", "e2"])
        self.assertEqual((eng.get_pad_groups()[1], eng.get_pad_groups()[2], len(eng.vias)), (1, 1, 3))
        self.assertEqual(eng._ckpts, {})

    def test_batch_replan_asks_again_when_the_answer_is_unusable(self):
        # same board and patch as above, but the Global's first batch answer is unusable (it failed
        # its checks even after the repair round: replan_fn returns None). With a round left the
        # edges stay in the batch and the next round asks again; with none left they are skipped.
        PATCH = dict(replace_corridors=[{"id": "c1", "layer": "F.Cu", "bbox": [0.5, 0.5, 6.0, 6.0]},
                                        {"id": "c2", "layer": "B.Cu", "bbox": [0.5, 0.5, 19.5, 2.9]},
                                        {"id": "c3", "layer": "F.Cu", "bbox": [16.0, 0.5, 19.5, 19.5]},
                                        {"id": "d1", "layer": "F.Cu", "bbox": [0.5, 14.0, 6.0, 19.5]},
                                        {"id": "d2a", "layer": "B.Cu", "bbox": [0.5, 0.5, 6.0, 19.5]},
                                        {"id": "d2b", "layer": "B.Cu", "bbox": [0.5, 0.5, 19.5, 2.9]}],
                     replace_via_regions=[{"id": "v1", "layers": ["F.Cu", "B.Cu"], "bbox": [1.0, 1.0, 2.8, 2.8]},
                                          {"id": "v2", "layers": ["B.Cu", "F.Cu"], "bbox": [17.0, 1.0, 19.0, 2.9]},
                                          {"id": "w1", "layers": ["F.Cu", "B.Cu"], "bbox": [1.0, 15.0, 3.0, 17.0]}],
                     replace_edges=[{"id": "e1", "source_cluster": "cluster:1:0", "target_cluster": "cluster:1:1",
                                     "stages": ["c1", "c2", "c3"], "via_regions": ["v1", "v2"], "profile_ref": "Default"},
                                    {"id": "e2", "source_cluster": "cluster:2:0", "target_cluster": "cluster:2:1",
                                     "stages": ["d1", "d2a", "d2b"], "via_regions": ["w1"], "profile_ref": "Default"}])

        def run(batch_rounds):
            eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: [[9.0, 0.0, 11.0, 19.0]], 2: [[9.0, 3.0, 11.0, 20.0]]})
            ad = NativeAdapter(eng)
            snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
            g = guide_for(snap)
            g["budget"]["executor_calls"] = 400
            g["budget"]["replan_cap"] = 4
            plan, rep0 = validate_guide_dict(g, snap)
            self.assertTrue(rep0.ok, rep0.errors)
            state = {"plan": plan}
            calls, hist = [], []

            def replan(packets, decision):
                calls.append((sorted(p.edge_id for p in packets), decision.request))
                hist.append({p.edge_id: list(p.history) for p in packets})
                if len(calls) == 1:
                    return None                                                     # unusable answer
                patch = GlobalGuidePatch(patch_id="p-batch", based_on={"snapshot_id": packets[0].snapshot_ref,
                                                                       "plan_revision": state["plan"].revision}, **PATCH)
                state["plan"], _aff = apply_patch(state["plan"], patch, snap)
                return state["plan"]

            b = Budget.from_plan(plan.budget)
            gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
            det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id)
            orch = Orchestrator(gw, det, b, replan_fn=replan, recompile_fn=lambda p: compile_plan(p, snap), target_net_names=["N1", "N2"],
                                on_unroutable="skip", batch_replan=True, batch_rounds=batch_rounds)
            return orch.run(compile_plan(plan, snap)), eng, calls, hist

        rep, eng, calls, hist = run(batch_rounds=2)
        self.assertEqual(calls, [(["e1", "e2"], "batch"), (["e1", "e2"], "batch")])    # asked again, same packets
        # the first call sees no history; the second sees the one round that failed (recorded once,
        # although the same packets were handed over twice), with the shape executed and where it died
        self.assertEqual([{k: len(v) for k, v in h.items()} for h in hist], [{"e1": 0, "e2": 0}, {"e1": 1, "e2": 1}])
        h1 = hist[1]["e1"][0]
        self.assertEqual((h1["round"], [st["corridor"] for st in h1["stages"]], h1["via_regions"]), (1, ["c1", "c2", "c3"], ["v1", "v2"]))
        self.assertEqual((h1["stages"][0]["layer"], len(h1["stages"][0]["bbox"])), ("F.Cu", 4))
        self.assertTrue(h1["died_at"] and h1["reason"] and len(h1["head"]) == 3, h1)
        self.assertIn("history", to_dict(next(iter(rep.tasks))["feedback"]) if isinstance(next(iter(rep.tasks))["feedback"], FeedbackPacket)
                      else next(iter(rep.tasks))["feedback"])                                    # serialized for the bundle
        self.assertEqual(rep.replans, 2)
        self.assertEqual(rep.failed, [], [(t["task_id"], t["outcome"]) for t in rep.tasks])
        self.assertEqual(sorted(rep.final["done"]), ["e1", "e2"])
        again = [e for e in rep.events if e.kind == "defer_replan" and "asked again" in e.detail["reason"]]
        self.assertEqual(sorted(e.detail["task"] for e in again), ["e1", "e2"])
        self.assertEqual((eng.get_pad_groups()[1], eng.get_pad_groups()[2], len(eng.vias)), (1, 1, 3))
        self.assertEqual(eng._ckpts, {})
        rep1, _eng1, calls1, _h = run(batch_rounds=1)                               # no round left: skipped as before
        self.assertEqual(len(calls1), 1)
        self.assertEqual(sorted(rep1.failed), ["e1", "e2"])
        skips = [e for e in rep1.events if e.kind == "skip"]
        self.assertTrue(skips and all("no usable" in e.detail["reason"] for e in skips))

    def test_shove_that_breaks_a_routed_net_is_rolled_back_and_retried_without_shove(self):
        # measured on sensorboard (no model): the shove `finish` of one edge split a net routed long
        # before into two pad groups; the loop never re-checked it and reported every edge done while
        # the official count was one short. Fake: net 2's path is blocked for walkaround by a soft
        # wall that shove passes, and this engine's shove "pushes" a net-1 track off the board.
        class ShoveBreaks(FakeEngine):
            def fix_route(self, x, y, **kw):
                ok = super().fix_route(x, y, **kw)
                if ok and self._mode == 1 and self._net == 2:
                    i = next((i for i, t in enumerate(self.tracks) if t.net_code == 1), None)
                    if i is not None:
                        del self.tracks[i]
                return ok

        WALL = self.DIAG_BLOCK                                                    # on both diagonals, F.Cu

        def run(guard: bool):
            eng = ShoveBreaks(PADS, bbox=[0, 0, 20, 20], shovable={1: WALL})
            ad = NativeAdapter(eng)
            snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
            plan, rep0 = validate_guide_dict(guide_for(snap), snap)
            self.assertTrue(rep0.ok, rep0.errors)
            b = Budget.from_plan(plan.budget)
            gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
            det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id)
            orch = Orchestrator(gw, det, b, target_net_names=["N1", "N2"], on_unroutable="skip",
                                rippable_nets={1, 2} if guard else None)
            return orch.run(compile_plan(plan, snap)), eng, det

        # without the guard (rippable_nets None): e2 "done" by shove, net 1 silently in two pieces
        rep0, eng0, _d = run(guard=False)
        self.assertEqual(([t["outcome"] for t in rep0.tasks], rep0.failed), (["done", "done"], []))
        self.assertEqual(eng0.get_pad_groups()[1], 2)
        # with it: the shoved attempt is detected and rolled back, e2 is run again walkaround-only (here
        # the obstacle-aware walk gets round the small block without shove); net 1 stays whole, the
        # record of e2 holds no shove step and the shove fallback is back on for the next task
        rep, eng, det = run(guard=True)
        dist = [e.detail for e in rep.events if e.kind == "disturbed"]
        self.assertEqual([(d["task"], d["nets"], d["retry"]) for d in dist], [("e2", [1], "no shove")])
        self.assertEqual((eng.get_pad_groups()[1], eng.get_pad_groups()[2], rep.failed), (1, 1, []))
        e2 = [t for t in rep.tasks if t["task_id"] == "e2"]
        self.assertEqual([t["outcome"] for t in e2], ["done"])
        self.assertTrue(e2[0]["steps"] and all(st["mode"] == 2 for st in e2[0]["steps"]))
        self.assertTrue(det.shove_fallback)
        self.assertEqual(eng._ckpts, {})

    def test_edge_already_joined_is_done_without_routing(self):
        # measured (MOD-MPU9150 n2_e9): GND was one pad group already, the engine answered "no ratsnest
        # edges for this net", the edge was reported failed and carried through four batch rounds while
        # the Global kept writing "redundant" in its note. Now: clusters already joined -> done, no step.
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20])
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
        plan, rep0 = validate_guide_dict(guide_for(snap), snap)
        self.assertTrue(rep0.ok, rep0.errors)
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id)
        e1 = next(t for t in compile_plan(plan, snap) if t.task_id == "e1")
        first = det.run_task(e1, b)
        self.assertEqual((first.outcome, eng.get_pad_groups()[1]), ("done", 1))
        tracks, calls = len(eng.tracks), ad.log.calls
        again = det.run_task(e1, b)                                                    # same edge, copper already there
        self.assertEqual((again.outcome, again.steps, len(eng.tracks)), ("done", [], tracks))
        self.assertEqual(again.feedback.failure_evidence.native_reason, "already joined")
        self.assertEqual(again.feedback.connectivity["pad_groups"], 1)
        self.assertLess(ad.log.calls - calls, 12)                                         # a few connectivity reads, no routing

    def test_budget_exhaustion_rolls_back_and_stops(self):
        def tight(g): g["budget"]["executor_calls"] = 14                   # reserve 8 -> 6 executor calls
        rep, eng, gw, _s, _p = self._run(FakeEngine(PADS, bbox=[0, 0, 20, 20]), guide_mut=tight)
        self.assertIn("budget exhausted", rep.stopped)
        self.assertEqual(rep.tasks[0]["outcome"], "aborted")
        self.assertTrue(rep.tasks[0]["rolled_back"])
        self.assertEqual((len(eng.tracks), len(eng.vias)), (0, 0))         # partial copper of the aborted task gone
        self.assertFalse(eng.is_routing())
        self.assertEqual(rep.final["drc_scoped"], 0)                        # final check still ran (reserve)
        self.assertEqual(eng._ckpts, {})
        self.assertLessEqual(rep.budget["spent_calls"], 14)

    def test_blocked_corridor_rolls_back_and_stops_without_global(self):
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={1: self.DIAG_BLOCK, 2: [[9.0, 0.0, 11.0, 20.0]]})   # wall across B.Cu trunk
        rep, eng, gw, _s, _p = self._run(eng)
        e1 = rep.tasks[0]
        self.assertEqual(e1["outcome"], "blocked")
        self.assertTrue(e1["rolled_back"])
        self.assertEqual((len(eng.tracks), len(eng.vias)), (0, 0))      # stable snapshot restored
        fb = e1["feedback"]
        self.assertEqual(fb["failure_evidence"].region_ref, "v2")
        self.assertIn(rep.stopped, ("replan requested but no Global provider",))

    def test_feedback_patch_closes_loop(self):
        # v0 corridor walk (finish_first=False): this test exercises the patch loop, not finish-first
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], blocked={2: [[9.0, 0.0, 11.0, 20.0]]})
        calls = []

        def replan(packet, decision):
            calls.append((packet.edge_id, decision.request))
            # Global moves the trunk to F.Cu entirely (no layer change) -> one wide F.Cu corridor
            patch = GlobalGuidePatch(patch_id="p1", based_on={"snapshot_id": packet.snapshot_ref, "plan_revision": self._plan.revision},
                                     replace_corridors=[{"id": "c2", "layer": "F.Cu", "bbox": [0.5, 0.5, 19.5, 19.5]}],
                                     replace_edges=[{"id": "e1", "source_cluster": "cluster:1:0", "target_cluster": "cluster:1:1",
                                                     "stages": ["c1", "c2", "c3"], "via_regions": [], "profile_ref": "Default"}])
            new, _aff = apply_patch(self._plan, patch, self._snap)
            self._plan = new
            return new

        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1, 2])
        self._snap = snap
        plan, rep0 = validate_guide_dict(guide_for(snap), snap)
        self._plan = plan
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id, finish_first=False)
        orch = Orchestrator(gw, det, b, replan_fn=replan, recompile_fn=lambda p: compile_plan(p, snap), target_net_names=["N1", "N2"])
        rep = orch.run(compile_plan(plan, snap))
        self.assertEqual(calls, [("e1", "alternate_layer")])
        self.assertEqual(rep.replans, 1)
        self.assertEqual([t["outcome"] for t in rep.tasks], ["blocked", "done", "done"])
        self.assertEqual(eng.get_pad_groups()[1], 1)
        self.assertEqual(len(eng.vias), 0)
        self.assertEqual(self._plan.revision, 2)

    def test_unsupported_task_reports_without_touching_board(self):
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20])
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1])
        plan, _ = validate_guide_dict(guide_for(snap), snap)
        tasks = compile_plan(plan, snap, capabilities=["route_to", "start_route", "select_net", "finish"])
        gw = ToolGateway(ad, budget=Budget())
        res = ScriptedDetailed(gw, "s", "g").run_task(tasks[0], Budget())
        self.assertEqual(res.outcome, "unsupported")
        self.assertEqual(len(eng.tracks), 0)

    def test_candidates_stay_inside_region(self):
        pts = candidate_points([10.0, 10.0, 12.0, 14.0], 0.5)
        for x, y in pts:
            self.assertTrue(10.5 <= x <= 11.5 and 10.5 <= y <= 13.5)

    def test_save_and_trace(self):
        rep, eng, gw, _s, _p = self._run(FakeEngine(PADS, bbox=[0, 0, 20, 20]))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.kicad_pcb")
            r = gw.call("save", output_path=p)
            self.assertTrue(r["ok"]); self.assertTrue(os.path.exists(p))
        self.assertTrue(all("tool" in t for t in gw.trace))


class TestAngleRepair(unittest.TestCase):
    """Post-routing joint repair (exec/angles.py): the evaluator's track-angle marks on our own nets."""

    def _board(self, **kw):
        eng = FakeEngine(PADS, bbox=[0, 0, 20, 20], **kw)
        ad = NativeAdapter(eng)
        # net 1 routed by hand as a hairpin: A (2,2) -> P (19, 19.5) -> B (18, 18): the joint at P is ~10 degrees
        for (x1, y1, x2, y2) in ((2.0, 2.0, 19.0, 19.5), (19.0, 19.5, 18.0, 18.0)):
            eng.tracks.append(NS(x1_mm=x1, y1_mm=y1, x2_mm=x2, y2_mm=y2, width_mm=0.2, layer=0, net_code=1, net_name="N1", uuid=""))
        return eng, ad

    def test_joint_detection_matches_the_evaluator_semantics(self):
        from route_agent.exec.angles import find_joints
        eng, ad = self._board()
        js = find_joints(eng, 1)
        self.assertEqual([(round(j["x"]), round(j["y"]), j["layer"]) for j in js], [(19, 20, 0)])
        self.assertLess(js[0]["angle"], 90.0)
        # a 45-degree corner is not a joint to fix; a corner sitting on a pad never is
        eng.tracks[:] = [NS(x1_mm=2.0, y1_mm=2.0, x2_mm=10.0, y2_mm=2.0, width_mm=0.2, layer=0, net_code=1, net_name="N1", uuid=""),
                         NS(x1_mm=10.0, y1_mm=2.0, x2_mm=18.0, y2_mm=10.0, width_mm=0.2, layer=0, net_code=1, net_name="N1", uuid=""),
                         NS(x1_mm=18.0, y1_mm=10.0, x2_mm=18.0, y2_mm=18.0, width_mm=0.2, layer=0, net_code=1, net_name="N1", uuid=""),
                         NS(x1_mm=18.0, y1_mm=18.0, x2_mm=19.0, y2_mm=5.0, width_mm=0.2, layer=0, net_code=1, net_name="N1", uuid="")]
        self.assertEqual(find_joints(eng, 1), [])                                   # 135, 135, pad B

    def test_hairpin_is_rerouted_as_one_move_and_connectivity_is_kept(self):
        from route_agent.exec.angles import repair_joints
        eng, ad = self._board()
        self.assertEqual(eng.get_pad_groups()[1], 1)
        r = repair_joints(ad, [1], {1: "N1"})
        self.assertEqual((r["before"], r["repaired"], r["failed"], r["after"], r["acute_before"], r["acute_after"]), (1, 1, 0, 0, 1, 0))
        t1 = [t for t in eng.tracks if t.net_code == 1]
        self.assertEqual(len(t1), 1)                                                 # A -> B in one segment
        self.assertEqual((eng.get_pad_groups()[1], eng._ckpts), (1, {}))

    def test_failed_repair_is_rolled_back(self):
        from route_agent.exec.angles import repair_joints
        eng, ad = self._board(blocked={1: [[9.6, 9.6, 10.4, 10.4]]})                 # the straight A -> B is blocked
        before = [(t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm) for t in eng.tracks if t.net_code == 1]
        r = repair_joints(ad, [1], {1: "N1"})
        self.assertEqual((r["before"], r["repaired"], r["failed"], r["after"]), (1, 0, 1, 1))
        self.assertEqual([(t.x1_mm, t.y1_mm, t.x2_mm, t.y2_mm) for t in eng.tracks if t.net_code == 1], before)
        self.assertEqual((eng.get_pad_groups()[1], eng._ckpts), (1, {}))

    def test_whole_pass_is_rolled_back_when_the_cleanup_would_cost_connectivity(self):
        # measured (TOBS, no model): a repair passed its own checks but KiCad's cleanup afterwards saw a
        # loose end and deleted 16 tracks of the net. The pass runs under one checkpoint and compares
        # every work-set net's pad groups after the cleanup; any loss reverts the whole pass.
        from route_agent.exec.angles import guarded_repair
        eng, ad = self._board()

        class _Breaks:
            def __init__(self, e): self.e = e
        def cleanup_tracks(**kw):
            if kw.get("dangling_tracks"):                                           # the finalize pass
                i = next((i for i, t in enumerate(eng.tracks) if t.net_code == 1), None)
                if i is not None:
                    del eng.tracks[i]
            return NS(ran=True, reject_reason="", items=[], removed=[], modified=[], changed=True)
        eng.cleanup_tracks = cleanup_tracks
        r = guarded_repair(ad, [1], {1: "N1"})
        self.assertTrue(r["pass_check"]["reverted"]); self.assertEqual(r["pass_check"]["nets_worse"], [1])
        self.assertEqual((eng.get_pad_groups()[1], len([t for t in eng.tracks if t.net_code == 1]), eng._ckpts), (1, 2, {}))
        self.assertEqual((r["after"], r["acute_after"]), (r["before"], r["acute_before"]))     # reported as not repaired

    def test_remaining_acute_joint_gets_one_more_guarded_pass(self):
        from unittest.mock import patch
        from route_agent.exec.angles import guarded_repair_passes
        first = {"before": 4, "after": 1, "acute_before": 1, "acute_after": 1,
                 "repaired": 3, "failed": 1, "nets": [], "pass_check": {"reverted": False}}
        second = {"before": 1, "after": 0, "acute_before": 1, "acute_after": 0,
                  "repaired": 1, "failed": 0, "nets": [], "pass_check": {"reverted": False}}
        with patch("route_agent.exec.angles.guarded_repair", side_effect=[first, second]) as repair:
            out = guarded_repair_passes(object(), [1], {1: "N1"})
        self.assertEqual(repair.call_count, 2)
        self.assertEqual((out["before"], out["after"], out["acute_before"], out["acute_after"]), (4, 0, 1, 0))
        self.assertEqual((out["repaired"], len(out["passes"])), (4, 2))

    def test_acute_joint_stops_at_five_guarded_passes(self):
        from unittest.mock import patch
        from route_agent.exec.angles import guarded_repair_passes
        stuck = {"before": 1, "after": 1, "acute_before": 1, "acute_after": 1,
                 "repaired": 0, "failed": 1, "nets": [], "pass_check": {"reverted": True}}
        with patch("route_agent.exec.angles.guarded_repair", return_value=stuck) as repair:
            out = guarded_repair_passes(object(), [1], {1: "N1"})
        self.assertEqual((repair.call_count, len(out["passes"]), out["acute_after"]), (5, 5, 1))

    def test_pipeline_flag_and_report_field(self):
        from route_agent.cli.pipeline import build_parser
        self.assertTrue(build_parser().parse_args(["benchset", "--list", "l", "--out", "o"]).angle_repair)
        self.assertFalse(build_parser().parse_args(["benchset", "--list", "l", "--out", "o", "--no-angle-repair"]).angle_repair)
        a = build_parser().parse_args(["repair-angles", "b.kicad_pcb", "--snapshot", "s.json", "--out", "o"])
        self.assertEqual((a.fn.__name__, a.mode, a.reward_config), ("cmd_repair_angles", 2, "drc_dense_errors_only_eval"))


class TestBaselineAndMultiCluster(unittest.TestCase):
    def test_baseline_corridor_keeps_overhanging_connector_anchors(self):
        eng = FakeEngine([
            {"ref": "J1", "pad": "1", "x": -0.425, "y": 5.0, "net": 1, "name": "N1"},
            {"ref": "J2", "pad": "1", "x": 10.425, "y": 5.0, "net": 1, "name": "N1"},
        ], bbox=[0, 0, 10, 10])
        snap = build_snapshot(NativeAdapter(eng), board_id="overhang", board_path="b", project_path="p", work_set_codes=[1])
        plan = baseline_plan(snap)
        rep = validate_guide(plan, snap)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(len(plan.connection_trees[0].edges), 1)

    def _orchestrate(self, eng, work):
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=list(work))
        plan = baseline_plan(snap)
        rep = validate_guide(plan, snap)
        self.assertTrue(rep.ok, rep.errors)
        b = Budget.from_plan(plan.budget)
        names = [snap.net(c).net_name for c in work]
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=names)
        det = ScriptedDetailed(gw, snap.board.snapshot_id, plan.plan_id)
        orch = Orchestrator(gw, det, b, target_net_names=names, on_unroutable="skip")
        return orch.run(compile_plan(plan, snap)), plan, snap

    def test_three_cluster_net_completes_edge_by_edge(self):
        pads = [{"ref": "A", "pad": "1", "x": 2.0, "y": 2.0, "net": 1, "name": "N1"},
                {"ref": "B", "pad": "1", "x": 10.0, "y": 2.0, "net": 1, "name": "N1"},
                {"ref": "C", "pad": "1", "x": 18.0, "y": 2.0, "net": 1, "name": "N1"}]
        eng = FakeEngine(pads, bbox=[0, 0, 20, 20])
        rep, plan, snap = self._orchestrate(eng, [1])
        self.assertEqual(baseline_summary(plan)["edges"], 2)
        # the first edge is done although the net still has two pad groups at that moment
        self.assertEqual([(t["outcome"], t["pad_groups"]) for t in rep.tasks], [("done", 2), ("done", 1)])
        self.assertEqual(eng.get_pad_groups()[1], 1)
        self.assertEqual(rep.failed, [])

    def test_baseline_cross_layer_edge_gets_a_via_region(self):
        pads = [{"ref": "A", "pad": "1", "x": 2.0, "y": 2.0, "net": 1, "name": "N1", "layer": 0, "type": "smd"},
                {"ref": "B", "pad": "1", "x": 18.0, "y": 18.0, "net": 1, "name": "N1", "layer": 2, "type": "smd"}]
        eng = FakeEngine(pads, bbox=[0, 0, 20, 20])
        rep, plan, snap = self._orchestrate(eng, [1])
        self.assertEqual((len(plan.corridors), len(plan.via_regions)), (2, 1))
        self.assertEqual([c.layer for c in plan.corridors], ["F.Cu", "B.Cu"])
        self.assertEqual([t["outcome"] for t in rep.tasks], ["done"])
        self.assertEqual(len(eng.vias), 1)

    def test_baseline_is_deterministic_and_orders_short_nets_first(self):
        eng, ad, snap = make_snapshot(work=(1, 2))
        p1, p2 = baseline_plan(snap), baseline_plan(snap)
        self.assertEqual(to_dict(p1), to_dict(p2))
        self.assertEqual(p1.net_order, [1, 2])                   # equal length -> net code order
        self.assertTrue(all(c.bbox[0] >= 0 and c.bbox[2] <= 20 for c in p1.corridors))   # clamped to the board
        self.assertTrue(validate_guide(p1, snap).ok)
        from route_agent.cli.pipeline import build_parser
        a = build_parser().parse_args(["bench", "b.kicad_pcb", "--out", "o", "--skip-nets", "GNDPWR", "--max-clusters", "6"])
        self.assertEqual((a.skip_nets, a.max_clusters, a.margin), (["GNDPWR"], 6, 1.5))


class TestProviders(unittest.TestCase):
    def test_file_provider_needs_a_path_and_replays(self):
        from route_agent.providers import FileGlobalProvider, make_global_provider
        with self.assertRaises(ValueError):
            make_global_provider("file")
        with self.assertRaises(ValueError):
            make_global_provider("nope")
        with tempfile.TemporaryDirectory() as d:
            g = os.path.join(d, "g.json"); p = os.path.join(d, "p.json")
            with open(g, "w") as fg, open(p, "w") as fp:
                json.dump({"plan_id": "g"}, fg); json.dump({"patch_id": "p"}, fp)
            prov = make_global_provider("file", guide_path=g, patch_path=p)
            self.assertEqual(prov.plan({})[0]["plan_id"], "g")
            self.assertEqual(prov.replan({})[0]["patch_id"], "p")
            self.assertEqual(FileGlobalProvider(None, p).replan({})[0]["patch_id"], "p")
            with self.assertRaises(ValueError):
                FileGlobalProvider(None, p).plan({})
            self.assertEqual(FileGlobalProvider(g).replan({})[0], {})

    def test_llm_detailed_returns_task_results_and_survives_bad_tools(self):
        from route_agent.providers import LLMDetailed
        from route_agent.prompts.bundle import build_detailed_bundle

        class FakeOpenAI:
            def __init__(self, rounds):
                self.rounds = list(rounds)
                self.chat = NS(completions=NS(create=self._create))
            def _create(self, **kw):
                r = self.rounds.pop(0)
                if isinstance(r, list):
                    tcs = [NS(id=f"c{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(r)]
                    return NS(choices=[NS(message=NS(tool_calls=tcs, content=None))])
                return NS(choices=[NS(message=NS(tool_calls=None, content=r))])

        eng, ad, snap = make_snapshot()
        plan, _ = validate_guide_dict(guide_for(snap), snap)
        tasks = compile_plan(plan, snap)
        b = Budget.from_plan(plan.budget)
        gw = ToolGateway(ad, budget=b, allowed_profiles=snap.rules.profiles, target_net_names=["N1", "N2"])
        client = FakeOpenAI([
            [("apply_profile", '{"name": "Default"}'), ("select_net", '{"net_code": 1}'),
             ("start_route", '{"x": 2.0, "y": 2.0, "layer": "F.Cu"}')],
            [("teleport", '{}'), ("route_to", 'not json'), ("finish", '{"mode": 2}')],
            '{"outcome": "done", "last_head": [18.0, 18.0, 1]}',
            [("select_net", '{"net_code": 2}'), ("start_route", '{"x": 2.0, "y": 18.0, "layer": "F.Cu"}'),
             ("route_to", '{"x": 18.0, "y": 2.0}')],
            "blocked, giving up",
        ])
        det = LLMDetailed(gw, lambda t: build_detailed_bundle(t, snap, tools_schema()["tools"]),
                          snap.board.snapshot_id, plan.plan_id, client=client, max_rounds=5)
        orch = Orchestrator(gw, det, b, target_net_names=["N1", "N2"])
        rep = orch.run(tasks)
        self.assertEqual([t["outcome"] for t in rep.tasks], ["done", "done"])
        self.assertEqual(eng.get_pad_groups()[1], 1)
        kinds = [(s["kind"], s["ok"]) for s in rep.tasks[0]["steps"]]
        self.assertIn(("teleport", False), kinds)                           # unknown tool fed back, not raised
        self.assertIn(("route_to", False), kinds)                           # bad JSON fed back, not raised
        self.assertEqual(rep.tasks[0]["feedback"]["last_head"], [18.0, 18.0, 1])
        self.assertFalse(eng.is_routing()); self.assertIsNone(ad.current_net)
        self.assertEqual(rep.final["drc_errors"], 0)

    def test_claude_fences_are_stripped_only_in_the_anthropic_provider(self):
        from route_agent.providers import AnthropicGlobalProvider, _extract_json, _strip_fences
        with self.assertRaises(ValueError):
            _extract_json('```json\n{"a": 1}\n```')                        # shared boundary stays strict
        self.assertEqual(_strip_fences('```json\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(_strip_fences('{"a": 1}'), '{"a": 1}')
        self.assertEqual(_strip_fences('Reasoning summary.\n\n```json\n{"a": 1}\n```'), '{"a": 1}')
        with self.assertRaises(ValueError):
            _extract_json(_strip_fences('```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```'))

        class FakeAnthropic:
            def __init__(self):
                self.messages = NS(create=self._create)
            def _create(self, **kw):
                self.kw = kw
                return NS(content=[NS(type="text", text='```json\n{"plan_id": "p"}\n```')],
                          usage=NS(input_tokens=10, output_tokens=5))

        fake = FakeAnthropic()
        prov = AnthropicGlobalProvider(model="claude-haiku-5-5", client=fake)
        plan, usage = prov.plan({"system": "s", "user": "u", "images": []})
        self.assertEqual((plan, usage["model"], usage["prompt_tokens"]), ({"plan_id": "p"}, "claude-haiku-5-5", 10))
        self.assertEqual(fake.kw["model"], "claude-haiku-5-5")
        AnthropicGlobalProvider(model="claude-opus-5-5", client=fake, effort="low").plan(
            {"system": "s", "user": "u", "images": []})
        self.assertEqual(fake.kw["output_config"], {"effort": "low"})
        with self.assertRaises(ValueError):
            AnthropicGlobalProvider(client=fake, effort="invalid")

    def test_env_file_loader(self):
        from route_agent.env import load_env_file, parse_env
        text = ("# comment\nexport OPENAI_API_KEY='sk-live-123'\nROUTEAGENT_OPENAI_MODEL=gpt-5-nano\n"
                "ANTHROPIC_API_KEY=sk-ant-REPLACE-ME\nBROKEN LINE\nEMPTY=\n")
        self.assertEqual(parse_env(text), {"OPENAI_API_KEY": "sk-live-123", "ROUTEAGENT_OPENAI_MODEL": "gpt-5-nano"})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "env")
            with open(path, "w") as fh:
                fh.write(text)
            env = {"OPENAI_API_KEY": "stale-from-bashrc", "OTHER": "kept"}
            info = load_env_file(path, env)
            self.assertEqual((info["loaded"], info["keys"]), (True, ["OPENAI_API_KEY", "ROUTEAGENT_OPENAI_MODEL"]))
            self.assertEqual(env["OPENAI_API_KEY"], "sk-live-123")            # the file wins over the shell
            self.assertEqual(env["OTHER"], "kept")
            self.assertFalse(load_env_file(os.path.join(d, "missing"), env)["loaded"])

    def test_cli_global_replay_and_status(self):
        from route_agent.cli.pipeline import build_parser
        a = build_parser().parse_args(["global", "s.json", "--provider", "file", "--guide", "g.json"])
        self.assertEqual((a.provider, a.guide), ("file", "g.json"))
        a = build_parser().parse_args(["execute", "s.json", "g.json", "--out", "o", "--provider", "file", "--patch", "p.json"])
        self.assertEqual(a.patch, "p.json")
        self.assertTrue(build_parser().parse_args(["status", "--probe"]).probe)


class TestOfficialScore(unittest.TestCase):
    def test_trim_and_injected_evaluator(self):
        from route_agent.eval_official import headline, official_score, trim_metrics
        raw = {"success": False, "clean_pass": False, "routability": 0.8, "potential_gain": 12.5, "final_potential": 20.0,
               "drv_errors_only_count": 3, "drv_errors_and_promoted_count": 3,
               "total_drv_count": 5, "wirelength_mm": 12.3456, "via_count": 2, "track_count": 9, "reward_config": "pcbworld_reward",
               "track_angle_drv": {"count": 2, "violations": [{"angle_deg": 90.0}, {"angle_deg": 90.0}]},
               "drv_breakdown": {"errors_only_by_type": [
                   {"severity": "ERROR", "error_code": 1, "error_type": "Missing connection between items", "count": 2},
                   {"severity": "ERROR", "error_code": 4, "error_type": "Clearance violation", "count": 1}]},
               "extras": {"initial_unrouted_edges": 10, "unrouted_edges_remaining": 2, "pad_groups_initial": 12,
                          "pad_groups_remaining": 4, "per_net": {}}}
        t = trim_metrics(raw)
        self.assertEqual((t["edges_routed"], t["edges_total"], t["drv_unconnected"], t["drv_rule_errors"], t["track_angle_drv"],
                          t["wirelength_mm"], t["via_count"]), (8, 10, 2, 1, 2, 12.346, 2))
        calls = []
        def fake_eval(pcb, pro, check_angle=45):
            calls.append((pcb, pro, check_angle)); return raw
        o = official_score("/x/b.kicad_pcb", "/x/b.kicad_pro", _evaluate=fake_eval)
        self.assertEqual(calls, [("/x/b.kicad_pcb", "/x/b.kicad_pro", 45)])
        self.assertIn("edges 8/10", headline(o)); self.assertIn("rule 1", headline(o))
        from route_agent.cli.pipeline import build_parser
        a = build_parser().parse_args(["score", "b.kicad_pcb", "--violations"])
        self.assertEqual((a.board, a.check_angle, a.violations), ("b.kicad_pcb", 45, True))


class TestPromptBundles(unittest.TestCase):
    def test_bundles(self):
        _e, _a, snap = make_snapshot()
        gb = build_global_bundle(snap)
        self.assertIn(snap.board.snapshot_id, gb["user"])
        self.assertEqual(gb["response_schema"]["title"], "RouteGuidePlan")
        plan, _ = validate_guide_dict(guide_for(snap), snap)
        tasks = compile_plan(plan, snap)
        db = build_detailed_bundle(tasks[0], snap, tools_schema()["tools"])
        self.assertIn("e1", db["user"])
        self.assertTrue(len(db["tools"]) >= 10)


if __name__ == "__main__":
    unittest.main()


class TestViewCalibration(unittest.TestCase):
    def test_landmark_calibration_recovers_known_affine(self):
        from route_agent.snapshot.views import ViewFrame, calibrate_native_view, render_views
        pads = [{"ref": "A", "pad": "1", "x": 2.0, "y": 2.0, "net": 1, "w": 1.2, "h": 1.2, "type": "smd", "shape": "rect", "layer": 0},
                {"ref": "B", "pad": "1", "x": 17.0, "y": 15.0, "net": 1, "w": 1.0, "h": 2.0, "type": "smd", "shape": "rect", "layer": 0},
                {"ref": "C", "pad": "1", "x": 6.0, "y": 12.0, "net": 2, "w": 1.5, "h": 1.0, "type": "smd", "shape": "rect", "layer": 0},
                {"ref": "D", "pad": "1", "x": 12.0, "y": 4.0, "net": 2, "w": 1.0, "h": 1.0, "type": "smd", "shape": "rect", "layer": 0}]
        eng = FakeEngine(pads, bbox=[0, 0, 20, 20])
        ad = NativeAdapter(eng)
        snap = build_snapshot(ad, board_id="f", board_path="f", project_path="p", work_set_codes=[1])
        with tempfile.TemporaryDirectory() as d:
            recs = render_views(snap, d, px_per_mm=10.0)
            f_cu = next(r for r in recs if r.view_id == "F_Cu_state")
            frame = ViewFrame(snap.geometry.outline_bbox, 10.0)
            svg = os.path.join(d, "fake.svg")
            with open(svg, "w") as fh:
                fh.write(f'<svg width="{frame.w_mm:.4f}mm" height="{frame.h_mm:.4f}mm" viewBox="0 0 1 1"></svg>')
            cal = calibrate_native_view(f_cu.path, svg, snap, "F.Cu")
            self.assertTrue(cal["ok"], cal)
            a = cal["pixel_to_mm"]
            self.assertAlmostEqual(a[0], f_cu.pixel_to_mm[0], places=2)   # savefig may round the canvas by 1 px
            self.assertLess(abs(a[2] - f_cu.pixel_to_mm[2]), 2 * a[0])
            self.assertLess(abs(a[5] - f_cu.pixel_to_mm[5]), 2 * a[4])
