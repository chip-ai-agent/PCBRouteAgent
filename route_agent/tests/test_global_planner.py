"""Global boundary tests with FakeEngine fixtures, tiny PNGs and local providers.

No model API, native routing, or performance result is exercised by these tests.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from route_agent.global_planner.planner import GlobalPlanner
from route_agent.global_planner.serialization import (
    canonical_json,
    parse_plan,
    require_json,
)
from route_agent.guide.compiler import compile_plan
from route_agent.guide.validate import validate_guide_dict
from route_agent.providers import FileGlobalProvider, _extract_json, _image_base64
from route_agent.schema.common import to_dict
from route_agent.schema.snapshot import ViewRecord
from route_agent.tests.test_pipeline import guide_for, make_snapshot


def tiny_png(rgb=(80, 120, 160), width=4, height=4):
    """A complete noninterlaced RGB PNG, without Pillow or a plotting dependency."""

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\0" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def add_views(snap, root):
    layers = list(snap.geometry.layer_names)
    roles = [
        ("board_context", layers),
        ("F_Cu_state", ["F.Cu"]),
        ("B_Cu_state", ["B.Cu"]),
        ("connectivity_demand", layers),
    ]
    directory = root / "views"
    directory.mkdir()
    for index, (role, visible) in enumerate(roles):
        path = directory / f"{role}.png"
        path.write_bytes(tiny_png((40 + index * 30, 80, 120)))
        snap.views.append(
            ViewRecord(
                view_id=role,
                snapshot_id=snap.board.snapshot_id,
                layers=visible,
                bbox_mm=[0, 0, 20, 20],
                image_size=[4, 4],
                pixel_to_mm=[5, 0, 0, 0, 5, 0],
                mirrored=False,
                path=str(Path("views") / path.name),
                calibration="exact",
            )
        )


class RecordingProvider:
    """An explicit local backend double using the team's plan/replan interface."""

    def __init__(self, response, callback=None):
        self.response = response
        self.callback = callback
        self.requests = []

    def _respond(self, method, bundle):
        self.requests.append((method, copy.deepcopy(bundle)))
        if self.callback:
            self.callback(bundle)
        return copy.deepcopy(self.response), {"provider": "test_stub", "tokens": 0}

    def plan(self, bundle):
        return self._respond("plan", bundle)

    def replan(self, bundle):
        return self._respond("replan", bundle)


class StrictGlobalJSONTests(unittest.TestCase):
    def test_plain_object_is_accepted_without_changing_numbers(self):
        self.assertEqual(
            parse_plan(' {"revision": 2, "value": 0.2} '), {"revision": 2, "value": 0.2}
        )

    def test_duplicate_keys_are_rejected_at_every_depth(self):
        for text in (
            '{"a": 1, "a": 2}',
            '{"based_on": {"revision": 1, "revision": 2}}',
        ):
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(ValueError, "duplicate"),
            ):
                parse_plan(text)

    def test_nonfinite_constants_and_overflow_are_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                parse_plan('{"value": ' + token + "}")

    def test_wrapped_text_fences_and_multiple_objects_are_rejected(self):
        for text in (
            'Here is the plan: {"a": 1}',
            '```json\n{"a":1}\n```',
            '{"a":1}\n{"b":2}',
            '{"a":1} trailing',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _extract_json(text)

    def test_nonobject_and_nonjson_values_are_rejected(self):
        for text in ("[]", "null", "1", '"plan"'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_plan(text)
        for value in ({1: "not a JSON key"}, {"x": float("nan")}, {"x": (1, 2)}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                require_json(value)

    def test_canonical_json_is_stable_and_strict(self):
        self.assertEqual(canonical_json({"b": 2, "a": [1]}), '{"a":[1],"b":2}')
        with self.assertRaises(ValueError):
            canonical_json({"a": float("inf")})


class GlobalPlannerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.engine, self.adapter, self.snap = make_snapshot()
        add_views(self.snap, self.root)
        self.guide = guide_for(self.snap)
        self.guide["based_on"]["revision"] = self.snap.board.revision

    def planner(self, response=None, callback=None):
        provider = RecordingProvider(
            self.guide if response is None else response, callback
        )
        return GlobalPlanner(provider, asset_root=self.root), provider

    def test_bundle_contains_real_png_bytes_and_team_schema(self):
        planner = GlobalPlanner(asset_root=self.root)
        bundle = planner.build_bundle(self.snap)
        self.assertEqual(bundle["response_schema"]["title"], "RouteGuidePlan")
        self.assertEqual(
            [image["view_id"] for image in bundle["images"]],
            [view.view_id for view in self.snap.views],
        )
        for image in bundle["images"]:
            data = base64.b64decode(image["data_base64"], validate=True)
            self.assertEqual(data, Path(image["path"]).read_bytes())
            self.assertEqual(image["sha256"], hashlib.sha256(data).hexdigest())
        manifest = bundle["input_manifest"]
        self.assertEqual(manifest["snapshot_id"], self.snap.board.snapshot_id)
        self.assertEqual(manifest["revision"], self.snap.board.revision)
        self.assertIn(self.snap.board.rules_hash, bundle["user"])
        self.assertIn("renderer provenance required", manifest["image_binding"])

    def test_returned_plan_compiles_with_existing_team_compiler(self):
        planner, provider = self.planner()
        guide, usage = planner.plan(self.snap)
        plan, report = validate_guide_dict(guide, self.snap)
        self.assertTrue(report.ok, report.errors)
        tasks = compile_plan(plan, self.snap)
        self.assertEqual([task.task_id for task in tasks], ["e1", "e2"])
        self.assertTrue(all(task.execution["supported"] for task in tasks))
        self.assertEqual(provider.requests[0][0], "plan")
        self.assertTrue(usage["backend_invoked"])
        self.assertIsNone(usage["provider_invoked"])
        self.assertFalse(usage["routing_executed"])
        self.assertEqual((len(self.engine.tracks), len(self.engine.vias)), (0, 0))

    def test_missing_or_wrong_current_state_binding_is_rejected(self):
        for key in ("snapshot_id", "rules_hash", "revision"):
            for missing in (True, False):
                with self.subTest(key=key, missing=missing):
                    guide = copy.deepcopy(self.guide)
                    if missing:
                        del guide["based_on"][key]
                    else:
                        guide["based_on"][key] = 99 if key == "revision" else "stale"
                    planner, _ = self.planner(guide)
                    with self.assertRaises(ValueError):
                        planner.plan(self.snap)

    def test_plan_cannot_include_nets_outside_work_set(self):
        self.snap.work_set = [1]
        for net in self.snap.connectivity:
            net.in_work_set = net.net_code == 1
        planner, _ = self.planner()
        with self.assertRaisesRegex(ValueError, "work-set"):
            planner.plan(self.snap)

    def test_missing_work_set_tree_is_rejected(self):
        guide = copy.deepcopy(self.guide)
        guide["connection_trees"].pop()
        guide["net_order"] = [1]
        planner, _ = self.planner(guide)
        with self.assertRaisesRegex(ValueError, "work-set"):
            planner.plan(self.snap)

    def test_unconnected_component_cannot_be_omitted_from_tree(self):
        third = copy.deepcopy(self.snap.net(1).clusters[0])
        third.id = "cluster:1:2"
        third.anchors[0].pad_id = "F:1"
        third.anchors[0].x_mm = 10
        third.anchors[0].y_mm = 10
        self.snap.net(1).clusters.append(third)
        self.snap.net(1).pad_groups = 3
        planner, _ = self.planner()
        with self.assertRaisesRegex(ValueError, "spanning"):
            planner.plan(self.snap)

    def test_duplicate_net_order_and_trees_are_rejected(self):
        for target in ("net_order", "connection_trees"):
            with self.subTest(target=target):
                guide = copy.deepcopy(self.guide)
                guide[target].append(copy.deepcopy(guide[target][0]))
                planner, _ = self.planner(guide)
                with self.assertRaises(ValueError):
                    planner.plan(self.snap)

    def test_correct_edge_count_cannot_hide_a_cycle_and_omitted_component(self):
        third = copy.deepcopy(self.snap.net(1).clusters[0])
        third.id = "cluster:1:2"
        third.anchors[0].pad_id = "F:1"
        self.snap.net(1).clusters.append(third)
        guide = copy.deepcopy(self.guide)
        reverse = copy.deepcopy(guide["connection_trees"][0]["edges"][0])
        reverse.update(
            id="reverse-e1",
            source_cluster="cluster:1:1",
            target_cluster="cluster:1:0",
            stages=list(reversed(reverse["stages"])),
            via_regions=list(reversed(reverse["via_regions"])),
        )
        guide["connection_trees"][0]["edges"].append(reverse)
        planner, _ = self.planner(guide)
        with self.assertRaisesRegex(ValueError, "cycle"):
            planner.plan(self.snap)

    def test_frozen_scope_is_rejected(self):
        self.snap.execution.frozen_scope = [1]
        planner, _ = self.planner()
        with self.assertRaisesRegex(ValueError, "frozen_scope"):
            planner.plan(self.snap)

    def test_layer_change_requires_advertised_via_capability(self):
        self.snap.execution.capabilities.remove("route_via")
        planner, _ = self.planner()
        with self.assertRaisesRegex(ValueError, "unsupported Detailed capabilities"):
            planner.plan(self.snap)

    def test_snapshot_change_during_provider_generation_is_rejected(self):
        planner, _ = self.planner(
            callback=lambda _bundle: setattr(self.snap.board, "revision", 1)
        )
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            planner.plan(self.snap)
        self.assertIsNone(planner.last_run)

    def test_rule_change_during_provider_generation_is_rejected(self):
        def change(_bundle):
            self.snap.rules.profiles["Default"]["track_width_mm"] = 0.8

        planner, _ = self.planner(callback=change)
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            planner.plan(self.snap)

    def test_image_change_during_provider_generation_is_rejected(self):
        image = self.root / self.snap.views[0].path
        planner, _ = self.planner(
            callback=lambda _bundle: image.write_bytes(tiny_png((240, 10, 10)))
        )
        with self.assertRaisesRegex(ValueError, "image inputs changed"):
            planner.plan(self.snap)

    def test_provider_receives_a_copy_of_the_bundle(self):
        before = to_dict(self.snap)

        def change(bundle):
            bundle["input_manifest"]["snapshot_id"] = "provider-local-change"
            bundle["images"][0]["sha256"] = "provider-local-change"

        planner, _ = self.planner(callback=change)
        guide, usage = planner.plan(self.snap)
        self.assertEqual(to_dict(self.snap), before)
        self.assertEqual(guide, self.guide)
        self.assertEqual(
            usage["input_manifest"]["snapshot_id"], self.snap.board.snapshot_id
        )

    def test_file_replay_is_recorded_without_claiming_model_invocation(self):
        path = self.root / "guide.json"
        path.write_text(json.dumps(self.guide))
        planner = GlobalPlanner(FileGlobalProvider(str(path)), asset_root=self.root)
        guide, usage = planner.plan(self.snap)
        self.assertEqual(guide, self.guide)
        self.assertEqual(usage["mode"], "file_replay")
        self.assertFalse(usage["provider_invoked"])
        self.assertTrue(usage["backend_invoked"])
        self.assertFalse(usage["routing_executed"])

    def test_file_replay_uses_the_same_strict_json_boundary(self):
        path = self.root / "duplicate.json"
        path.write_text('{"plan_id":"first", "plan_id":"second"}')
        planner = GlobalPlanner(FileGlobalProvider(str(path)), asset_root=self.root)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            planner.plan(self.snap)

    def test_provider_uses_frozen_bytes_without_reopening_path(self):
        data = tiny_png()
        encoded = base64.b64encode(data).decode("ascii")
        self.assertEqual(
            _image_base64({"path": "missing.png", "data_base64": encoded}), encoded
        )

    def test_nonjson_provider_output_is_rejected(self):
        for response in ([], {**self.guide, "note": float("nan")}):
            with self.subTest(response=response):
                planner, _ = self.planner(response)
                with self.assertRaises(ValueError):
                    planner.plan(self.snap)

    def test_replan_uses_current_snapshot_and_returns_complete_incremented_plan(self):
        previous = copy.deepcopy(self.guide)
        old_id = self.snap.board.snapshot_id
        self.snap.board.snapshot_id = "new-committed-snapshot"
        self.snap.board.revision = 7
        for view in self.snap.views:
            view.snapshot_id = self.snap.board.snapshot_id
        revised = copy.deepcopy(previous)
        revised["revision"] = previous["revision"] + 1
        revised["based_on"] = {
            "snapshot_id": self.snap.board.snapshot_id,
            "rules_hash": self.snap.board.rules_hash,
            "revision": 7,
        }
        planner, provider = self.planner(revised)
        guide, usage = planner.plan(
            self.snap,
            feedback={"snapshot_ref": old_id, "outcome": "blocked"},
            previous_plan=previous,
        )
        self.assertEqual(guide, revised)
        self.assertEqual(provider.requests[0][0], "replan")
        self.assertIn("Previous plan", provider.requests[0][1]["user"])
        self.assertEqual(usage["input_manifest"]["revision"], 7)
        parsed, report = validate_guide_dict(guide, self.snap)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(len(compile_plan(parsed, self.snap)), 2)

    def test_replan_rejects_partial_patch_and_unincremented_revision(self):
        for response in (
            {"replace_corridors": []},
            self.guide,
            {**self.guide, "revision": "2"},
        ):
            with self.subTest(response=response):
                planner, _ = self.planner(response)
                with self.assertRaises(ValueError):
                    planner.plan(
                        self.snap,
                        feedback={"snapshot_ref": self.snap.board.snapshot_id},
                        previous_plan=self.guide,
                    )

    def test_replan_requires_complete_previous_plan_and_bound_feedback(self):
        cases = [
            ({"snapshot_ref": self.snap.board.snapshot_id}, None),
            ({}, {}),
            ({}, self.guide),
            ({"snapshot_ref": "another-snapshot"}, self.guide),
        ]
        for feedback, previous in cases:
            with self.subTest(feedback=feedback, previous=previous):
                planner, provider = self.planner()
                with self.assertRaises(ValueError):
                    planner.plan(self.snap, feedback=feedback, previous_plan=previous)
                self.assertEqual(provider.requests, [])

    def test_feedback_and_previous_plan_mutation_are_rejected(self):
        for target in ("feedback", "previous_plan"):
            with self.subTest(target=target):
                previous = copy.deepcopy(self.guide)
                feedback = {
                    "snapshot_ref": self.snap.board.snapshot_id,
                    "outcome": "blocked",
                }
                revised = copy.deepcopy(self.guide)
                revised["revision"] = 2

                def change(_bundle, target=target, feedback=feedback, previous=previous):
                    if target == "feedback":
                        feedback["outcome"] = "done"
                    else:
                        previous["revision"] = 9

                planner, _ = self.planner(revised, callback=change)
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    planner.plan(self.snap, feedback=feedback, previous_plan=previous)


if __name__ == "__main__":
    unittest.main()
