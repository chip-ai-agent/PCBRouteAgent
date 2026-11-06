"""Global image tests: synthetic PNGs and Snapshot dataclasses, no engine/model."""

import base64
import copy
import hashlib
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from route_agent.global_planner.images import GlobalImageError, read_global_images
from route_agent.schema.snapshot import (
    BoardRef,
    ExecutionContext,
    Geometry,
    RuleContext,
    Snapshot,
    ViewRecord,
)


def png_bytes(width=20, height=20, color=0):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = (b"\0" + bytes([color]) * width) * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class TestGlobalImages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snap = Snapshot(
            board=BoardRef(
                "synthetic",
                "not-loaded.kicad_pcb",
                "",
                "snap-1",
                rules_hash="opaque-team-hash",
            ),
            geometry=Geometry(["F.Cu", "B.Cu"], [0, 0, 20, 20], [], [], [], [], [], []),
            connectivity=[],
            rules=RuleContext([], 0.1, 0.1, 0.5, 0.1, {}, []),
            views=[],
            execution=ExecutionContext(),
            work_set=[],
        )
        self.add_view("board_context", self.snap.geometry.layer_names)
        self.add_view("F_Cu_state", ["F.Cu"])
        self.add_view("B_Cu_state", ["B.Cu"])
        self.add_view("connectivity_demand", self.snap.geometry.layer_names)

    def add_view(self, view_id, layers, calibration="exact"):
        relative = view_id + ".png"
        (self.root / relative).write_bytes(png_bytes(color=len(self.snap.views)))
        view = ViewRecord(
            view_id,
            self.snap.board.snapshot_id,
            list(layers),
            [0, 0, 20, 20],
            [20, 20],
            [1, 0, 0, 0, 1, 0],
            False,
            relative,
            calibration,
        )
        self.snap.views.append(view)
        return view

    def rejected(self, pattern):
        with self.assertRaisesRegex(GlobalImageError, pattern):
            read_global_images(self.snap, self.root)

    def test_core_bytes_are_frozen_and_manifest_is_honest(self):
        result = read_global_images(self.snap, self.root)
        self.assertEqual(
            [v["view_id"] for v in result], [v.view_id for v in self.snap.views]
        )
        for got, view in zip(result, self.snap.views, strict=True):
            actual = (self.root / view.path).read_bytes()
            self.assertEqual(base64.b64decode(got["data_base64"]), actual)
            self.assertEqual(got["sha256"], hashlib.sha256(actual).hexdigest())
            self.assertFalse(got["integrity"]["historical_hash_verified"])
            self.assertFalse(got["integrity"]["geometry_freshness_verified"])
        (self.root / self.snap.views[0].path).write_bytes(png_bytes(color=99))
        self.assertNotEqual(
            base64.b64decode(result[0]["data_base64"]),
            (self.root / self.snap.views[0].path).read_bytes(),
        )

    def test_multilayer_uses_one_core_image_per_layer(self):
        self.snap.geometry.layer_names = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
        self.snap.views[0].layers = list(self.snap.geometry.layer_names)
        self.snap.views[-1].layers = list(self.snap.geometry.layer_names)
        self.add_view("In1_Cu_state", ["In1.Cu"])
        self.add_view("In2_Cu_state", ["In2.Cu"])
        result = read_global_images(self.snap, self.root)
        self.assertEqual(
            [v["view_id"] for v in result],
            [
                "board_context",
                "F_Cu_state",
                "In1_Cu_state",
                "In2_Cu_state",
                "B_Cu_state",
                "connectivity_demand",
            ],
        )

    def test_missing_core_is_rejected(self):
        self.snap.views.pop(1)
        self.rejected("missing core")

    def test_duplicate_role_is_rejected(self):
        self.snap.views.append(copy.deepcopy(self.snap.views[0]))
        self.rejected("duplicate.*view_id")

    def test_stale_snapshot_id_is_rejected(self):
        self.snap.views[1].snapshot_id = "old"
        self.rejected("snapshot_id mismatch")

    def test_incorrect_copper_layer_is_rejected(self):
        self.snap.views[1].layers = ["B.Cu"]
        self.rejected("copper layer mismatch")

    def test_calibrated_native_supplement_is_included(self):
        self.add_view("native_F_Cu", ["F.Cu", "Edge.Cuts"], "landmark")
        result = read_global_images(self.snap, self.root, include_native=True)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[-1]["source_kind"], "native_supplement")

    def test_native_supplements_are_opt_in(self):
        self.add_view("native_F_Cu", ["F.Cu", "Edge.Cuts"], "landmark")
        result = read_global_images(self.snap, self.root)
        self.assertEqual(len(result), 4)
        self.assertEqual(
            result[0]["integrity"]["omitted_native_views"], ["native_F_Cu"]
        )

    def test_uncalibrated_native_is_excluded_and_reported(self):
        view = self.add_view("native_B_Cu", ["B.Cu", "Edge.Cuts"], "uncalibrated")
        view.bbox_mm = [0, 0, 0, 0]
        view.pixel_to_mm = [0] * 6
        (self.root / view.path).unlink()
        result = read_global_images(self.snap, self.root)
        self.assertEqual(len(result), 4)
        self.assertEqual(
            result[0]["integrity"]["excluded_uncalibrated_views"], ["native_B_Cu"]
        )

    def test_uncalibrated_core_is_rejected(self):
        self.snap.views[0].calibration = "uncalibrated"
        self.rejected("calibrated view required")

    def test_absolute_path_inside_root_is_accepted(self):
        self.snap.views[0].path = str(self.root / self.snap.views[0].path)
        self.assertEqual(len(read_global_images(self.snap, self.root)), 4)

    def test_foreign_absolute_path_relocates_only_by_view_id(self):
        self.snap.views[0].path = "/home/other-computer/case/views/board_context.png"
        result = read_global_images(self.snap, self.root)
        self.assertEqual(result[0]["path_resolution"], "asset_root_view_id")
        self.assertEqual(
            Path(result[0]["path"]), (self.root / "board_context.png").resolve()
        )

    def test_foreign_path_supports_views_directory_and_rejects_ambiguity(self):
        view = self.snap.views[0]
        (self.root / "views").mkdir()
        source = self.root / view.path
        dest = self.root / "views" / view.path
        dest.write_bytes(source.read_bytes())
        view.path = "/home/other-computer/board_context.png"
        self.rejected("exactly one portable")
        source.unlink()
        result = read_global_images(self.snap, self.root)
        self.assertEqual(Path(result[0]["path"]), dest.resolve())

    def test_relative_traversal_is_rejected(self):
        self.snap.views[0].path = "../other.png"
        self.rejected("traversal")

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.png"
            target.write_bytes(png_bytes())
            link = self.root / "escape.png"
            link.symlink_to(target)
            self.snap.views[0].path = link.name
            self.rejected("escapes asset_root")

    def test_actual_dimensions_are_checked(self):
        self.snap.views[0].image_size = [21, 20]
        self.snap.views[0].bbox_mm = [0, 0, 21, 20]
        self.rejected("actual PNG dimensions")

    def test_png_crc_is_checked(self):
        path = self.root / self.snap.views[0].path
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        self.rejected("CRC")

    def test_affine_bbox_mismatch_is_rejected(self):
        self.snap.views[0].pixel_to_mm[2] = 3
        self.rejected("affine/bbox mismatch")

    def test_raster_rounding_tolerance_matches_main_viewframe(self):
        self.snap.views[0].bbox_mm = [0, 0, 20.4, 19.6]
        self.assertEqual(len(read_global_images(self.snap, self.root)), 4)

    def test_mirroring_must_match_affine(self):
        self.snap.views[2].mirrored = True
        self.rejected("mirroring mismatch")
        self.snap.views[2].pixel_to_mm = [-1, 0, 20, 0, 1, 0]
        self.assertEqual(len(read_global_images(self.snap, self.root)), 4)

    def test_geometry_overlap_required(self):
        self.snap.geometry.outline_bbox = [100, 100, 120, 120]
        self.rejected("does not overlap")

    def test_byte_and_pixel_limits_are_enforced(self):
        for kw in (
            {"max_image_bytes": 10},
            {"max_total_bytes": 10},
            {"max_pixels": 10},
        ):
            with (
                self.subTest(limit=kw),
                self.assertRaisesRegex(GlobalImageError, "limit|excessive"),
            ):
                read_global_images(self.snap, self.root, **kw)

    def test_unknown_geometry_layer_is_rejected(self):
        self.snap.geometry.layer_names = ["F.Cu", "Bogus"]
        self.rejected("canonical copper")


if __name__ == "__main__":
    unittest.main()
