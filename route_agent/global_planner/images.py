"""Validate and freeze Global image content using the team's Snapshot contract.

Core views follow render_views: board_context, one <layer>_state per copper
layer, then connectivity_demand. Optional native_<layer> supplements require
include_native=True and calibration; uncalibrated native artifacts are excluded.

Computed hashes describe bytes read in this call. ViewRecord has no historical
image hash or geometry-content provenance, so matching snapshot_id cannot prove
that the image was regenerated after a geometry change or an ID reassignment.
PNG framing/CRCs are checked here; this module is not a full raster decoder.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
import struct
import zlib
from dataclasses import asdict
from pathlib import Path

from route_agent.schema.snapshot import Snapshot, ViewRecord


class GlobalImageError(ValueError):
    """Image content or its declared registration is not acceptable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GlobalImageError(message)


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _bbox(value, label: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == 4 and all(_finite(x) for x in value),
        f"{label}: invalid bbox",
    )
    _require(value[0] < value[2] and value[1] < value[3], f"{label}: degenerate bbox")
    return value


def _png_size(data: bytes, max_pixels: int) -> tuple[int, int]:
    _require(
        data.startswith(b"\x89PNG\r\n\x1a\n"), "only PNG image content is supported"
    )
    offset, size, saw_data, saw_end = 8, None, False, False
    depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
    while offset < len(data):
        _require(offset + 12 <= len(data), "truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        end = offset + length + 12
        _require(end <= len(data), "truncated PNG payload")
        payload = data[offset + 8 : offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        _require(zlib.crc32(kind + payload) & 0xFFFFFFFF == crc, "PNG CRC mismatch")
        if size is None:
            _require(
                kind == b"IHDR" and length == 13, "PNG must start with a valid IHDR"
            )
            w, h, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            _require(
                w > 1 and h > 1 and w * h <= max_pixels,
                "invalid or excessive PNG dimensions",
            )
            _require(
                depth in depths.get(color, set())
                and compression == 0
                and filtering == 0
                and interlace in (0, 1),
                "invalid PNG encoding",
            )
            size = (w, h)
        else:
            _require(kind != b"IHDR", "duplicate PNG IHDR")
        _require(kind != b"acTL", "animated PNG is unsupported")
        saw_data = saw_data or kind == b"IDAT"
        offset = end
        if kind == b"IEND":
            _require(
                length == 0 and offset == len(data), "invalid PNG end or trailing data"
            )
            saw_end = True
            break
    _require(size is not None and saw_data and saw_end, "incomplete PNG")
    return size


def _registration(view: ViewRecord, board_bbox: list[float]) -> None:
    _require(
        view.calibration in ("exact", "landmark"),
        f"{view.view_id}: calibrated view required",
    )
    image_size = view.image_size
    _require(
        isinstance(image_size, list)
        and len(image_size) == 2
        and all(
            isinstance(n, int) and not isinstance(n, bool) and n > 1 for n in image_size
        ),
        f"{view.view_id}: invalid image_size",
    )
    affine = view.pixel_to_mm
    _require(
        isinstance(affine, list)
        and len(affine) == 6
        and all(_finite(n) for n in affine),
        f"{view.view_id}: invalid affine",
    )
    a, b, c, d, e, f = affine
    det = a * e - b * d
    _require(abs(det) > 1e-15, f"{view.view_id}: singular affine")
    _require(
        isinstance(view.mirrored, bool) and (det < 0) == view.mirrored,
        f"{view.view_id}: affine/mirroring mismatch",
    )
    declared = _bbox(view.bbox_mm, view.view_id)
    w, h = image_size
    corners = [
        (a * x + b * y + c, d * x + e * y + f)
        for x, y in ((0, 0), (w, 0), (w, h), (0, h))
    ]
    mapped = [
        min(x for x, _ in corners),
        min(y for _, y in corners),
        max(x for x, _ in corners),
        max(y for _, y in corners),
    ]
    # render_views rounds raster dimensions while retaining its physical frame.
    # Allow at most one pixel of registration rounding, not arbitrary offsets.
    tolerance = max(math.hypot(a, d), math.hypot(b, e)) * 1.01 + 1e-9
    _require(
        all(abs(x - y) <= tolerance for x, y in zip(declared, mapped, strict=True)),
        f"{view.view_id}: affine/bbox mismatch",
    )
    _require(
        max(declared[0], board_bbox[0]) < min(declared[2], board_bbox[2])
        and max(declared[1], board_bbox[1]) < min(declared[3], board_bbox[3]),
        f"{view.view_id}: view does not overlap board geometry",
    )


def read_global_images(
    snap: Snapshot,
    asset_root: Path,
    max_image_bytes: int = 20_000_000,
    max_total_bytes: int = 50_000_000,
    max_pixels: int = 40_000_000,
    *,
    include_native: bool = False,
) -> list[dict]:
    """Return frozen provider-ready PNG blocks in core-then-supplement order.

    Absolute paths are accepted within asset_root; relative paths resolve against
    that root. Foreign absolute paths are never opened: they map only to the
    unique existing asset_root/<view_id>.png or asset_root/views/<view_id>.png.
    Ambiguous relocation, traversal and symlink escapes are rejected. Providers use
    data_base64 from this result instead of re-reading path. The computed sha256
    is an invocation manifest, not comparison with a trusted prior image hash.
    This method deliberately does not adopt the old routeagent schema or hash.
    """
    try:
        _require(isinstance(snap, Snapshot), "Snapshot instance required")
        _require(
            isinstance(snap.board.snapshot_id, str) and bool(snap.board.snapshot_id),
            "nonempty snapshot_id required",
        )
        _require(
            isinstance(snap.board.board_id, str) and bool(snap.board.board_id),
            "nonempty board_id required",
        )
        for value in (max_image_bytes, max_total_bytes, max_pixels):
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value > 0,
                "image limits must be positive integers",
            )
        _require(isinstance(include_native, bool), "include_native must be boolean")
        board_bbox = _bbox(snap.geometry.outline_bbox, "board geometry")
        layers = snap.geometry.layer_names
        _require(
            isinstance(layers, list)
            and len(layers) >= 2
            and all(
                isinstance(name, str)
                and re.fullmatch(r"(?:F|B|In[1-9][0-9]*)\.Cu", name)
                for name in layers
            ),
            "invalid canonical copper layers",
        )
        _require(
            len(set(layers)) == len(layers) and "F.Cu" in layers and "B.Cu" in layers,
            "duplicate or incomplete copper layers",
        )
        copper_roles = {f"{layer.replace('.', '_')}_state": layer for layer in layers}
        core = ["board_context", *copper_roles, "connectivity_demand"]
        native_roles = {f"native_{layer.replace('.', '_')}": layer for layer in layers}
        _require(
            isinstance(snap.views, list)
            and all(isinstance(view, ViewRecord) for view in snap.views),
            "ViewRecord list required",
        )
        records = {}
        excluded = []
        for view in snap.views:
            _require(
                isinstance(view.view_id, str) and view.view_id not in records,
                "duplicate or invalid view_id",
            )
            _require(
                view.view_id in core or view.view_id in native_roles,
                f"unsupported Global view_id: {view.view_id}",
            )
            _require(
                view.snapshot_id == snap.board.snapshot_id,
                f"{view.view_id}: snapshot_id mismatch",
            )
            _require(
                isinstance(view.layers, list)
                and len(view.layers) == len(set(view.layers)),
                f"{view.view_id}: duplicate or invalid layer metadata",
            )
            if view.view_id in copper_roles:
                _require(
                    view.layers == [copper_roles[view.view_id]],
                    f"{view.view_id}: copper layer mismatch",
                )
            elif view.view_id in native_roles:
                wanted = native_roles[view.view_id]
                _require(
                    wanted in view.layers and set(view.layers) <= {wanted, "Edge.Cuts"},
                    f"{view.view_id}: native layer mismatch",
                )
            else:
                _require(
                    set(view.layers) == set(layers),
                    f"{view.view_id}: context layer mismatch",
                )
            records[view.view_id] = view
            if view.calibration == "uncalibrated" and view.view_id in native_roles:
                excluded.append(view.view_id)
        _require(all(role in records for role in core), "missing core Global view(s)")
        root = Path(asset_root).resolve(strict=True)
        _require(root.is_dir(), "asset_root must be a directory")
        omitted_native = [
            role for role in native_roles if role in records and not include_native
        ]
        ordered = core + (
            [role for role in native_roles if role in records and role not in excluded]
            if include_native
            else []
        )
        outputs, used_paths, total = [], set(), 0
        for role in ordered:
            view = records[role]
            _registration(view, board_bbox)
            _require(
                isinstance(view.path, str) and bool(view.path),
                f"{role}: image path required",
            )
            path = Path(view.path)
            _require(".." not in path.parts, f"{role}: image path traversal rejected")
            resolution = "declared_path"
            if path.is_absolute():
                resolved = path.resolve(strict=False)
                if not resolved.is_relative_to(root):
                    # A path located under the root but following a symlink out
                    # is an escape, not a portable foreign-machine path.
                    _require(
                        not path.is_relative_to(root)
                        and not path.is_relative_to(Path(asset_root).absolute()),
                        f"{role}: image escapes asset_root",
                    )
                    candidates = [root / f"{role}.png", root / "views" / f"{role}.png"]
                    existing = [p for p in candidates if p.exists()]
                    _require(
                        len(existing) == 1,
                        f"{role}: foreign path requires exactly one portable asset-root match",
                    )
                    path = existing[0]
                    resolution = "asset_root_view_id"
                else:
                    path = resolved
            else:
                path = root / path
            path = path.resolve(strict=True)
            _require(
                path.is_relative_to(root) and path.is_file(),
                f"{role}: image escapes asset_root or is not a file",
            )
            _require(
                path not in used_paths, "Global roles must use distinct image files"
            )
            used_paths.add(path)
            with path.open("rb") as stream:
                data = stream.read(max_image_bytes + 1)
            _require(len(data) <= max_image_bytes, f"{role}: image exceeds byte limit")
            total += len(data)
            _require(total <= max_total_bytes, "Global image bundle exceeds byte limit")
            _require(
                list(_png_size(data, max_pixels)) == view.image_size,
                f"{role}: actual PNG dimensions differ from metadata",
            )
            outputs.append(
                {
                    **asdict(view),
                    "declared_path": view.path,
                    "path": str(path),
                    "path_resolution": resolution,
                    "data_base64": base64.b64encode(data).decode("ascii"),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "media_type": "image/png",
                    "byte_length": len(data),
                    "source_kind": "core" if role in core else "native_supplement",
                    "integrity": {
                        "hash_scope": "bytes_read_this_call",
                        "historical_hash_verified": False,
                        "geometry_freshness_verified": False,
                        "excluded_uncalibrated_views": list(excluded),
                        "omitted_native_views": omitted_native,
                    },
                }
            )
        return outputs
    except GlobalImageError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as exc:
        raise GlobalImageError(f"Global image input rejected: {exc}") from exc
