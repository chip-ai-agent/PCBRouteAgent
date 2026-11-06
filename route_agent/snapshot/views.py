"""The four Global input views, rendered from the structured snapshot with an exact affine.

All views share one physical mm frame (the board bbox + margin) at ``px_per_mm`` so a pixel
in any of them maps to the same board point. ``pixel_to_mm`` is exact by construction
(no landmark calibration needed); the optional native kicad-cli layer SVG is exported as a
raw artifact next to them and its ViewRecord is marked ``uncalibrated``.

Views
    board_context        outline, footprint courtyards + refs, pads, keepouts
    <layer>_state        per copper layer: pads on that layer, tracks, vias, outline (not mirrored)
    connectivity_demand  remaining ratsnest, work-set nets highlighted, rest dimmed
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from route_agent.schema.snapshot import Snapshot, ViewRecord

_WORK_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#bcbd22"]


class ViewFrame:
    def __init__(self, bbox: list[float], px_per_mm: float = 20.0, margin_mm: float = 1.5) -> None:
        x1, y1, x2, y2 = bbox
        self.x0, self.y0 = x1 - margin_mm, y1 - margin_mm
        self.w_mm, self.h_mm = (x2 - x1) + 2 * margin_mm, (y2 - y1) + 2 * margin_mm
        self.s = float(px_per_mm)
        self.w_px, self.h_px = int(round(self.w_mm * self.s)), int(round(self.h_mm * self.s))

    def mm_to_px(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) * self.s, (y - self.y0) * self.s

    @property
    def affine(self) -> list[float]:
        return [1.0 / self.s, 0.0, self.x0, 0.0, 1.0 / self.s, self.y0]

    @property
    def bbox(self) -> list[float]:
        return [self.x0, self.y0, self.x0 + self.w_mm, self.y0 + self.h_mm]


def _fig(frame: ViewFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = 100
    fig = plt.figure(figsize=(frame.w_px / dpi, frame.h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(frame.x0, frame.x0 + frame.w_mm)
    ax.set_ylim(frame.y0 + frame.h_mm, frame.y0)        # y down, like KiCad
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax, dpi


def _pad_patch(p, color, alpha=0.9, lw=0.0):
    from matplotlib.patches import Circle, Ellipse, Rectangle
    if p.shape == "circle":
        return Circle((p.x_mm, p.y_mm), max(p.width_mm, p.height_mm) / 2, fc=color, ec="none", alpha=alpha, lw=lw)
    if p.shape == "oval":
        return Ellipse((p.x_mm, p.y_mm), p.width_mm, p.height_mm, fc=color, ec="none", alpha=alpha, lw=lw)
    return Rectangle((p.x_mm - p.width_mm / 2, p.y_mm - p.height_mm / 2), p.width_mm, p.height_mm,
                     fc=color, ec="none", alpha=alpha, lw=lw)


def _outline(ax, snap: Snapshot):
    for x1, y1, x2, y2 in snap.geometry.outline:
        ax.plot([x1, x2], [y1, y2], color="#222222", lw=1.2, solid_capstyle="round")


def render_views(snap: Snapshot, out_dir: str | Path, px_per_mm: float = 20.0,
                 crop_bbox: list[float] | None = None, tag: str = "") -> list[ViewRecord]:
    """Render the bundle; returns ViewRecords (also appended to ``snap.views``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = ViewFrame(crop_bbox or snap.geometry.outline_bbox, px_per_mm)
    g = snap.geometry
    records: list[ViewRecord] = []
    work = set(snap.work_set)
    color_of = {code: _WORK_COLORS[i % len(_WORK_COLORS)] for i, code in enumerate(sorted(work))}
    layer_color = {"F.Cu": "#c0392b", "B.Cu": "#1f5fbf"}

    def save(fig, dpi, view_id, layers, object_map=None):
        path = out_dir / f"{tag}{view_id}.png"
        fig.savefig(path, dpi=dpi, facecolor="white")
        import matplotlib.pyplot as plt
        plt.close(fig)
        rec = ViewRecord(view_id=view_id, snapshot_id=snap.board.snapshot_id, layers=layers,
                         bbox_mm=frame.bbox, image_size=[frame.w_px, frame.h_px], pixel_to_mm=frame.affine,
                         mirrored=False, path=str(path), calibration="exact", object_map=object_map or {})
        records.append(rec)

    # 1. board_context
    fig, ax, dpi = _fig(frame)
    _outline(ax, snap)
    omap = {}
    for f in g.footprints:
        if f.courtyard_bbox:
            x1, y1, x2, y2 = f.courtyard_bbox
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#999999", lw=0.5, ls="--"))
        ax.text(f.x_mm, f.y_mm, f.ref, fontsize=5, ha="center", va="center", color="#333333")
    for p in g.pads:
        col = "#666666" if p.pad_type == "smd" else "#8e6a2b"
        ax.add_patch(_pad_patch(p, col, alpha=0.8))
        px, py = frame.mm_to_px(p.x_mm, p.y_mm)
        omap[p.id] = [round(px - p.width_mm * frame.s / 2, 1), round(py - p.height_mm * frame.s / 2, 1),
                      round(px + p.width_mm * frame.s / 2, 1), round(py + p.height_mm * frame.s / 2, 1)]
    for z in g.keepouts:
        xs = [pt[0] for pt in z.pts] + [z.pts[0][0]]
        ys = [pt[1] for pt in z.pts] + [z.pts[0][1]]
        ax.fill(xs, ys, color="#ff00ff", alpha=0.25, hatch="//")
    save(fig, dpi, "board_context", list(g.layer_names), omap)

    # 2. per-layer copper state
    for layer in g.layer_names:
        fig, ax, dpi = _fig(frame)
        _outline(ax, snap)
        lc = layer_color.get(layer, "#444444")
        for p in g.pads:
            if layer in p.layers:
                ax.add_patch(_pad_patch(p, lc if p.net_code > 0 else "#aaaaaa", alpha=0.75))
                if p.net_code in work:
                    ax.add_patch(_pad_patch(p, color_of[p.net_code], alpha=1.0))
        for t in g.tracks:
            if t.layer == layer:
                ax.plot([t.x1_mm, t.x2_mm], [t.y1_mm, t.y2_mm], color=lc, lw=max(0.5, t.width_mm * frame.s * 0.72),
                        solid_capstyle="round", alpha=0.9)
        for v in g.vias:
            if layer in v.layers:
                from matplotlib.patches import Circle
                ax.add_patch(Circle((v.x_mm, v.y_mm), v.diameter_mm / 2, fc="#555555", ec="none"))
                ax.add_patch(Circle((v.x_mm, v.y_mm), v.drill_mm / 2, fc="white", ec="none"))
        for z in g.keepouts:
            if z.layer == layer:
                xs = [pt[0] for pt in z.pts] + [z.pts[0][0]]
                ys = [pt[1] for pt in z.pts] + [z.pts[0][1]]
                ax.fill(xs, ys, color="#ff00ff", alpha=0.25, hatch="//")
        save(fig, dpi, f"{layer.replace('.', '_')}_state", [layer])

    # 3. connectivity_demand
    fig, ax, dpi = _fig(frame)
    _outline(ax, snap)
    for p in g.pads:
        ax.add_patch(_pad_patch(p, "#cccccc", alpha=0.6))
    for n in snap.connectivity:
        col = color_of.get(n.net_code, "#bbbbbb")
        lw = 1.6 if n.in_work_set else 0.5
        alpha = 1.0 if n.in_work_set else 0.5
        for x1, y1, x2, y2 in n.remaining_edges:
            ax.plot([x1, x2], [y1, y2], color=col, lw=lw, alpha=alpha, ls="-" if n.in_work_set else ":")
        if n.in_work_set:
            for c in n.clusters:
                for a in c.anchors:
                    ax.plot(a.x_mm, a.y_mm, "o", ms=4, mfc="none", mec=col, mew=1.2)
                    ax.text(a.x_mm + 0.4, a.y_mm - 0.4, f"{a.pad_id}", fontsize=5, color=col)
            if n.remaining_edges:
                x1, y1, x2, y2 = n.remaining_edges[0]
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, n.net_name, fontsize=6, color=col, weight="bold")
    save(fig, dpi, "connectivity_demand", list(g.layer_names))

    snap.views.extend(records)
    return records


def find_kicad_cli(repo_root: str | Path | None = None) -> str | None:
    env = os.environ.get("KICAD_CLI")
    if env and Path(env).exists():
        return env
    w = shutil.which("kicad-cli")
    if w:
        return w
    from route_agent.paths import pcbworld_root
    root = Path(repo_root) if repo_root else pcbworld_root()
    cand = root / "build_rl" / "kicad" / "kicad-cli"
    return str(cand) if cand.exists() else None


def export_native_layer_svgs(board_path: str, out_dir: str | Path, layers: list[str],
                             repo_root: str | Path | None = None, rasterize: bool = True) -> list[dict]:
    """kicad-cli ``pcb export svg`` per copper layer (+Edge.Cuts). Returns [{layer, svg, png}].
    The in-tree build of kicad-cli needs ``KICAD_RUN_FROM_BUILD_DIR=1`` to find its kiface."""
    cli = find_kicad_cli(repo_root)
    if not cli:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KICAD_RUN_FROM_BUILD_DIR="1")
    out = []
    for layer in layers:
        svg = out_dir / f"native_{layer.replace('.', '_')}.svg"
        cmd = [cli, "pcb", "export", "svg", "--layers", f"{layer},Edge.Cuts", "--mode-single",
               "--page-size-mode", "2", "-o", str(svg), str(board_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
        if r.returncode != 0 or not svg.exists():
            continue
        rec = {"layer": layer, "svg": str(svg), "png": ""}
        if rasterize:
            try:
                import cairosvg
                png = svg.with_suffix(".png")
                cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=1600)
                rec["png"] = str(png)
            except Exception as ex:          # rasteriser optional
                rec["error"] = f"rasterize: {ex}"
        out.append(rec)
    return out


# --------------------------------------------------------------------------- native calibration
def _svg_page_mm(svg_path: str) -> tuple[float, float] | None:
    import re
    head = Path(svg_path).read_text(errors="ignore")[:4000]
    m = re.search(r'width="([\d.]+)mm"\s+height="([\d.]+)mm"', head)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _pad_template(pads, sx: float, sy: float, X0: float, Y0: float, W: int, H: int):
    """Binary image of the snapshot's pads at the plot's scale, placed with provisional origin (X0, Y0)."""
    import numpy as np
    T = np.zeros((H, W), dtype=np.float32)
    for p in pads:
        cx, cy = (p.x_mm - X0) / sx, (p.y_mm - Y0) / sy
        hw, hh = max(1.0, p.width_mm / sx / 2), max(1.0, p.height_mm / sy / 2)
        x1, x2 = int(max(0, cx - hw)), int(min(W, cx + hw + 1))
        y1, y2 = int(max(0, cy - hh)), int(min(H, cy + hh + 1))
        if x2 > x1 and y2 > y1:
            T[y1:y2, x1:x2] = 1.0
    return T


def calibrate_native_view(png_path: str, svg_path: str, snap: Snapshot, layer: str,
                          tol_px: float = 2.0) -> dict:
    """Landmark calibration of a kicad-cli layer plot.

    Scale comes from the SVG page size (mm) over the PNG size (px). The translation is
    found by FFT cross-correlation of the plot's copper mask with a template of the
    snapshot's pads on that layer (robust to the outline and to extra items). Every pad is
    then checked against the nearest copper blob centroid; ``ok`` when the median residual
    is within ``tol_px`` and >= 90% of the pads hit copper.
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    from scipy.signal import fftconvolve
    page = _svg_page_mm(svg_path)
    if page is None:
        return {"ok": False, "reason": "svg page size not found"}
    im = Image.open(png_path).convert("RGBA")
    W, H = im.size
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    rgb = np.asarray(Image.alpha_composite(bg, im).convert("RGB")).astype(int)
    sx, sy = page[0] / W, page[1] / H
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    cand = sat > 60
    if cand.sum() < 50:
        return {"ok": False, "reason": "no coloured copper pixels"}
    # copper = every saturated pixel (pads of any alpha / net colour); the outline and the
    # drill holes are unsaturated and drop out
    mask = cand
    pads = [p for p in snap.geometry.pads if layer in p.layers]
    if not pads:
        return {"ok": False, "reason": f"no pads on {layer}"}
    bb = snap.geometry.outline_bbox
    # provisional origin: board bbox centred on the page
    X0 = bb[0] - (page[0] - (bb[2] - bb[0])) / 2
    Y0 = bb[1] - (page[1] - (bb[3] - bb[1])) / 2
    T = _pad_template(pads, sx, sy, X0, Y0, W, H)
    M = mask.astype(np.float32)
    corr = fftconvolve(M, T[::-1, ::-1], mode="same")
    py_, px_ = np.unravel_index(int(corr.argmax()), corr.shape)
    dy, dx = py_ - H // 2, px_ - W // 2          # M ≈ T shifted by (dy, dx)
    x0, y0 = X0 - sx * dx, Y0 - sy * dy
    # per-pad verification
    lab, n = ndimage.label(mask)
    cents = ndimage.center_of_mass(mask, lab, range(1, n + 1)) if n else []
    cents = np.array([[c[1], c[0]] for c in cents]) if n else np.zeros((0, 2))
    res, hits = [], 0
    for p in pads:
        px, py = (p.x_mm - x0) / sx, (p.y_mm - y0) / sy
        r = max(2.0, 0.5 * min(p.width_mm / sx, p.height_mm / sy))
        if 0 <= int(px) < W and 0 <= int(py) < H:
            win = mask[max(0, int(py - r)):int(py + r) + 1, max(0, int(px - r)):int(px + r) + 1]
            if win.any():
                hits += 1
        if len(cents):
            res.append(float(np.hypot(cents[:, 0] - px, cents[:, 1] - py).min()))
    med = float(np.median(res)) if res else float("inf")
    return {"ok": bool(med <= tol_px and hits >= 0.9 * len(pads)),
            "pixel_to_mm": [float(sx), 0.0, float(x0), 0.0, float(sy), float(y0)],
            "bbox_mm": [float(x0), float(y0), float(x0 + page[0]), float(y0 + page[1])], "image_size": [W, H],
            "shift_px": [int(dx), int(dy)], "median_residual_px": round(med, 2), "median_residual_mm": round(med * sx, 4),
            "pads_checked": len(pads), "pads_hit": hits, "tol_px": tol_px}


def native_view_records(snap: Snapshot, native: list[dict]) -> list[ViewRecord]:
    """ViewRecords for native exports: landmark-calibrated when the check passes, else
    ``uncalibrated`` (the image is still shipped, only as a raw artifact)."""
    recs = []
    for n in native:
        if not n.get("png"):
            continue
        cal = calibrate_native_view(n["png"], n["svg"], snap, n["layer"])
        if cal.get("ok"):
            rec = ViewRecord(view_id=f"native_{n['layer'].replace('.', '_')}", snapshot_id=snap.board.snapshot_id,
                             layers=[n["layer"], "Edge.Cuts"], bbox_mm=cal["bbox_mm"], image_size=cal["image_size"],
                             pixel_to_mm=cal["pixel_to_mm"], mirrored=False, path=n["png"], calibration="landmark",
                             object_map={"_calibration": [cal["median_residual_px"], cal["pads_hit"], cal["pads_checked"]]})
        else:
            from PIL import Image
            W, H = Image.open(n["png"]).size
            rec = ViewRecord(view_id=f"native_{n['layer'].replace('.', '_')}", snapshot_id=snap.board.snapshot_id,
                             layers=[n["layer"], "Edge.Cuts"], bbox_mm=[0, 0, 0, 0], image_size=[W, H],
                             pixel_to_mm=[0, 0, 0, 0, 0, 0], mirrored=False, path=n["png"], calibration="uncalibrated")
        n["calibration"] = {k: v for k, v in cal.items() if k != "pixel_to_mm"}
        recs.append(rec)
    snap.views.extend(recs)
    return recs


# --------------------------------------------------------------------------- guide overlay
def render_guide_overlay(snap: Snapshot, plan, out_path: str | Path, px_per_mm: float = 20.0,
                         crop_bbox: list[float] | None = None, edges: list[str] | None = None,
                         routed: Snapshot | None = None) -> str:
    """Corridors (shaded, per layer colour) and via regions (orange) over the pad geometry.
    ``routed`` optionally overlays committed copper from a post-execution snapshot."""
    from matplotlib.patches import Rectangle
    frame = ViewFrame(crop_bbox or snap.geometry.outline_bbox, px_per_mm)
    fig, ax, dpi = _fig(frame)
    _outline(ax, snap)
    g = snap.geometry
    work = set(snap.work_set)
    layer_color = {"F.Cu": "#c0392b", "B.Cu": "#1f5fbf"}
    for p in g.pads:
        ax.add_patch(_pad_patch(p, "#bbbbbb" if p.net_code not in work else "#222222", alpha=0.7))
    want = None
    if edges:
        want = set()
        for _t, e in plan.edges():
            if e.id in edges:
                want |= set(e.stages) | set(e.via_regions)
    for c in plan.corridors:
        if want is not None and c.id not in want:
            continue
        x1, y1, x2, y2 = c.bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc=layer_color.get(c.layer, "#888"), ec=layer_color.get(c.layer, "#888"),
                               alpha=0.18, lw=0.8))
        ax.text(x1 + 0.2, y1 + 0.9, c.id, fontsize=5, color=layer_color.get(c.layer, "#888"))
    for v in plan.via_regions:
        if want is not None and v.id not in want:
            continue
        x1, y1, x2, y2 = v.bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc="#ff9900", ec="#cc6600", alpha=0.35, lw=0.8))
    if routed is not None:
        for t in routed.geometry.tracks:
            if t.net_code in work:
                ax.plot([t.x1_mm, t.x2_mm], [t.y1_mm, t.y2_mm], color=layer_color.get(t.layer, "#000"),
                        lw=max(0.8, t.width_mm * frame.s * 0.72), solid_capstyle="round")
        for vv in routed.geometry.vias:
            if vv.net_code in work:
                from matplotlib.patches import Circle
                ax.add_patch(Circle((vv.x_mm, vv.y_mm), vv.diameter_mm / 2, fc="#333333", ec="none"))
                ax.add_patch(Circle((vv.x_mm, vv.y_mm), vv.drill_mm / 2, fc="white", ec="none"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(out_path)
