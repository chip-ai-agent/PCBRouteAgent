"""Obstacle-aware waypoint candidates for the Detailed role.

``obstacle_aware_candidates`` replaces the blind centre + diamond of ``candidate_points``: it
lays a grid over the (keep-out shrunk) region, drops every point whose copper would land
within clearance of a foreign pad, track or via (as reported by ``inspect_region``), and
ranks the survivors by how well they continue the straight intent ``head -> hint`` and how
close they get to the hint. The region centre keeps the first slot when it is clear, so a
clear region behaves exactly like before. No path search: PNS still does the routing, these
are only the points it is asked to reach.
"""
from __future__ import annotations

import math

from route_agent.guide.geometry import center, normalize, shrink


def _pt_seg_dist(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _pt_rect_dist(px, py, cx, cy, hw, hh) -> float:
    dx = max(abs(px - cx) - hw, 0.0)
    dy = max(abs(py - cy) - hh, 0.0)
    return math.hypot(dx, dy)


def foreign_obstacles(region: dict, net_code: int) -> dict:
    """Split an ``inspect_region`` result into the obstacles of OTHER nets."""
    return {
        "pads": [p for p in region.get("pads", []) if int(p.get("net_code", -1)) != int(net_code)],
        "tracks": [t for t in region.get("tracks", []) if int(t.get("net_code", -1)) != int(net_code)],
        "vias": [v for v in region.get("vias", []) if int(v.get("net_code", -1)) != int(net_code)],
    }


def clearance_ok(x: float, y: float, keep: float, obstacles: dict, layers: list[str]) -> bool:
    """``keep`` = half-width of the copper that will land at (x, y) + the clearance. A pad counts
    on every layer it occupies (thru-hole on all), a track on its layer, a via on all layers."""
    for p in obstacles["pads"]:
        pl = p.get("layers") or []
        if pl and not any(ln in pl for ln in layers):
            continue
        hw, hh = float(p.get("w_mm", 1.0)) / 2, float(p.get("h_mm", 1.0)) / 2
        if _pt_rect_dist(x, y, float(p["x_mm"]), float(p["y_mm"]), hw, hh) < keep:
            return False
    for t in obstacles["tracks"]:
        if t.get("layer") not in layers:
            continue
        x1, y1, x2, y2 = t["seg"]
        if _pt_seg_dist(x, y, x1, y1, x2, y2) < keep + float(t.get("width_mm", 0.2)) / 2:
            return False
    for v in obstacles["vias"]:
        if math.hypot(x - float(v["x_mm"]), y - float(v["y_mm"])) < keep + float(v.get("diameter_mm", 0.6)) / 2:
            return False
    return True


def grid_points(bbox: list[float], keep: float, nx: int = 5, ny: int = 5) -> list[list[float]]:
    inner = shrink(bbox, keep) or normalize(bbox)
    x1, y1, x2, y2 = inner
    xs = [x1 + (x2 - x1) * i / (nx - 1) for i in range(nx)] if nx > 1 else [(x1 + x2) / 2]
    ys = [y1 + (y2 - y1) * j / (ny - 1) for j in range(ny)] if ny > 1 else [(y1 + y2) / 2]
    pts = [[round(x, 4), round(y, 4)] for y in ys for x in xs]
    c = [round(v, 4) for v in center(inner)]
    return [c] + [p for p in pts if p != c]


def obstacle_aware_candidates(bbox: list[float], keep: float, obstacles: dict, layers: list[str],
                              head: list[float], hint: list[float] | None, n: int = 5,
                              nx: int = 5, ny: int = 5) -> list[list[float]]:
    """Clear grid points of ``bbox`` ranked by straight-line intent. Empty when nothing is clear
    (the caller then falls back to the blind candidates)."""
    pts = [p for p in grid_points(bbox, keep, nx, ny) if clearance_ok(p[0], p[1], keep, obstacles, layers)]
    if not pts:
        return []
    cen = [round(v, 4) for v in center(shrink(bbox, keep) or normalize(bbox))]
    hx, hy = float(head[0]), float(head[1])
    tx, ty = (float(hint[0]), float(hint[1])) if hint else (cen[0], cen[1])

    def key(p):
        off_line = _pt_seg_dist(p[0], p[1], hx, hy, tx, ty)       # stay on the intended straight run
        to_hint = math.hypot(p[0] - tx, p[1] - ty)                 # and get on with it
        return (0 if p == cen else 1, round(off_line, 3), round(to_hint, 3))
    pts.sort(key=key)
    return pts[:n]


def _orient(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_cross(a, b, c, d, eps: float = 1e-9) -> bool:
    """Proper crossing of segments a-b and c-d (touching at an endpoint does not count)."""
    o1, o2 = _orient(*a, *b, *c), _orient(*a, *b, *d)
    o3, o4 = _orient(*c, *d, *a), _orient(*c, *d, *b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def segment_hits_rect(x1, y1, x2, y2, rect) -> bool:
    """Segment touches an axis-aligned rect: an endpoint inside, or a crossing of one of its edges."""
    rx1, ry1, rx2, ry2 = normalize(rect)
    if rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2 or rx1 <= x2 <= rx2 and ry1 <= y2 <= ry2:
        return True
    corners = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
    return any(segments_cross((x1, y1), (x2, y2), corners[k], corners[(k + 1) % 4]) for k in range(4))


def expand(bbox: list[float], d: float) -> list[float]:
    x1, y1, x2, y2 = normalize(bbox)
    return [x1 - d, y1 - d, x2 + d, y2 + d]


def ring_points(cx: float, cy: float, radii=(0.7, 1.0, 1.4, 1.9, 2.5), n_dirs: int = 8) -> list[tuple[int, float, list[float]]]:
    """``(radius index, angle, point)`` for rings of candidate sites around (cx, cy)."""
    out = []
    for k, r in enumerate(radii):
        for d in range(n_dirs):
            ang = 2 * math.pi * d / n_dirs
            out.append((k, ang, [round(cx + r * math.cos(ang), 4), round(cy + r * math.sin(ang), 4)]))
    return out


def via_sites_around(head, keep: float, obstacles: dict, layers: list[str], toward=None,
                     radii=(0.7, 1.0, 1.4, 1.9, 2.5), n_dirs: int = 8, n: int = 5) -> list[list[float]]:
    """Where a via would fit next to a head that is boxed in on its layer: ring points around it
    whose via copper keeps clearance from the given obstacles on BOTH layers (a via is a hole
    through both), nearest ring first and, within a ring, the direction closest to ``toward``
    first. ``keep`` = via radius + clearance. The Detailed only REPORTS these (the Global decides
    whether to put a via region there); it never places a via on its own."""
    hx, hy = float(head[0]), float(head[1])
    want = math.atan2(float(toward[1]) - hy, float(toward[0]) - hx) if toward is not None else None

    def turn(a: float) -> float:
        if want is None:
            return 0.0
        d = abs(a - want) % (2 * math.pi)
        return min(d, 2 * math.pi - d)
    pts = [(k, turn(a), p) for k, a, p in ring_points(hx, hy, radii, n_dirs)
           if clearance_ok(p[0], p[1], keep, obstacles, layers)]
    pts.sort(key=lambda t: (t[0], round(t[1], 3)))
    return [p for _k, _t, p in pts[:n]]
