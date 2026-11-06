"""Same-layer maze guidance for the Detailed.

A coarse occupancy grid over the neighbourhood of the stage, an A* path from the head to the
point PNS could not reach in a straight attempt, simplified to a few corners that the executor
then commits one ``route_to`` at a time. PNS still does the routing and owns legality; this only
decides where to ask it to go when a straight attempt is stuck on copper that a short detour
avoids (the classic detailed-routing job inside the channel the Global assigned). It is bounded
to the stage bbox grown by a margin: a detour that would need another channel or another layer
is the Global's decision and is reported as evidence instead.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from route_agent.exec.waypoints import clearance_ok
from route_agent.guide.geometry import normalize


@dataclass
class Grid:
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int
    blocked: bytearray

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        i = min(self.nx - 1, max(0, int((x - self.x0) / self.cell)))
        j = min(self.ny - 1, max(0, int((y - self.y0) / self.cell)))
        return i, j

    def centre(self, i: int, j: int) -> list[float]:
        return [round(self.x0 + (i + 0.5) * self.cell, 4), round(self.y0 + (j + 0.5) * self.cell, 4)]

    def free(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[j * self.nx + i]


def build_grid(bbox: list[float], cell: float, obstacles: dict, layer: str, keep: float) -> Grid:
    """Cells whose centre would put copper of half-width+clearance ``keep`` within reach of a
    foreign pad / track / via on ``layer`` are blocked."""
    x1, y1, x2, y2 = normalize(bbox)
    nx = max(2, int(math.ceil((x2 - x1) / cell)))
    ny = max(2, int(math.ceil((y2 - y1) / cell)))
    g = Grid(x1, y1, cell, nx, ny, bytearray(nx * ny))
    for j in range(ny):
        cy = y1 + (j + 0.5) * cell
        for i in range(nx):
            cx = x1 + (i + 0.5) * cell
            if not clearance_ok(cx, cy, keep, obstacles, [layer]):
                g.blocked[j * nx + i] = 1
    return g


_STEPS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def astar(g: Grid, start: tuple[int, int], goal: tuple[int, int], max_expand: int = 20000) -> list[tuple[int, int]] | None:
    """8-connected A* on free cells (a diagonal step needs both orthogonal neighbours free, so
    the path never clips a blocked corner). Start and goal count as free."""
    def h(c):
        dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
    def ok(i, j):
        return (i, j) in (start, goal) or g.free(i, j)
    open_: list = [(h(start), 0.0, start)]
    came: dict = {start: None}
    cost = {start: 0.0}
    n = 0
    while open_:
        _f, c, cur = heapq.heappop(open_)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        if c > cost.get(cur, math.inf):
            continue
        n += 1
        if n > max_expand:
            return None
        for dx, dy, w in _STEPS:
            nb = (cur[0] + dx, cur[1] + dy)
            if not ok(*nb):
                continue
            if dx and dy and not (ok(cur[0] + dx, cur[1]) and ok(cur[0], cur[1] + dy)):
                continue
            nc = c + w
            if nc < cost.get(nb, math.inf):
                cost[nb] = nc
                came[nb] = cur
                heapq.heappush(open_, (nc + h(nb), nc, nb))
    return None


def _visible(g: Grid, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Every cell the segment a-b passes through is free (supercover walk)."""
    (x0, y0), (x1, y1) = a, b
    n = max(abs(x1 - x0), abs(y1 - y0))
    if n == 0:
        return True
    for k in range(n + 1):
        t = k / n
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        for i in {int(math.floor(x)), int(math.ceil(x))}:
            for j in {int(math.floor(y)), int(math.ceil(y))}:
                if (i, j) not in (a, b) and not g.free(i, j):
                    return False
    return True


def simplify(g: Grid, path: list[tuple[int, int]], max_pts: int) -> list[tuple[int, int]]:
    """Greedy line-of-sight string pulling: from each corner jump to the farthest visible cell."""
    out = []
    k = 0
    while k < len(path) - 1:
        far = k + 1
        for m in range(len(path) - 1, k, -1):
            if _visible(g, path[k], path[m]):
                far = m
                break
        out.append(path[far])
        k = far
        if len(out) > max_pts:
            return out[:max_pts]
    return out


def maze_waypoints(bbox: list[float], cell: float, obstacles: dict, layer: str, keep: float,
                   start: list[float], goal: list[float], max_pts: int = 6, max_stretch: float = 3.0) -> list[list[float]] | None:
    """Corners of a same-layer detour from ``start`` to ``goal`` (the goal itself last, exact),
    or None when the grid has no path, or the path is longer than ``max_stretch`` x the straight
    distance (+ 2 mm): a detour that long is a different channel, which is the Global's call.
    A single corner means the straight line is already clear on this grid, so the caller gains
    nothing from it."""
    g = build_grid(bbox, cell, obstacles, layer, keep)
    s, t = g.cell_of(start[0], start[1]), g.cell_of(goal[0], goal[1])
    path = astar(g, s, t)
    if path is None:
        return None
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])) * cell
    straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
    if length > max_stretch * straight + 2.0:
        return None
    corners = simplify(g, path, max_pts)
    pts = [g.centre(i, j) for i, j in corners[:-1]] + [[float(goal[0]), float(goal[1])]]
    return pts
