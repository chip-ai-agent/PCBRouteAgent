"""bbox helpers (x1,y1,x2,y2 in mm)."""
from __future__ import annotations


def normalize(b: list[float]) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in b)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def intersect(a: list[float], b: list[float]) -> list[float] | None:
    a, b = normalize(a), normalize(b)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def contains(b: list[float], x: float, y: float, tol: float = 1e-6) -> bool:
    x1, y1, x2, y2 = normalize(b)
    return x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol


def center(b: list[float]) -> list[float]:
    x1, y1, x2, y2 = normalize(b)
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def area(b: list[float]) -> float:
    x1, y1, x2, y2 = normalize(b)
    return (x2 - x1) * (y2 - y1)


def shrink(b: list[float], d: float) -> list[float] | None:
    x1, y1, x2, y2 = normalize(b)
    if x2 - x1 <= 2 * d or y2 - y1 <= 2 * d:
        return None
    return [x1 + d, y1 + d, x2 - d, y2 - d]


def clamp_point(b: list[float], x: float, y: float) -> list[float]:
    x1, y1, x2, y2 = normalize(b)
    return [min(max(x, x1), x2), min(max(y, y1), y2)]
