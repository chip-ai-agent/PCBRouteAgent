"""Post-routing joint repair: the two segments around a non-45-degree joint are re-routed as ONE PNS
move, so the corner the Detailed left between two committed hops becomes a 45/90-degree one.

Why: PNS (corner mode MITERED_45) keeps every corner inside one ``route_to`` at 45 degrees, but the
joint between two hops is unconstrained — a hop that folds back after a walkaround waypoint leaves a
45-degree hairpin. Measured on the final D3-B / D3-C boards: 8.8 / 20.9 flagged joints per board on
the work-set nets, of which 2.9 / 4.3 acute (< 90), against 1.4 / 1.9 acute in the designers' own
routing of the same nets.

The joint test mirrors the official evaluator (``eval.metrics._check_routing_angles``): same-net,
same-layer joints of exactly two segments that do not sit on a pad or via, flagged when the angle is
not within 0.5 degrees of 135 or 180. Each repair runs under a checkpoint and is kept only if the net's
pad groups are unchanged, the scoped DRC error count did not grow, the flagged joints of the net
decreased without a new acute one, and the net's copper did not grow by more than a small margin.
"""
from __future__ import annotations

import math
from collections import defaultdict

ALLOWED_45 = (135.0, 180.0)
ANGLE_TOL_DEG = 0.5
_R = 6


def _key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), _R), round(float(y), _R))


def net_tracks(eng, net_code: int) -> list:
    return [t for t in eng.get_tracks() if int(t.net_code) == int(net_code)]


def net_length(eng, net_code: int) -> float:
    return sum(math.hypot(t.x2_mm - t.x1_mm, t.y2_mm - t.y1_mm) for t in net_tracks(eng, net_code))


def find_joints(eng, net_code: int, allowed=ALLOWED_45, tol_deg: float = ANGLE_TOL_DEG) -> list[dict]:
    """Flagged joints of one net: ``{x, y, layer (board id), angle, a, b}`` with ``a`` / ``b`` the two
    track records meeting there. Pad anchors and via centres are never joints (evaluator semantics)."""
    exclude: set[tuple[int, tuple[float, float]]] = set()
    for pad in eng.get_pads():
        if int(pad.net_code) != int(net_code):
            continue
        exclude.add((int(pad.layer), _key(pad.x_mm, pad.y_mm)))
    for via in eng.get_vias():
        if int(via.net_code) != int(net_code):
            continue
        lo, hi = sorted((int(via.top_layer), int(via.bottom_layer)))
        for ly in range(lo, hi + 1):
            exclude.add((ly, _key(via.x_mm, via.y_mm)))
    joints: dict = defaultdict(list)
    for t in net_tracks(eng, net_code):
        joints[(int(t.layer), _key(t.x1_mm, t.y1_mm))].append((t, (t.x2_mm, t.y2_mm)))
        joints[(int(t.layer), _key(t.x2_mm, t.y2_mm))].append((t, (t.x1_mm, t.y1_mm)))
    out = []
    for (layer, p), nb in joints.items():
        if len(nb) != 2 or (layer, p) in exclude or (-2, p) in exclude:
            continue
        (ta, fa), (tb, fb) = nb
        v1 = (fa[0] - p[0], fa[1] - p[1]); v2 = (fb[0] - p[0], fb[1] - p[1])
        n1 = math.hypot(*v1); n2 = math.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))))
        if any(abs(ang - a) <= tol_deg for a in allowed):
            continue
        out.append({"x": p[0], "y": p[1], "layer": layer, "angle": round(ang, 2), "a": ta, "b": tb, "far_a": fa, "far_b": fb})
    out.sort(key=lambda j: j["angle"])                     # sharpest first
    return out


def _delete(adapter, t) -> bool:
    name = adapter.layers.name_from_board(int(t.layer))
    return bool(adapter.e.delete_track_near(float(t.x1_mm), float(t.y1_mm), float(t.x2_mm), float(t.y2_mm),
                                            adapter.layers.human(name), int(t.net_code)))


def _other_segment(eng, net_code: int, layer: int, p, exclude) -> object | None:
    """The one other track of this net/layer ending at ``p`` (None when 0 or 2+)."""
    k = _key(*p); found = []
    ex = (_key(exclude.x1_mm, exclude.y1_mm), _key(exclude.x2_mm, exclude.y2_mm))
    for t in net_tracks(eng, net_code):
        if int(t.layer) != int(layer) or (_key(t.x1_mm, t.y1_mm), _key(t.x2_mm, t.y2_mm)) == ex:
            continue
        if _key(t.x1_mm, t.y1_mm) == k or _key(t.x2_mm, t.y2_mm) == k:
            found.append(t)
    return found[0] if len(found) == 1 else None


def _far(t, p):
    return (t.x2_mm, t.y2_mm) if _key(t.x1_mm, t.y1_mm) == _key(*p) else (t.x1_mm, t.y1_mm)


def _anchor(eng, net_code: int, layer: int, p) -> bool:
    k = _key(*p)
    for pad in eng.get_pads():
        if int(pad.net_code) == int(net_code) and _key(pad.x_mm, pad.y_mm) == k and int(pad.layer) in (int(layer), -2):
            return True
    for via in eng.get_vias():
        if int(via.net_code) == int(net_code) and _key(via.x_mm, via.y_mm) == k:
            return True
    return False


def _route_points(adapter, net_code: int, layer_name: str, pts, mode: int) -> str:
    """start_route at pts[0], one route_to per following point (each closes the session)."""
    if not adapter.select_net(int(net_code)).get("ok"):
        return "select"
    for a, b in zip(pts, pts[1:]):
        if not adapter.start_route(a[0], a[1], layer_name).get("ok"):
            return "start"
        r = adapter.route_to(b[0], b[1], mode=mode)
        if not r.get("ok"):
            return "route:" + str(r.get("native_reason") or r.get("reason") or "?")
    return ""


def repair_net(adapter, net_code: int, net_name: str | None = None, *, mode: int = 2, max_attempts: int = 24,
               length_slack_mm: float = 3.0, length_slack_frac: float = 0.15) -> dict:
    """Repair the flagged joints of one net, sharpest first. Several strategies per joint: the two
    segments re-routed as one move (A->B, then B->A), as an L through either corner candidate (a 90
    degree corner replaces a hairpin), the same with the shove mode, and finally a window one segment
    wider on each side. Acceptance: pad groups unchanged, scoped DRC errors not grown, copper not
    longer than the slack, no new acute joint, and either fewer flagged joints or (for an acute joint)
    fewer acute joints. Returns counts, the failure reasons and the joints left."""
    eng = adapter.e
    names = [net_name] if net_name else None
    js0 = find_joints(eng, net_code)
    stats = {"net_code": int(net_code), "before": len(js0), "repaired": 0, "failed": 0, "after": None,
             "acute_before": sum(1 for j in js0 if j["angle"] < 89.5), "reasons": {}}
    tried: set[tuple] = set()
    for _ in range(max_attempts):
        joints = [j for j in find_joints(eng, net_code) if (j["layer"], _key(j["x"], j["y"])) not in tried]
        if not joints:
            break
        j = joints[0]
        tried.add((j["layer"], _key(j["x"], j["y"])))
        acute = j["angle"] < 89.5
        layer_name = adapter.layers.name_from_board(int(j["layer"]))
        groups0 = int(adapter.inspect_net(int(net_code)).get("pad_groups", 0))
        drc0 = int(adapter.run_drc(names).get("errors", 0))
        len0 = net_length(eng, net_code)
        flagged0 = len(find_joints(eng, net_code)); acute0 = sum(1 for q in find_joints(eng, net_code) if q["angle"] < 89.5)
        A, B = j["far_a"], j["far_b"]
        # strategies: (segments to delete, points to route, mode)
        plans = []
        if j["angle"] < 10.0:
            # overshoot: both segments leave the joint along one line (one lies on the other); drop the
            # longer one and close the short gap N -> F, the stub left at the joint falls to the cleanup
            la = math.hypot(A[0] - j["x"], A[1] - j["y"]); lb = math.hypot(B[0] - j["x"], B[1] - j["y"])
            longer, shorter = (j["a"], j["b"]) if la >= lb else (j["b"], j["a"])
            N, F = (B, A) if la >= lb else (A, B)
            plans += [("trim", [longer], [N, F], mode), ("trim-rev", [longer], [F, N], mode), ("trim-both", [j["a"], j["b"]], [N, F], mode)]
        rounding = []
        if acute:
            # round the hairpin: leave the apex on both legs at distance d and go round the outside of
            # the apex (A -> Q1 -> T1 -> T2 -> Q2 -> B): corners 180-theta/2, 90, 90, 180-theta/2, none acute
            la = math.hypot(A[0] - j["x"], A[1] - j["y"]); lb = math.hypot(B[0] - j["x"], B[1] - j["y"])
            if la > 0.3 and lb > 0.3:
                ua = ((A[0] - j["x"]) / la, (A[1] - j["y"]) / la); ub = ((B[0] - j["x"]) / lb, (B[1] - j["y"]) / lb)
                mx, my = ua[0] + ub[0], ua[1] + ub[1]; mn = math.hypot(mx, my)
                if mn > 1e-6:
                    m = (mx / mn, my / mn)
                    for d in (min(la, lb, 1.5) * 0.6, min(la, lb, 1.0) * 0.4):
                        Q1 = (j["x"] + ua[0] * d, j["y"] + ua[1] * d); Q2 = (j["x"] + ub[0] * d, j["y"] + ub[1] * d)
                        T1 = (Q1[0] - m[0] * d, Q1[1] - m[1] * d); T2 = (Q2[0] - m[0] * d, Q2[1] - m[1] * d)
                        rounding.append((f"round{d:.2f}", [j["a"], j["b"]], [A, Q1, T1, T2, Q2, B], mode))
        plans += [("direct", [j["a"], j["b"]], [A, B], mode), ("reverse", [j["a"], j["b"]], [B, A], mode),
                 ("L1", [j["a"], j["b"]], [A, (A[0], B[1]), B], mode), ("L2", [j["a"], j["b"]], [A, (B[0], A[1]), B], mode),
                 ("shove", [j["a"], j["b"]], [A, B], 1)]
        na = _other_segment(eng, net_code, j["layer"], A, j["a"]) if not _anchor(eng, net_code, j["layer"], A) else None
        nb = _other_segment(eng, net_code, j["layer"], B, j["b"]) if not _anchor(eng, net_code, j["layer"], B) else None
        if na is not None or nb is not None:
            A2 = _far(na, A) if na is not None else A
            B2 = _far(nb, B) if nb is not None else B
            segs = [j["a"], j["b"]] + ([na] if na is not None else []) + ([nb] if nb is not None else [])
            plans.append(("wide", segs, [A2, B2], mode))
            plans.append(("wide-L", segs, [A2, (A2[0], B2[1]), B2], mode))
        plans += rounding                       # last resort: more corners, none acute
        why_all = []
        done = False
        for name, segs, pts, m in plans:
            ck = adapter.checkpoint(); handle = ck.get("handle")
            if handle is None:
                why_all.append(name + ":checkpoint"); break
            why = ""
            keys0 = {(_key(t.x1_mm, t.y1_mm), _key(t.x2_mm, t.y2_mm), int(t.layer)) for t in net_tracks(eng, net_code)}
            if not all(_delete(adapter, t) for t in segs):
                why = "delete"
            else:
                why = _route_points(adapter, net_code, layer_name, pts, m)
                if why == "select" and int(adapter.inspect_net(int(net_code)).get("pad_groups", 0)) == groups0:
                    why = ""                      # nothing left to route: the removed copper was redundant
                elif not why:
                    # the new copper must end exactly on the far ends: PNS may stop within its arrive
                    # tolerance, which the engine's connectivity accepts but KiCad's cleanup treats as a
                    # dangling end and deletes the whole chain afterwards (measured: a net lost 16 tracks)
                    ends = set()
                    for t in net_tracks(eng, net_code):
                        if ((_key(t.x1_mm, t.y1_mm), _key(t.x2_mm, t.y2_mm), int(t.layer))) not in keys0:
                            ends.add(_key(t.x1_mm, t.y1_mm)); ends.add(_key(t.x2_mm, t.y2_mm))
                    for q in (pts[0], pts[-1]):
                        if not any(math.hypot(e[0] - q[0], e[1] - q[1]) <= 1e-3 for e in ends):
                            why = "loose_end"
                if not why:
                    # the cleanup the run ends with, applied now to this net so the checks see the final state
                    adapter.cleanup(net_codes=[int(net_code)], dry_run=False, finalize=True)
            if not why:
                groups1 = int(adapter.inspect_net(int(net_code)).get("pad_groups", 0))
                drc1 = int(adapter.run_drc(names).get("errors", 0))
                len1 = net_length(eng, net_code)
                now = find_joints(eng, net_code)
                flagged1 = len(now); acute1 = sum(1 for q in now if q["angle"] < 89.5)
                if groups1 != groups0: why = f"connectivity {groups0}->{groups1}"
                elif drc1 > drc0: why = "drc"
                elif acute1 > acute0: why = "new_acute"
                elif len1 > len0 + max(length_slack_mm, length_slack_frac * len0): why = "too_long"
                elif not (flagged1 < flagged0 or (acute and acute1 < acute0)): why = "joints_same"
            if not why:
                adapter.release_checkpoint(handle)
                stats["repaired"] += 1
                stats.setdefault("strategies", {})[name] = stats.get("strategies", {}).get(name, 0) + 1
                done = True
                break
            adapter.cancel()
            adapter.rollback(handle)
            adapter.release_checkpoint(handle)
            why_all.append(f"{name}:{why}")
            if why.startswith("connectivity") and name in ("direct", "trim"):
                why_all[-1] += " new=" + ";".join(f"({t.x1_mm:.3f},{t.y1_mm:.3f})-({t.x2_mm:.3f},{t.y2_mm:.3f})" for t in net_tracks(eng, net_code)
                                                   if min(math.hypot(t.x1_mm - j["x"], t.y1_mm - j["y"]), math.hypot(t.x2_mm - j["x"], t.y2_mm - j["y"])) < 7.0)
        if not done:
            stats["failed"] += 1
            key = why_all[0].split(":", 1)[1].split(" ")[0] if why_all else "?"
            stats["reasons"][key] = stats["reasons"].get(key, 0) + 1
            la = math.hypot(j["a"].x2_mm - j["a"].x1_mm, j["a"].y2_mm - j["a"].y1_mm); lb = math.hypot(j["b"].x2_mm - j["b"].x1_mm, j["b"].y2_mm - j["b"].y1_mm)
            stats.setdefault("detail", []).append({"angle": j["angle"], "acute": acute, "tried": why_all, "layer": j["layer"],
                                                   "xy": [round(j["x"], 3), round(j["y"], 3)], "len_a": round(la, 3), "len_b": round(lb, 3)})
    adapter.end_net()
    left = find_joints(eng, net_code)
    stats["after"] = len(left); stats["acute_after"] = sum(1 for j in left if j["angle"] < 89.5)
    stats["acute_left"] = [{"x": j["x"], "y": j["y"], "layer": j["layer"], "angle": j["angle"]} for j in left if j["angle"] < 89.5]
    return stats


def repair_joints(adapter, net_codes: list[int], net_names: dict[int, str] | None = None, **kw) -> dict:
    """Repair every net of the work set; a per-net and total summary (serialisable)."""
    per = [repair_net(adapter, int(c), (net_names or {}).get(int(c)), **kw) for c in net_codes]
    tot = {k: sum(int(p[k] or 0) for p in per) for k in ("before", "repaired", "failed", "after", "acute_before", "acute_after")}
    return {"ok": True, **tot, "nets": per}


def guarded_repair(adapter, net_codes: list[int], net_names: dict[int, str] | None = None, *, final_cleanup: bool = True, **kw) -> dict:
    """``repair_joints`` under one checkpoint for the whole pass: the pad groups of every work-set
    net are read before, and again after the pass plus the cleanup the run ends with; if any net
    has more groups than before the whole pass is rolled back (and the cleanup redone), so the pass
    can never cost connectivity whatever the per-repair checks missed."""
    nets = [int(c) for c in net_codes]
    groups0 = {n: int(adapter.inspect_net(n).get("pad_groups", 0)) for n in nets}
    ck = adapter.checkpoint()
    handle = ck.get("handle")
    out = repair_joints(adapter, nets, net_names, **kw)
    if final_cleanup:
        adapter.cleanup(net_codes=nets, dry_run=False, finalize=True)
    groups1 = {n: int(adapter.inspect_net(n).get("pad_groups", 0)) for n in nets}
    worse = sorted(n for n in nets if groups1[n] > groups0[n])
    out["pass_check"] = {"nets_worse": worse, "reverted": False}
    if worse and handle is not None:
        adapter.cancel()
        adapter.rollback(handle)
        out["pass_check"]["reverted"] = True
        out["after"] = out["before"]; out["acute_after"] = out["acute_before"]
    if handle is not None:
        adapter.release_checkpoint(handle)
    return out


def guarded_repair_passes(adapter, net_codes: list[int], net_names: dict[int, str] | None = None,
                          *, max_passes: int = 5, **kw) -> dict:
    """Retry remaining acute joints for up to five guarded passes.

    A joint can be blocked during the first pass but become routable after another joint is
    repaired and cleaned up. Each pass retains the same checkpoint, connectivity and DRC guards;
    stop early only when no work-set acute joints remain.
    """
    if max_passes < 1:
        raise ValueError("max_passes must be positive")
    passes = []
    for _ in range(max_passes):
        rep = guarded_repair(adapter, net_codes, net_names, **kw)
        passes.append(rep)
        if rep["acute_after"] == 0:
            break
    first, last = passes[0], passes[-1]
    return {**last, "before": first["before"], "acute_before": first["acute_before"],
            "repaired": sum(p["repaired"] for p in passes),
            "failed": sum(p["failed"] for p in passes), "passes": passes}
