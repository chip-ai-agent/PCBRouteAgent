"""``FakeEngine``: an explicit test double with the ``KiCadEngine`` surface the adapter uses.

Routing model: straight segments between committed points; a segment is refused when it
crosses a configured ``blocked`` rectangle on that layer (simulates a walkaround failure)
or the target lands inside a ``via_blocked`` rect for vias. Connectivity: a pad belongs to
the cluster of any committed point within ``snap_mm`` of it, so ``finish`` connects when the
head is near the target pad. Checkpoints deep-copy the state. DRC returns configured
violations. Nothing here is a routing or performance result.
"""
from __future__ import annotations

import copy

from pcb_world.engine.drc import DRC_SEVERITY_ERROR, DRC_SEVERITY_WARNING  # noqa: F401 (re-exported for tests)
from pcb_world.engine.layer_mapping import LayerMapping
import math
from types import SimpleNamespace as NS


def _seg_hits_rect(x1, y1, x2, y2, r) -> bool:
    rx1, ry1, rx2, ry2 = r
    # sample the segment
    n = 32
    for i in range(n + 1):
        t = i / n
        x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
        if rx1 <= x <= rx2 and ry1 <= y <= ry2:
            return True
    return False


class FakeEngine:
    def __init__(self, pads: list[dict], layers: int = 2, netclass: dict | None = None,
                 blocked: dict | None = None, via_blocked: list | None = None, snap_mm: float = 0.3,
                 bbox: list[float] | None = None, drc: list | None = None, shovable: dict | None = None,
                 via_offset: tuple[float, float] = (0.0, 0.0), tracks_block: bool = False) -> None:
        self.layers = layers
        self.pads = [NS(x_mm=p["x"], y_mm=p["y"], width_mm=p.get("w", 1.0), height_mm=p.get("h", 1.0),
                        layer=p.get("layer", -2), net_code=p["net"], net_name=p.get("name", f"net{p['net']}"),
                        pad_name=p["pad"], footprint_ref=p["ref"], pad_type=p.get("type", "thru_hole"),
                        shape=p.get("shape", "circle")) for p in pads]
        self.net_names = {p.net_code: p.net_name for p in self.pads}
        nc = netclass or {}
        self.nc = NS(name=nc.get("name", "Default"), clearance_mm=nc.get("clearance_mm", 0.2),
                     track_width_mm=nc.get("track_width_mm", 0.2), via_diameter_mm=nc.get("via_diameter_mm", 0.6),
                     via_drill_mm=nc.get("via_drill_mm", 0.4), uvia_diameter_mm=0.3, uvia_drill_mm=0.1)
        self.blocked = {int(k): v for k, v in (blocked or {}).items()}   # human layer -> [rects]
        self.shovable = {int(k): v for k, v in (shovable or {}).items()}  # soft rects: block walkaround, yield to shove
        self.via_blocked = via_blocked or []
        self._mode = 2
        self.via_offset = tuple(via_offset)
        self.tracks_block = bool(tracks_block)      # other nets' same-layer tracks are hard obstacles
        self.snap = snap_mm
        self.bbox = bbox or [0, 0, 50, 50]
        self._drc = drc or []
        self.tracks: list[NS] = []
        self.vias: list[NS] = []
        self._ckpts: dict[int, tuple] = {}
        self._next_ckpt = 1
        self._routing = False
        self._head = (0.0, 0.0)
        self._layer = -1
        self._net = -1
        self._via_mode = False
        self._mode = 2
        self._cancel_count = 0
        self.drc_helper = _FakeDRCHelper(self)
        self.layer_map = LayerMapping(layers)      # same object a KiCadEngine exposes
        self.locked: dict[int, bool] = {}

    # --- board queries
    def get_copper_layer_count(self): return self.layers
    def get_pads(self): return list(self.pads)
    def get_tracks(self): return list(self.tracks)
    def get_vias(self): return list(self.vias)
    def get_keepouts(self): return []
    def get_footprints(self): return []
    def get_board_bbox(self):
        x1, y1, x2, y2 = self.bbox
        return NS(x_mm=x1, y_mm=y1, width_mm=x2 - x1, height_mm=y2 - y1)
    def get_board_outline(self):
        x1, y1, x2, y2 = self.bbox
        return [NS(x1_mm=x1, y1_mm=y1, x2_mm=x2, y2_mm=y1), NS(x1_mm=x2, y1_mm=y1, x2_mm=x2, y2_mm=y2),
                NS(x1_mm=x2, y1_mm=y2, x2_mm=x1, y2_mm=y2), NS(x1_mm=x1, y1_mm=y2, x2_mm=x1, y2_mm=y1)]
    def get_net_names(self): return dict(self.net_names)
    def get_netclass_for_net(self, net_code): return self.nc
    def get_design_rules(self):
        return NS(min_clearance_mm=0.0, min_track_width_mm=0.2, min_via_diameter_mm=0.4, copper_edge_clearance_mm=0.0,
                  default_netclass=self.nc, netclasses=[])
    def was_project_loaded_from_file(self): return True
    def get_copper_layer_names(self): return ["F.Cu", "B.Cu"][: self.layers]

    # --- connectivity (committed only)
    def _clusters(self, net):
        pts = [(p.x_mm, p.y_mm, "pad", i) for i, p in enumerate(self.pads) if p.net_code == net]
        items = list(pts)
        for t in self.tracks:
            if t.net_code == net:
                items.append((t.x1_mm, t.y1_mm, "t", id(t)))
                items.append((t.x2_mm, t.y2_mm, "t", id(t)))
        for v in self.vias:
            if v.net_code == net:
                items.append((v.x_mm, v.y_mm, "v", id(v)))
        parent = list(range(len(items)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                same_item = a[2] == b[2] == "t" and a[3] == b[3]
                if same_item or math.hypot(a[0] - b[0], a[1] - b[1]) <= self.snap:
                    union(i, j)
        groups = {}
        for i, it in enumerate(items):
            groups.setdefault(find(i), []).append(it)
        return [g for g in groups.values()]

    def build_connectivity(self): pass
    def get_pad_groups(self):
        out = {}
        for net in set(p.net_code for p in self.pads):
            out[net] = sum(1 for g in self._clusters(net) if any(it[2] == "pad" for it in g))
        return out
    def get_ratsnest(self):
        edges = []
        for net in sorted(set(p.net_code for p in self.pads)):
            pg = [g for g in self._clusters(net) if any(it[2] == "pad" for it in g)]
            for a, b in zip(pg, pg[1:]):
                pa = next(it for it in a if it[2] == "pad"); pb = next(it for it in b if it[2] == "pad")
                edges.append(NS(x1_mm=pa[0], y1_mm=pa[1], x2_mm=pb[0], y2_mm=pb[1], net_code=net, layer1=0, layer2=0))
        return edges
    def get_connected_points(self, x, y, human_layer):
        # mirrors KiCadEngine.get_connected_points: (x_mm, y_mm, human_layer) tuples for the pad /
        # via centres and track ends of the cluster, one per copper layer the item occupies
        for net in set(p.net_code for p in self.pads):
            for g in self._clusters(net):
                if any(math.hypot(it[0] - x, it[1] - y) <= 1e-6 for it in g):
                    out = []
                    for it in g:
                        if it[2] == "pad":
                            p = self.pads[it[3]]
                            lays = [1, self.layers] if p.layer == -2 else ([1] if p.layer == 0 else [self.layers])
                        elif it[2] == "v":
                            lays = [1, self.layers]
                        else:
                            t = next((t for t in self.tracks if id(t) == it[3]), None)
                            lays = [1 if (t is None or t.layer == 0) else self.layers]
                        out.extend((it[0], it[1], L) for L in lays)
                    return out
        return []
    def get_routable_nets(self):
        c = {}
        for p in self.pads: c[p.net_code] = c.get(p.net_code, 0) + 1
        return frozenset(n for n, k in c.items() if k >= 2)
    def get_unrouted_count(self): return len(self.get_ratsnest())

    # --- session
    def get_routing_session_state(self):
        return NS(state_code=3 if self._routing else 0, is_routing=self._routing, is_dragging=False,
                  is_placing_via=self._via_mode, current_layer=self._layer if self._routing else -1,
                  route_head=(*self._head, float(self._layer)), current_net_code=self._net,
                  routing_target=(0.0, 0.0, -1.0))
    def is_routing(self): return self._routing
    def is_dragging(self): return False
    def get_current_layer(self): return self._layer if self._routing else -1
    def get_current_net_code(self): return self._net
    def get_via_count(self): return len(self.vias)
    def set_routing_mode(self, m): self._mode = m
    def set_corner_mode(self, m): pass
    def set_track_width(self, w): self.nc.track_width_mm = w
    def set_via_diameter(self, d): self.nc.via_diameter_mm = d
    def set_via_drill(self, d): self.nc.via_drill_mm = d
    def route_item_radius_mm(self, *, for_via): return (self.nc.via_diameter_mm if for_via else self.nc.track_width_mm) / 2
    def pad_block_reason(self, x, y, *, item_radius_mm, for_via):
        if for_via:
            for r in self.via_blocked:
                if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                    return "via_on_thru_pad"
        return None

    def start_route(self, x, y, layer):
        if self._routing or layer < 1 or layer > self.layers:
            return False
        # must start on copper of some net
        net = None
        for p in self.pads:
            if math.hypot(p.x_mm - x, p.y_mm - y) <= self.snap:
                net = p.net_code
        for t in self.tracks:
            if math.hypot(t.x2_mm - x, t.y2_mm - y) <= self.snap or math.hypot(t.x1_mm - x, t.y1_mm - y) <= self.snap:
                net = t.net_code
        for v in self.vias:
            if math.hypot(v.x_mm - x, v.y_mm - y) <= self.snap:
                net = v.net_code
        if net is None:
            return False
        self._routing, self._head, self._layer, self._net = True, (x, y), layer, net
        return True
    def toggle_via(self): self._via_mode = not self._via_mode
    def fix_route(self, x, y, force_finish=True, reject_if_stuck=None, expected_layer=-1, arrive_tol_mm=0.0, require_via=False):
        if not self._routing:
            return False
        hx, hy = self._head
        rects = list(self.blocked.get(self._layer, []))
        if self._mode != 1:                                     # soft obstacles yield to shove (mode 1) only
            rects += list(self.shovable.get(self._layer, []))
        for r in rects:
            if _seg_hits_rect(hx, hy, x, y, r):
                self._routing = False; self._via_mode = False
                return False
        if self.tracks_block:
            from route_agent.exec.waypoints import segments_cross
            lay = 0 if self._layer == 1 else 2
            for t in self.tracks:
                if t.net_code != self._net and t.layer == lay and segments_cross((hx, hy), (x, y), (t.x1_mm, t.y1_mm), (t.x2_mm, t.y2_mm)):
                    self._routing = False; self._via_mode = False
                    return False
        if self._via_mode:                                       # PNS may land the via within its radius
            x, y = x + self.via_offset[0], y + self.via_offset[1]
        self.tracks.append(NS(x1_mm=hx, y1_mm=hy, x2_mm=x, y2_mm=y, width_mm=self.nc.track_width_mm,
                              layer=0 if self._layer == 1 else 2, net_code=self._net, net_name=self.net_names.get(self._net, ""), uuid=""))
        if self._via_mode:
            self.vias.append(NS(x_mm=x, y_mm=y, diameter_mm=self.nc.via_diameter_mm, drill_mm=self.nc.via_drill_mm,
                                top_layer=0, bottom_layer=2, net_code=self._net, net_name=self.net_names.get(self._net, ""), uuid=""))
            self._via_mode = False
        self._head = (x, y)
        if force_finish:
            self._routing = False
        return True
    def finish(self, max_attempts=5):
        if not self._routing:
            return False
        hx, hy = self._head
        # like PNS Finish(): the nearest pad of the net NOT already joined to the head
        joined = set()
        for g in self._clusters(self._net):
            if any(math.hypot(it[0] - hx, it[1] - hy) <= self.snap for it in g):
                joined = {(it[0], it[1]) for it in g if it[2] == "pad"}
        best = None
        for p in self.pads:
            if p.net_code == self._net and (p.x_mm, p.y_mm) not in joined and self._pad_on_layer(p, self._layer):
                d = math.hypot(p.x_mm - hx, p.y_mm - hy)
                if best is None or d < best[0]:
                    best = (d, p)
        if best is None:
            self._routing = False
            return False
        ok = self.fix_route(best[1].x_mm, best[1].y_mm, force_finish=True)
        self._routing = False
        return ok
    def _pad_on_layer(self, p, human_layer):
        """Pad layer field: -2 = every copper layer; else KiCad id 0 = F.Cu (human 1), 2 = B.Cu (human N)."""
        if p.layer == -2:
            return True
        return (p.layer == 0 and human_layer == 1) or (p.layer == 2 and human_layer == self.layers)
    def cancel_route(self): self._routing = False; self._via_mode = False; self._cancel_count += 1
    def undo_last_segment(self):
        if self.tracks: self.tracks.pop(); return True
        return False
    def flip_posture(self): pass
    def switch_layer(self, layer): self._layer = layer; return True

    # --- edit
    def delete_track_near(self, x1, y1, x2, y2, human_layer, net_code, tol_mm=0.1):
        lay = 0 if human_layer == 1 else 2
        for i, t in enumerate(self.tracks):
            if t.net_code != net_code or t.layer != lay:
                continue
            same = (math.hypot(t.x1_mm - x1, t.y1_mm - y1) <= tol_mm and math.hypot(t.x2_mm - x2, t.y2_mm - y2) <= tol_mm) or \
                   (math.hypot(t.x1_mm - x2, t.y1_mm - y2) <= tol_mm and math.hypot(t.x2_mm - x1, t.y2_mm - y1) <= tol_mm)
            if same:
                del self.tracks[i]
                return True
        return False
    def delete_routing_of_nets(self, nets):
        nets = set(nets); before = len(self.tracks) + len(self.vias)
        self.tracks = [t for t in self.tracks if t.net_code not in nets]
        self.vias = [v for v in self.vias if v.net_code not in nets]
        return before - len(self.tracks) - len(self.vias)
    def lock_net(self, net, locked=True): self.locked[net] = locked; return 1
    def cleanup_tracks(self, **kw):
        return NS(ran=not self._routing, reject_reason="" if not self._routing else "session open", items=[], removed=[], modified=[], changed=False)

    # --- transaction
    def checkpoint(self):
        h = self._next_ckpt; self._next_ckpt += 1
        self._ckpts[h] = (copy.deepcopy(self.tracks), copy.deepcopy(self.vias))
        return h
    def restore(self, h):
        if h not in self._ckpts: return False
        t, v = self._ckpts[h]
        self.tracks, self.vias = copy.deepcopy(t), copy.deepcopy(v)
        self._routing = False
        return True
    def has_checkpoint(self, h): return h in self._ckpts
    def release_checkpoint(self, h): self._ckpts.pop(h, None)
    def reset_checkpoints(self): self._ckpts.clear()
    def run_drc(self, rules_path=""):
        self.drc_helper.update(list(self._drc))
        return list(self._drc)
    def get_drc_violations(self): return list(self._drc)
    def save(self, path, project_output_path=None):
        with open(path, "w") as f:
            f.write(f"(fake_kicad_pcb tracks={len(self.tracks)} vias={len(self.vias)})\n")
    def close(self): pass


class _FakeDRCHelper:
    def __init__(self, eng): self.eng = eng; self.names = None; self._v = []
    def set_target_net_names(self, names): self.names = set(names) if names is not None else None
    def update(self, v):
        self._v = [x for x in v if self.names is None or set(x.net_names) & self.names]
    def get_violations(self): return list(self._v)
    def clear(self): self._v = []


def violation(error_type="clearance", net_names=(), x=0.0, y=0.0, severity=DRC_SEVERITY_ERROR, layer=0, error_code=2):
    """A DRCViolation look-alike; ``severity`` uses KiCad's codes (0x20 error, 0x10 warning) and
    ``error_code`` KiCad's DRCE_* (1 = unconnected items, which is the ratsnest, not a violation)."""
    return NS(error_code=error_code, error_type=error_type, message=error_type, x_mm=x, y_mm=y, layer=layer,
              net_names=list(net_names), severity=severity, item_a="", item_b="")
