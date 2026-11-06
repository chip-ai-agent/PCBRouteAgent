"""Where PCBWorld lives.

PCBRouteAgent is a separate package that drives PCBWorld's KiCad engine. From a PCBWorld checkout
it needs the Python packages (``pcb_world``, ``eval``), the built engine (``build_rl/``) and,
for benchmark runs, the prepared datasets.

The checkout is resolved in this order: ``$PCBWORLD_ROOT``; ``third_party/PCBWorld`` next to
this package (the pinned submodule); the directory containing this package (PCBRouteAgent placed
inside a PCBWorld tree). Datasets come from ``$PCBWORLD_DATA_ROOT`` or
``<checkout>/var/datasets`` (PCBWorld's default local dataset directory).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def pcbworld_root() -> Path:
    env = os.environ.get("PCBWORLD_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    sub = PACKAGE_ROOT / "third_party" / "PCBWorld"
    if (sub / "pcb_world").is_dir():
        return sub
    return PACKAGE_ROOT


def data_root() -> Path:
    env = os.environ.get("PCBWORLD_DATA_ROOT")
    return Path(env).expanduser() if env else pcbworld_root() / "var" / "datasets"


def ensure_pcbworld_on_path() -> Path:
    """Append the PCBWorld checkout and its built engine module to ``sys.path`` (idempotent).

    Appended, not prepended: an environment that already provides ``pcb_world`` keeps its own.
    """
    root = pcbworld_root()
    for p in (root, root / "build_rl" / "pcbnew" / "python" / "rl"):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.append(s)
    return root
