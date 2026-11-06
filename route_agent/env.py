"""Local secrets and model selection for the CLIs, kept out of the repo and the shell rc.

``load_env_file`` reads ``$ROUTEAGENT_ENV``, else ``~/.config/routeagent/env``, else ``./.env`` — ``KEY=VALUE`` or
``export KEY=VALUE`` lines, ``#`` comments, optional quotes — and puts the values into the
process environment. Values from the file WIN over the inherited environment, so a stale key
exported by a shared ``~/.bashrc`` cannot shadow the valid one. Placeholder values (``REPLACE``,
``your-key``, ``<...>``) are skipped so an unedited template changes nothing. Only key NAMES are
ever reported, never values.

Recognised keys: ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``ROUTEAGENT_OPENAI_MODEL``,
``ROUTEAGENT_ANTHROPIC_MODEL`` (provider defaults), plus anything else the user puts there.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATH = Path.home() / ".config" / "routeagent" / "env"
CANDIDATES = (DEFAULT_PATH, Path(".env"))        # first existing one wins; $ROUTEAGENT_ENV overrides both
_PLACEHOLDERS = ("replace", "your-key", "your_key", "<", ">")


def parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        k, v = (part.strip() for part in line.split("=", 1))
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if not k or not v or any(m in v.lower() for m in _PLACEHOLDERS):
            continue
        out[k] = v
    return out


def load_env_file(path: str | os.PathLike | None = None, environ: dict | None = None) -> dict:
    """Load the env file into ``environ`` (default ``os.environ``). Returns ``{path, loaded, keys}``."""
    env = os.environ if environ is None else environ
    explicit = path or env.get("ROUTEAGENT_ENV")
    if explicit:
        p = Path(explicit).expanduser()
    else:
        p = next((c for c in CANDIDATES if c.expanduser().is_file()), DEFAULT_PATH).expanduser()
    if not p.is_file():
        return {"path": str(p), "loaded": False, "keys": []}
    values = parse_env(p.read_text())
    env.update(values)
    return {"path": str(p), "loaded": True, "keys": sorted(values)}
