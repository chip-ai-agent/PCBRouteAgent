"""Generic dataclass <-> dict conversion used by every contract in ``route_agent.schema``.

Dataclasses are the in-memory form; JSON is the on-disk / on-the-wire form. ``from_dict``
walks the type hints so nested dataclasses, ``list[X]``, ``dict[str, X]`` and ``Optional[X]``
round-trip without per-class boilerplate. Unknown keys are kept in ``extra`` when the
dataclass declares one, otherwise dropped (forward compatible with newer writers).
"""
from __future__ import annotations

import dataclasses
import json
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if f.name == "extra" and not v:
                continue
            out[f.name] = to_dict(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _unwrap_optional(tp):
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _convert(tp, value):
    if value is None:
        return None
    tp = _unwrap_optional(tp)
    if isinstance(tp, str):          # unresolved forward ref: leave as-is
        return value
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    if origin in (list, tuple):
        (inner,) = typing.get_args(tp) or (Any,)
        return [_convert(inner, v) for v in value]
    if origin is dict:
        _k, inner = typing.get_args(tp) or (str, Any)
        return {k: _convert(inner, v) for k, v in value.items()}
    if tp is float and isinstance(value, (int, float)):
        return float(value)
    return value


def from_dict(cls: type[T], data: dict) -> T:
    if dataclasses.is_dataclass(data):
        return data  # already built
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    extra = {}
    for k, v in data.items():
        if k in fields:
            kwargs[k] = _convert(hints[k], v)
        else:
            extra[k] = v
    if "extra" in fields and extra:
        kwargs.setdefault("extra", {}).update(extra)
    return cls(**kwargs)


def dump_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(obj), indent=2, ensure_ascii=False) + "\n")
    return path


def load_json(path: str | Path) -> Any:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)
