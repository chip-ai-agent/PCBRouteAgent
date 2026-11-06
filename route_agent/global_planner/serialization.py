"""Strict JSON at the Global model boundary."""

from __future__ import annotations

import json
import math


class GlobalPlanError(ValueError):
    """A Global input or response does not satisfy the planning contract."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def require_json(value):
    """Reject non-JSON values, including overflowing numbers and non-string keys."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise GlobalPlanError("JSON object keys must be strings")
            require_json(child)
    elif isinstance(value, list):
        for child in value:
            require_json(child)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise GlobalPlanError("non-JSON value")


def parse_plan(text: str) -> dict:
    value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    require_json(value)
    if not isinstance(value, dict):
        raise GlobalPlanError("Global response must be a JSON object")
    return value


def canonical_json(value) -> str:
    require_json(value)
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))
