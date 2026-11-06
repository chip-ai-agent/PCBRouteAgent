"""Tally recorded model usage of PCBRouteAgent runs (no network).

Walks RUNS_DIR for ``bench_report.json`` files, de-duplicates identical reports, and sums the
Global provider's recorded calls and tokens per run directory. Prices are list prices in
$ per 1M tokens and are easy to edit below.

Usage: python scripts/api_spend.py RUNS_DIR
"""
import collections
import glob
import hashlib
import json
import os
import sys

# model family -> [(max prompt tokens of the tier or None, $ per 1M input, $ per 1M output)]
PRICES = {
    "haiku": [(100_000, 0.10, 0.50), (None, 0.50, 2.50)],
    "sonnet": [(None, 2.0, 10.0)],
    "opus": [(None, 4.0, 20.0)],
}


def family(model: str) -> str:
    m = (model or "").lower()
    return next((f for f in PRICES if f in m), "other")


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    for limit, p_in, p_out in PRICES.get(family(model), []):
        if limit is None or tokens_in <= limit:
            return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out
    return 0.0


def main(root: str) -> None:
    seen = set()
    runs = collections.defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "models": set()})
    for path in glob.glob(os.path.join(root, "**", "bench_report.json"), recursive=True):
        raw = open(path, "rb").read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            usage = json.loads(raw)["execution"].get("global_usage") or []
        except (ValueError, KeyError):
            continue
        run = os.path.relpath(path, root).split(os.sep)[0]
        for u in usage:
            t_in, t_out = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            model = u.get("model") or u.get("provider") or "?"
            r = runs[run]
            r["calls"] += 1
            r["in"] += t_in
            r["out"] += t_out
            r["cost"] += price(model, t_in, t_out)
            r["models"].add(model)
    total = 0.0
    for run, r in sorted(runs.items(), key=lambda kv: -kv[1]["cost"]):
        total += r["cost"]
        print(f"{run:40} {','.join(sorted(r['models'])):22} calls {r['calls']:4}  in {r['in'] / 1e6:7.2f}M  "
              f"out {r['out'] / 1e6:6.3f}M  ${r['cost']:7.2f}")
    print(f"total ${total:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
