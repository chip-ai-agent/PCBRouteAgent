"""Print the official results of a run directory as a table.

Reads RUNS_DIR/<set>/summary.json written by ``benchset``. Every number is PCBWorld's official
evaluator on the saved board (``eval.metrics.evaluate_one``), averaged over the scored boards.

Usage: python scripts/summarize.py RUNS_DIR
"""
import glob
import json
import os
import sys

COLUMNS = [("boards", "boards", "{:d}"), ("CP", "CP", "{:.3f}"), ("Rout", "Rout.", "{:.3f}"), ("Pot", "Pot.", "{:.2f}"),
           ("DRV", "DRV", "{:.2f}"), ("WL_mm", "WL mm", "{:.1f}"), ("Via", "Via", "{:.2f}"), ("Time_s", "s/board", "{:.1f}"),
           ("global_calls", "LLM calls", "{:d}"), ("global_prompt_tokens", "tokens in", "{:,d}"),
           ("global_completion_tokens", "tokens out", "{:,d}")]


def cell(value, fmt: str) -> str:
    if value is None:
        return "-"
    try:
        return fmt.format(int(value) if fmt.endswith("d}") else float(value))
    except (TypeError, ValueError):
        return str(value)


def main(root: str) -> int:
    paths = sorted(glob.glob(os.path.join(root, "*", "summary.json")))
    if not paths:
        print(f"no <set>/summary.json under {root}")
        return 1
    print("| set | " + " | ".join(title for _key, title, _fmt in COLUMNS) + " |")
    print("|---|" + "---:|" * len(COLUMNS))
    for path in paths:
        agg = json.load(open(path)).get("aggregate") or {}
        name = os.path.basename(os.path.dirname(path))
        if agg.get("scored") not in (None, agg.get("boards")):
            name += f" ({agg['scored']} scored)"
        print(f"| {name} | " + " | ".join(cell(agg.get(key), fmt) for key, _title, fmt in COLUMNS) + " |")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
