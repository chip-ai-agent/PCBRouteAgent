#!/usr/bin/env bash
# Run PCBRouteAgent end to end on PCBWorld test sets: check the setup, route every board with LLM
# replanning, score each saved board with PCBWorld's official evaluator, then print the result
# table and the recorded model usage.
#
#   scripts/run_pcbworld.sh                                   # all four sets, Claude Haiku 5.5
#   scripts/run_pcbworld.sh -m claude-opus-5-5 -s "d3b d3c"   # another model, some sets
#   scripts/run_pcbworld.sh -m none -s d3a                    # without model calls
#   scripts/run_pcbworld.sh -s d3b -- --only 0113_maytal_Maytal   # extra arguments go to benchset
#
# Options: -m MODEL   claude-*, gpt-* or none        (default claude-haiku-5-5)
#          -s "SETS"  any of d2 d3a d3b d3c          (default: all four)
#          -o DIR     output directory               (default runs/<timestamp>-<model>)
#          -n         dry run: print the commands, run nothing
set -euo pipefail

MODEL=claude-haiku-5-5
SETS="d2 d3a d3b d3c"
OUT=""
DRY=0
usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while getopts "m:s:o:nh" opt; do
  case $opt in
    m) MODEL=$OPTARG ;;
    s) SETS=$OPTARG ;;
    o) OUT=$OPTARG ;;
    n) DRY=1 ;;
    h) usage 0 ;;
    *) usage 2 ;;
  esac
done
shift $((OPTIND - 1))

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PYTHON:-python}

case "$MODEL" in
  none)           PROVIDER=(--provider none) ;;
  claude-*)       PROVIDER=(--provider anthropic --model "$MODEL") ;;
  gpt-*|o[0-9]*)  PROVIDER=(--provider openai --model "$MODEL") ;;
  *) echo "unknown model '$MODEL' (use claude-*, gpt-* or none)" >&2; exit 2 ;;
esac
for s in $SETS; do
  [ -f "configs/board_lists/${s}_test.txt" ] || { echo "unknown set '$s' (use d2, d3a, d3b, d3c)" >&2; exit 2; }
done
OUT=${OUT:-runs/$(date +%Y%m%d-%H%M%S)-$MODEL}

# 1. setup check: PCBWorld sources, built engine, datasets, API key for the chosen model
"$PY" - "$MODEL" <<'PYEOF'
import json, subprocess, sys
model = sys.argv[1]
run = subprocess.run([sys.executable, "-m", "route_agent.cli.pipeline", "status"], capture_output=True, text=True)
if "{" not in run.stdout:
    sys.exit("could not run `python -m route_agent.cli.pipeline status`:\n" + run.stderr[-2000:])
st = json.loads(run.stdout[run.stdout.index("{"):])
problems = []
if not st.get("mod:pcb_world"):
    problems.append("PCBWorld sources not found: clone with --recurse-submodules, or set PCBWORLD_ROOT")
if not st.get("engine_so"):
    problems.append(f"PCBWorld engine is not built under {st['pcbworld_root']}/build_rl (see PCBWorld docs/QUICKSTART.md)")
if not st.get("data_root_exists"):
    problems.append(f"datasets not found at {st['data_root']}: set PCBWORLD_DATA_ROOT")
if model.startswith("claude") and not st.get("anthropic_key"):
    problems.append("ANTHROPIC_API_KEY is missing: put it in ~/.config/routeagent/env")
if model.startswith(("gpt", "o1", "o3", "o4")) and not st.get("openai_key"):
    problems.append("OPENAI_API_KEY is missing: put it in ~/.config/routeagent/env")
print(f"PCBWorld: {st['pcbworld_root']} (engine {st.get('engine_version')})   data: {st['data_root']}")
if problems:
    sys.exit("setup is incomplete:\n" + "\n".join("  - " + p for p in problems))
PYEOF

[ "$MODEL" = none ] || echo "model: $MODEL (API calls; use -n to preview)"

# 2. route + official evaluation, one benchset per test set
for s in $SETS; do
  cmd=("$PY" -m route_agent.cli.pipeline benchset --list "configs/board_lists/${s}_test.txt" --out "$OUT/$s" "${PROVIDER[@]}" "$@")
  echo "+ ${cmd[*]}"
  if [ "$DRY" = 0 ]; then
    mkdir -p "$OUT/$s"
    "${cmd[@]}" 2>&1 | grep --line-buffered -E '^\[[0-9]+/[0-9]+\]' | tee "$OUT/$s/progress.log" \
      || echo "benchset reported a failure on $s; see $OUT/$s/summary.json"
  fi
done
[ "$DRY" = 0 ] || exit 0

# 3. results (official evaluator) and recorded model usage
echo
"$PY" scripts/summarize.py "$OUT"
[ "$MODEL" = none ] || { echo; "$PY" scripts/api_spend.py "$OUT"; }
echo
echo "outputs: $OUT/<set>/summary.json, $OUT/<set>/<board>/{routed.kicad_pcb,bench_report.json}"
