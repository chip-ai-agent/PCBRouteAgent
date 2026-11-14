#!/bin/bash
# Build an image that runs PCBRouteAgent on KiCad 10.0.7, and the PCBWorld engine for it. The engine comes from
# the KiCad 10 ports of PCBWorld (LGAI-Research/PCBWorld PR #2) and PCBWorld-Engine (PR #1), not merged yet;
# this checkout's third_party/PCBWorld (KiCad 9.0.8) is left as it is.
#   usage: bash docker/build.sh [ROOT]      ROOT: where PCBWorld and its engine build land (default /opt/pcbworld-kicad10)
# The build tree keeps absolute paths, so a run mounts ROOT at the same path, e.g. from this checkout:
#   docker run --rm -v ROOT:ROOT -v "$PWD":/w -w /w -e PYTHONPATH=/w -e PCBWORLD_ROOT=ROOT/PCBWorld \
#     pcbrouteagent-kicad10 python -m route_agent.cli.pipeline bench BOARD.kicad_pcb --out OUT
# The build container is capped (MEM, CPUS): 14g with 12 ninja jobs passed; 8g with 8 jobs was OOM-killed on the
# pcbnew precompiled header.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:-/opt/pcbworld-kicad10}"
IMAGE=pcbrouteagent-kicad10
MEM="${MEM:-14g}"
CPUS="${CPUS:-0-9}"          # cpuset: ninja runs (cpus + 2) jobs, so this also caps parallelism

PCBW_URL=https://github.com/aloyzzz/PCBWorld.git
PCBW_SHA=e2ad599189edaa4b6187d4432cbba04e20e7bf05     # LGAI-Research/PCBWorld PR #2, on v1.0.1
ENG_URL=https://github.com/aloyzzz/PCBWorld-Engine.git
ENG_SHA=d64b46e0805649c3590431bb016471f3ee0de495      # LGAI-Research/PCBWorld-Engine PR #1, on v1.0.1
KICAD_URL=https://gitlab.com/kicad/code/kicad.git
KICAD_TAG=10.0.7
KICAD_SHA=69cafb907ebc09e7722a2a464b1d466fc2e3608d    # the commit build_rl_router.sh enforces

checkout() {  # url sha dir: clone once, then pin
  local url=$1 sha=$2 dir=$3
  [ -d "$dir/.git" ] || git clone -q "$url" "$dir"
  git -C "$dir" fetch -q origin "$sha" 2>/dev/null || git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q --detach "$sha"
}

PCBW="$ROOT/PCBWorld"
ENG="$PCBW/engine"
mkdir -p "$ROOT"
checkout "$PCBW_URL" "$PCBW_SHA" "$PCBW"
checkout "$ENG_URL" "$ENG_SHA" "$ENG"
# The engine reads KiCad from kicad-python/kicad; only that tree is needed, at the pinned tag.
if [ ! -d "$ENG/kicad-python/kicad/.git" ]; then
  mkdir -p "$ENG/kicad-python"
  git clone -q --depth 1 --branch "$KICAD_TAG" "$KICAD_URL" "$ENG/kicad-python/kicad"
fi
[ "$(git -C "$ENG/kicad-python/kicad" rev-parse HEAD)" = "$KICAD_SHA" ] \
  || { echo "KiCad $KICAD_TAG is not $KICAD_SHA" >&2; exit 1; }

docker build -q -t "$IMAGE" "$HERE"
docker run --rm --name pcbworld-kicad10-build --memory="$MEM" --memory-swap="$MEM" --cpuset-cpus="$CPUS" \
  -v "$ROOT:$ROOT" -w "$PCBW" "$IMAGE" \
  bash -c 'BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh'
