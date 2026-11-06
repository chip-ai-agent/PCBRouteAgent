# Third-party notices

PCBRouteAgent is distributed under the BSD 3-Clause License (see `LICENSE`). It depends on, and in
one file derives from, the following work.

## PCBWorld

- Source: https://github.com/LGAI-Research/PCBWorld (pinned as the submodule `third_party/PCBWorld`, v1.0.1)
- License: BSD 3-Clause, Copyright (c) 2026, LG Management Development Institute Co., Ltd. (LG AI Research)
- Use: PCBRouteAgent imports `pcb_world` (engine wrapper, action primitives, layer mapping, DRC helper)
  and `eval.metrics` (official evaluator) from a PCBWorld checkout. No PCBWorld source file is
  copied into this repository.
- `patches/pcbworld-anthropic-claude5-provider.patch` is a modification of PCBWorld's
  `methods/llm_agent/policy/model_provider.py` and contains context lines of that file. It is
  provided under PCBWorld's BSD 3-Clause License with the copyright notice above.

## PCBWorld-Engine and KiCad

PCBWorld's routing engine (https://github.com/LGAI-Research/PCBWorld-Engine) is a separate program
under GPLv3 that builds on KiCad. PCBRouteAgent calls it only through PCBWorld's Python interface.
Nothing from the engine or from KiCad is included in or distributed with this repository.

## Datasets

The board lists in `configs/board_lists/` name boards of the PCBWorld test sets by path. The
boards themselves are not distributed here; obtain and prepare them as described by PCBWorld.
