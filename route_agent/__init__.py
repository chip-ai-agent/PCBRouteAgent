"""PCBRouteAgent: Global / Detailed two-role LLM routing on top of the PCBWorld engine.

Package map:

    schema/    data contracts — snapshot, RouteGuidePlan, DetailedSubtask, feedback
    snapshot/  build the structured snapshot + the four Global input views from a live engine
    guide/     schema + semantic validation, corridor compiler, patch application
    loop/      budget accounting and the Detailed -> Global feedback policy router
    exec/      native adapter over ``pcb_world`` (lazy engine), tool gateway, scripted executor, test double
    prompts/   provider-neutral prompt bundles for both roles
    providers  LLM providers (file / OpenAI / Anthropic) for the Global role
    cli/       ``pipeline.py`` (snapshot, validate-guide, compile, execute, global, bench, benchset, score, ...)
    paths      where the PCBWorld checkout, its built engine and the datasets are
"""

from route_agent.paths import ensure_pcbworld_on_path

__version__ = "0.1.0"

ensure_pcbworld_on_path()
