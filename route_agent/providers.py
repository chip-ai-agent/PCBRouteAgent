"""Global-role providers. ``FileGlobalProvider`` replays a guide from disk (the design example);
``OpenAIGlobalProvider`` / ``AnthropicGlobalProvider`` send the multimodal bundle and parse
JSON. Each returns ``(plan_dict, usage)`` and records tokens + wall time separately from
the executor budget. The Detailed LLM loop (``LLMDetailed``) drives the tool gateway with
function calling; it is optional — the scripted executor is the default Detailed role.
"""
from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from route_agent.exec.gateway import openai_tools
from route_agent.global_planner.serialization import parse_plan


def _strip_fences(text: str) -> str:
    """Extract one fenced JSON object, including when the model adds prose around it.

    Ambiguous replies with multiple code blocks still fail strict JSON parsing.
    """
    t = text.strip()
    blocks = re.findall(r"^```(?:json)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*$", t,
                        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if len(blocks) == 1:
        return blocks[0].strip()
    return t.strip()


def _extract_json(text: str) -> dict:
    """Strict: a bare JSON object only (shared Global boundary, see global_planner.serialization)."""
    return parse_plan(text)


def _keep_raw(text: str) -> str:
    """Save a model response that could not be used, for inspection; returns the path."""
    d = Path(os.environ.get("ROUTEAGENT_DEBUG_DIR") or tempfile.gettempdir())
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"routeagent_global_raw_{int(time.time())}.txt"
    p.write_text(text)
    return str(p)


def _data_uri(path: str) -> str:
    b = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def _image_base64(image: dict) -> str:
    # GlobalPlanner supplies the verified bytes; legacy bundles can still use paths.
    if "data_base64" in image:
        return image["data_base64"]
    return base64.b64encode(Path(image["path"]).read_bytes()).decode()


class FileGlobalProvider:
    def __init__(self, guide_path: str | None, patch_path: str | None = None) -> None:
        self.guide_path = guide_path
        self.patch_path = patch_path

    def plan(self, bundle: dict) -> tuple[dict, dict]:
        if not self.guide_path:
            raise ValueError("FileGlobalProvider.plan: no guide_path to replay")
        return _extract_json(Path(self.guide_path).read_text()), {"provider": "file", "tokens": 0, "wall_s": 0.0}

    def replan(self, bundle: dict) -> tuple[dict, dict]:
        if self.patch_path:
            return _extract_json(Path(self.patch_path).read_text()), {"provider": "file", "tokens": 0, "wall_s": 0.0}
        return {}, {"provider": "file", "tokens": 0, "wall_s": 0.0}


class OpenAIGlobalProvider:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, max_tokens: int | None = None, base_url: str | None = None) -> None:
        self.model, self.temperature, self.base_url = model, temperature, base_url
        self.max_tokens = int(max_tokens or os.environ.get("ROUTEAGENT_MAX_TOKENS", 16000))

    def _call(self, bundle: dict) -> tuple[dict, dict]:
        from openai import OpenAI
        kw = {"base_url": self.base_url} if self.base_url else {}
        client = OpenAI(**kw)
        content: list[dict[str, Any]] = [{"type": "text", "text": bundle["user"]}]
        for im in bundle.get("images", []):
            content.append({"type": "text", "text": f"[image {im['view_id']}]"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + _image_base64(im), "detail": "high"}})
        reasoning = any(self.model.startswith(p) for p in ("o1", "o3", "o4", "gpt-5"))
        msgs = [{"role": "developer" if reasoning else "system", "content": bundle["system"]},
                {"role": "user", "content": content}]
        extra = {} if reasoning else {"temperature": self.temperature}
        tok = {"max_completion_tokens": self.max_tokens} if reasoning else {"max_tokens": self.max_tokens}
        t0 = time.time()
        resp = client.chat.completions.create(model=self.model, messages=msgs, response_format={"type": "json_object"}, **extra, **tok)
        text = resp.choices[0].message.content or ""
        usage = {"provider": "openai", "model": self.model, "prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens, "wall_s": round(time.time() - t0, 2), "raw": text}
        return _extract_json(text), usage

    plan = _call
    replan = _call


class AnthropicGlobalProvider:
    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int | None = None, client=None,
                 temperature: float | None = None, effort: str | None = None) -> None:
        # a whole-board plan (tens of nets, ~100 corridors) runs well past 8k tokens
        self.model = model
        self.max_tokens = int(max_tokens or os.environ.get("ROUTEAGENT_MAX_TOKENS", 32000))
        # sampling temperature: NOT sent by default. Claude Haiku 5.5 (and the other current models)
        # reject a non-default value with 400 "`temperature` is deprecated for this model", so every
        # run samples at the API default and run-to-run spread is inherent; repeat runs and report
        # mean / spread instead. ROUTEAGENT_TEMPERATURE (or the argument) sends a value anyway, for
        # older models that still accept one.
        env_t = os.environ.get("ROUTEAGENT_TEMPERATURE")
        self.temperature = temperature if temperature is not None else (float(env_t) if env_t not in (None, "") else None)
        self.effort = effort or os.environ.get("ROUTEAGENT_ANTHROPIC_EFFORT") or None
        if self.effort not in (None, "low", "medium", "high", "xhigh", "max"):
            raise ValueError(f"Invalid Anthropic effort: {self.effort}")
        self.client = client

    def _call(self, bundle: dict) -> tuple[dict, dict]:
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()
        client = self.client
        content: list[dict[str, Any]] = []
        for im in bundle.get("images", []):
            content.append({"type": "text", "text": f"[image {im['view_id']}]"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                         "data": _image_base64(im)}})
        content.append({"type": "text", "text": bundle["user"]})
        t0 = time.time()
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=bundle["system"],
                  messages=[{"role": "user", "content": content}])
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.effort is not None:
            kw["output_config"] = {"effort": self.effort}
        if self.max_tokens > 8192 and hasattr(client.messages, "stream"):
            # the SDK refuses long non-streaming requests; stream and keep the final message
            with client.messages.stream(**kw) as stream:
                resp = stream.get_final_message()
        else:
            resp = client.messages.create(**kw)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = {"provider": "anthropic", "model": self.model, "temperature": self.temperature, "effort": self.effort,
                 "prompt_tokens": resp.usage.input_tokens,
                 "completion_tokens": resp.usage.output_tokens, "wall_s": round(time.time() - t0, 2), "raw": text,
                 "stop_reason": getattr(resp, "stop_reason", None)}
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise ValueError(f"Global response truncated at max_tokens={self.max_tokens} "
                             f"({resp.usage.output_tokens} output tokens); raise ROUTEAGENT_MAX_TOKENS; raw kept at {_keep_raw(text)}")
        try:
            return _extract_json(_strip_fences(text)), usage      # Claude tends to fence; the JSON itself stays strict
        except ValueError as ex:
            raise ValueError(f"Global response is not a JSON object ({type(ex).__name__}: {ex}); "
                             f"stop_reason={getattr(resp, 'stop_reason', None)}, raw kept at {_keep_raw(text)}") from ex

    plan = _call
    replan = _call


def make_global_provider(name: str, **kw):
    """``file`` replays ``guide_path`` for plan() and ``patch_path`` for replan() (either may be
    omitted when that call is never made); ``openai`` / ``anthropic`` call the model named by
    ``model`` or the provider default."""
    if name == "file":
        guide, patch = kw.get("guide_path"), kw.get("patch_path")
        if not guide and not patch:
            raise ValueError("provider 'file' needs a guide to replay (--guide GUIDE) and/or a patch (--patch PATCH)")
        return FileGlobalProvider(guide, patch)
    if name == "openai":
        return OpenAIGlobalProvider(model=kw.get("model") or os.environ.get("ROUTEAGENT_OPENAI_MODEL", "gpt-4o"))
    if name == "anthropic":
        return AnthropicGlobalProvider(model=kw.get("model") or os.environ.get("ROUTEAGENT_ANTHROPIC_MODEL", "claude-opus-5-5"))
    raise ValueError(f"unknown provider {name!r} (expected file|openai|anthropic)")


class LLMDetailed:
    """OpenAI function-calling Detailed role over the ToolGateway (optional).

    Same ``run_task`` contract as ``ScriptedDetailed`` — it returns a ``TaskResult`` whose
    ``feedback`` is a FeedbackPacket — so the Orchestrator can drive either. Tool errors
    (unknown tool, bad arguments, refused preconditions) are fed back to the model as
    structured tool results, never raised; a budget refusal raises ``BudgetExhausted`` like
    the scripted executor. ``client`` may be injected (tests, other OpenAI-compatible APIs).
    """

    ROUTE_TOOLS = ("start_route", "route_to", "route_via", "finish")

    def __init__(self, gateway, bundle_fn, snapshot_id: str = "", guide_id: str = "", model: str = "gpt-4o",
                 max_rounds: int = 40, client=None) -> None:
        from route_agent.exec.executor import ScriptedDetailed
        self.gw, self.bundle_fn, self.model, self.max_rounds = gateway, bundle_fn, model, max_rounds
        self.client = client
        self.transcripts: list[dict] = []
        self._helper = ScriptedDetailed(gateway, snapshot_id, guide_id)   # packet / deviation builders

    def _client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()
        return self.client

    def _tool_call(self, name: str, raw_args: str) -> tuple[dict, dict]:
        from route_agent.exec.gateway import ToolError
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as ex:
            return {}, {"ok": False, "reason": f"bad JSON arguments: {ex}"}
        try:
            return args, self.gw.call(name, **args)
        except ToolError as ex:
            return args, {"ok": False, "reason": str(ex), "tools": self.gw.implemented()}
        except TypeError as ex:
            return args, {"ok": False, "reason": f"bad arguments for {name}: {ex}"}

    def run_task(self, task, budget):
        from route_agent.exec.executor import StepRecord, TaskResult
        from route_agent.loop.budget import BudgetExhausted
        from route_agent.schema.feedback import Attempt
        gw = self.gw
        calls0 = gw.adapter.log.calls
        if not task.execution.get("supported", True):
            return self._helper.run_task(task, budget)
        bundle = self.bundle_fn(task)
        msgs = [{"role": "system", "content": bundle["system"]}, {"role": "user", "content": bundle["user"]}]
        tools = openai_tools()
        steps: list[StepRecord] = []
        attempts: list[Attempt] = []
        final, cap_reason, rounds = None, "", 0
        for rounds in range(1, self.max_rounds + 1):
            resp = self._client().chat.completions.create(model=self.model, messages=msgs, tools=tools, temperature=0)
            m = resp.choices[0].message
            msgs.append(m)
            if not m.tool_calls:
                final = m.content
                break
            for tc in m.tool_calls:
                name = tc.function.name
                c0 = gw.adapter.log.calls
                args, out = self._tool_call(name, tc.function.arguments)
                if gw.exhausted:
                    raise BudgetExhausted(gw.exhausted)
                ok = bool(out.get("ok", True))
                reason = str(out.get("native_reason") or out.get("reason") or "")
                target = [float(args["x"]), float(args["y"])] if "x" in args and "y" in args else []
                steps.append(StepRecord(stage="", kind=name, target=target, ok=ok, reason=reason, calls=gw.adapter.log.calls - c0))
                if name in self.ROUTE_TOOLS:
                    attempts.append(Attempt(kind=name, target=[round(v, 4) for v in target], mode=int(args.get("mode", 2)),
                                            ok=ok, cost=gw.adapter.log.calls - c0, native_reason=reason or ("" if ok else "unknown"),
                                            region_ref=str(args.get("region_ref", ""))))
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)[:4000]})
        else:
            cap_reason = f"round cap {self.max_rounds} reached"
        # leave the engine idle and the net deselected, whatever the model did last
        if gw.adapter.e.is_routing():
            gw.call_recovery("cancel")
        if gw.adapter.current_net is not None:
            gw.call_recovery("end_net")
        parsed = _try_json(final)
        groups = self._helper._groups(task)
        joined = self._helper._edge_joined(task)
        dev = self._helper._deviations(task)
        head = parsed.get("last_head") if isinstance(parsed.get("last_head"), list) and len(parsed["last_head"]) == 3 else [0.0, 0.0, -1]
        last_stage = task.stages[-1].corridor_id
        if joined:
            outcome, reason, request = "done", "", ""
        else:
            outcome = parsed.get("outcome") if parsed.get("outcome") in ("blocked", "partial", "conflict") else \
                ("partial" if any(a.ok for a in attempts) else "blocked")
            reason = cap_reason or str(parsed.get("native_reason") or "target not reached")
            request = str(parsed.get("request") or "widen")
        region_ref = str(parsed.get("region_ref") or last_stage)
        fb = self._helper._packet(task, outcome, head, attempts, reason, region_ref, request, stage_ref=region_ref, deviations=dev)
        self.transcripts.append({"task": task.task_id, "rounds": rounds, "final": final, "outcome": outcome})
        return TaskResult(task.task_id, task.net_code, outcome, steps, fb, groups, dev, gw.adapter.log.calls - calls0)


def _try_json(text) -> dict:
    if not text:
        return {}
    try:
        d = _extract_json(str(text))
        return d if isinstance(d, dict) else {}
    except (ValueError, json.JSONDecodeError):
        return {}
