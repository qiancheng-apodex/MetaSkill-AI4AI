"""Small local workflow: build, run, learn, refine."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from .api import provider_for
from .builder import generate_bundle, reflect_skill
from .harness.target_runner import TargetBudget
from .modular.runtime import ModularTargetRunner, WORKBENCH_NAMES
from .modular.schema import HarnessBundle, SupportSkill


def _read(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _adapter(name: str) -> Any:
    return importlib.import_module(name)


def _tools(adapter: Any):
    tools = adapter.public_tools()
    if len({tool.name for tool in tools}) != len(tools):
        raise ValueError("adapter has duplicate public tools")
    return tools


def _bundle(path: str, tools: Any) -> HarnessBundle:
    return HarnessBundle.from_dict(_read(path), primitive_tools=[*(x.name for x in tools), *WORKBENCH_NAMES])


def _skills(path: str | None) -> tuple[SupportSkill, ...]:
    if path is None:
        return ()
    return tuple(SupportSkill.from_dict(value) for value in _read(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MetaSkill minimal harness")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "run", "learn", "refine"):
        command = sub.add_parser(name)
        command.add_argument("--adapter", required=True, help="Python module exposing public_tools()")
        command.add_argument("--provider", choices=("openai", "anthropic"), required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--output", required=True)
        if name in ("run", "learn", "refine"):
            command.add_argument("--bundle", required=True)
        if name in ("build", "refine"):
            command.add_argument("--card", required=True, help="Public distribution card JSON")
        if name == "run":
            command.add_argument("--task-id", required=True)
            command.add_argument("--max-steps", type=int, default=8)
            command.add_argument("--max-tool-calls", type=int, default=16)
            command.add_argument("--max-tokens", type=int, default=12000)
            command.add_argument("--max-output-tokens", type=int, default=2000)
            command.add_argument("--wall-seconds", type=float, default=300)
        if name == "learn":
            command.add_argument("--episode", required=True, help="Public training episode JSON")
        if name in ("learn", "refine"):
            command.add_argument("--skills", help="Current skill library JSON; omitted means empty")
    args = parser.parse_args(argv)
    adapter = _adapter(args.adapter)
    tools = _tools(adapter)
    provider = provider_for(args.provider)
    if args.command in ("build", "refine"):
        parent = _bundle(args.bundle, tools) if args.command == "refine" else None
        bundle = generate_bundle(provider, model=args.model, public_card=_read(args.card),
                                 tools=tools, parent=parent,
                                 skills=_skills(args.skills) if parent else ())
        _write(args.output, bundle.to_dict())
        print(f"bundle_sha256={bundle.sha256}")
    elif args.command == "run":
        bundle = _bundle(args.bundle, tools)
        task, task_tools = adapter.load_task(args.task_id)
        if {tool.name for tool in task_tools} != {tool.name for tool in tools}:
            raise ValueError("task tools differ from public adapter tools")
        result = ModularTargetRunner(provider).run(
            task=task, bundle=bundle, harness_id=bundle.sha256[:12], tools=task_tools,
            budget=TargetBudget(args.max_steps, args.max_tool_calls, args.max_tokens,
                                args.max_output_tokens, args.wall_seconds),
            model=args.model, reasoning_effort=None, seed=None, temperature=None)
        public_events = []
        for event in result.events:
            if event.get("type") == "module_failure":
                public_events.append({"type": "module_failure",
                                      "public_diagnostic": event.get("public_diagnostic")})
            elif event.get("type") != "model_request":
                public_events.append(dict(event))
        episode = {"instance_id": result.instance_id, "harness_sha256": result.harness_sha256,
                   "stop_reason": result.stop_reason, "final_answer": result.final_answer,
                   "model_calls": result.model_calls, "tool_calls": result.tool_calls,
                   "accounted_tokens": result.accounted_tokens,
                   "transcript": list(result.transcript), "events": public_events}
        if callable(getattr(adapter, "public_feedback", None)):
            episode["public_feedback"] = adapter.public_feedback(result)
        _write(args.output, episode)
        print(f"stop_reason={result.stop_reason} final_answer={result.final_answer!r}")
    else:
        bundle = _bundle(args.bundle, tools)
        skills = reflect_skill(provider, model=args.model, skills=_skills(args.skills),
                               bundle=bundle, public_episode=_read(args.episode))
        _write(args.output, [skill.to_dict() for skill in skills])
        print(f"skills={len(skills)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
