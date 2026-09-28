"""Builder construction and feedback-driven refinement over the v2 harness."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .harness.target_runner import Tool
from .modular.runtime import ModularSession, WORKBENCH_NAMES
from .modular.schema import HarnessBundle, SupportSkill, BundleValidationError


BUILDER_SYSTEM = """You design reusable environments for a separate Target agent.
Return one JSON object only. Do not solve a task or write a task answer into the harness.
The available components are instruction, memory, tools, context, controller,
verification, and workspace. A component can be null or {"source":"..."}.
Disabled components MUST be JSON null, never {"source":""}. Enable a component
only when its source contains all required nonempty callback function definitions.
Code sources contain only top-level Python function definitions in the bounded
policy language. No imports, files, network, model calls, eval, or exec.
Callbacks: memory.write(event, records) -> {"upsert":[],"delete":[]};
memory.retrieve(query, records) -> list of record IDs;
context.select_context(groups, retrieved) -> group IDs;
controller.on_event(event, snapshot) -> object with optional state, phase,
visible_tools, instruction; verification.before_finish(answer, snapshot) ->
{"allow":bool,"feedback":str,"actions":[]}; optional
verification.on_failure(failure, snapshot) -> {"actions":[],"feedback":str};
workspace.setup(task) -> {"files":[],"actions":[]};
each composed tool next_action(arguments, state, previous) ->
{"state":{},"call":{"tool":name,"arguments":{}}} or
{"state":{},"return":value}. A call uses a supplied primitive.
Required bundle fields: interface_version="modular-v2.0", instruction,
memory, tools, context, controller, verification, workspace. A valid minimal
example is {"interface_version":"modular-v2.0","instruction":"Use the public
tools and verify observations.","memory":null,"tools":[],"context":null,
"controller":null,"verification":null,"workspace":null}. tools is a list of
{name, description, parameters, source}. Preserve the Target's freedom to
choose actions. Enable a component only when it provides a concrete benefit over
the neutral environment. A verifier should allow an answer supported by public
tool observations; it should not force extra checks that the adapter does not
provide. Only use public capabilities and feedback."""


class BuilderError(ValueError):
    pass


def _tool_definitions(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    return [tool.api_definition() for tool in tools]


def _text(completion: Any) -> str:
    message = getattr(completion, "message", None)
    value = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise BuilderError("Builder returned no JSON text")
    return value


def _bundle_from_text(raw: str, tool_names: list[str]) -> HarnessBundle:
    bundle = HarnessBundle.from_json(raw, primitive_tools=[*tool_names, *WORKBENCH_NAMES])
    session = ModularSession(bundle)
    session.close()
    return bundle


def generate_bundle(
    provider: Any, *, model: str, public_card: Mapping[str, Any],
    tools: Sequence[Tool], parent: HarnessBundle | None = None,
    skills: Sequence[SupportSkill] = (), max_retries: int = 2,
    max_completion_tokens: int = 12000,
) -> HarnessBundle:
    """Build a reusable bundle, or refine one using learned support skills.

    Candidate repair addresses interface errors only. No Target score is used to
    select among candidates.
    """
    if not isinstance(public_card, Mapping):
        raise TypeError("public_card must be a mapping")
    names = [tool.name for tool in tools]
    if len(set(names)) != len(names):
        raise BuilderError("duplicate primitive tools")
    payload = {"public_benchmark_card": dict(public_card),
               "primitive_tools": _tool_definitions(tools),
               "current_bundle": parent.to_dict() if parent else None,
               "learned_support_skills": [skill.to_dict() for skill in skills]}
    instruction = (
        "Refine the current bundle to implement the support skills. Return the complete "
        "bundle. Keep unrelated components stable."
        if parent else "Build a reusable harness for this distribution. Return the complete bundle."
    )
    messages = [{"role": "system", "content": BUILDER_SYSTEM},
                {"role": "user", "content": instruction + "\n" + json.dumps(payload, ensure_ascii=False)}]
    for attempt in range(max_retries + 1):
        completion = provider.create(model=model, messages=messages,
                                     max_completion_tokens=max_completion_tokens,
                                     reasoning_effort=None, seed=None, temperature=None,
                                     additional_fields={})
        raw = _text(completion)
        try:
            return _bundle_from_text(raw, names)
        except (BundleValidationError, ValueError) as error:
            if attempt == max_retries:
                raise BuilderError(f"Harness remained invalid: {error}") from error
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content":
                          f"Correct only the JSON/interface error: {error}. Disabled components must be null, never empty source strings. Return the full JSON bundle."}]
    raise AssertionError("unreachable")


def reflect_skill(
    provider: Any, *, model: str, skills: Sequence[SupportSkill],
    bundle: HarnessBundle, public_episode: Mapping[str, Any],
    max_completion_tokens: int = 1500,
) -> tuple[SupportSkill, ...]:
    """One Builder reflection may keep the library, revise one skill, or add one.

    Only the current public episode ID may be cited as new evidence. This
    function has no access to private scoring or test data.
    """
    episode_id = public_episode.get("instance_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise BuilderError("public episode needs instance_id")
    payload = {"current_skills": [skill.to_dict() for skill in skills],
               "generated_bundle": bundle.to_dict(), "public_episode": dict(public_episode)}
    system = (
        "Reflect on the Builder's own generated harness and this public training "
        "episode. Return one JSON object with operation keep, add, or revise. "
        "For keep, return {\"operation\":\"keep\"}. For add or revise, include "
        "a skill with exactly skill_id, interface_version, when, provide, use, "
        "evidence_ids. A revise operation also includes prior_skill_id. Each of "
        "when/provide/use must be at most 768 UTF-8 bytes. Describe a reusable "
        "support need, not a task answer. Cite only this episode ID for new evidence."
    )
    completion = provider.create(model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        max_completion_tokens=max_completion_tokens, reasoning_effort=None,
        seed=None, temperature=None, additional_fields={})
    try:
        choice = json.loads(_text(completion))
        operation = choice["operation"]
        if operation == "keep":
            return tuple(skills)
        if operation not in ("add", "revise"):
            raise BuilderError("unknown skill operation")
        known_ids = {episode_id, *(evidence for prior in skills for evidence in prior.evidence_ids)}
        skill = SupportSkill.from_dict(choice["skill"], known_episode_ids=known_ids)
        if episode_id not in skill.evidence_ids:
            raise BuilderError("updated skill must cite the current episode")
        if any(len(getattr(skill, name).encode("utf-8")) > 768 for name in ("when", "provide", "use")):
            raise BuilderError("skill semantic field exceeds 768 bytes")
        result = list(skills)
        if operation == "add":
            if any(item.skill_id == skill.skill_id for item in skills):
                raise BuilderError("new skill ID already exists")
            result.append(skill)
        else:
            prior = choice["prior_skill_id"]
            index = next((i for i, item in enumerate(skills) if item.skill_id == prior), None)
            if index is None:
                raise BuilderError("revised skill ID does not exist")
            if not set(skills[index].evidence_ids) <= set(skill.evidence_ids):
                raise BuilderError("revision must preserve prior evidence IDs")
            if any(item.skill_id == skill.skill_id for i, item in enumerate(skills) if i != index):
                raise BuilderError("revised skill ID collides")
            result[index] = skill
        return tuple(result)
    except (KeyError, TypeError, json.JSONDecodeError, BundleValidationError) as error:
        raise BuilderError(f"Invalid reflection: {error}") from error
