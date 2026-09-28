"""Small contract checks for the released workflow and both official routes."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from meta_skill_ai4ai import (
    TargetBudget, TargetTask, Tool, ToolResult, ModularTargetRunner,
    generate_bundle, reflect_skill, neutral_bundle,
)
from meta_skill_ai4ai.api import ClaudeProvider, OpenAIProvider
from meta_skill_ai4ai.api.official import _claude_messages, _openai_input


class QueueProvider:
    def __init__(self, *messages):
        self.messages = list(messages)

    def create(self, **kwargs):
        return SimpleNamespace(message=self.messages.pop(0),
                               metadata=SimpleNamespace(usage=SimpleNamespace(total_tokens=30)))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tool = Tool("lookup", "Lookup a catalog title", {
            "type": "object", "properties": {"title": {"type": "string"}},
            "required": ["title"], "additionalProperties": False,
        }, lambda args: ToolResult(True, {"shelf": 3}))

    def test_build_run_reflect_refine(self):
        base = neutral_bundle()
        build = QueueProvider({"role": "assistant", "content": json.dumps(base.to_dict())})
        bundle = generate_bundle(build, model="test", public_card={"name": "catalog"}, tools=[self.tool])
        self.assertEqual(bundle.sha256, base.sha256)
        run = QueueProvider(
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "type": "function", "function":
                 {"name": "lookup", "arguments": '{"title":"atlas"}'}}]},
            {"role": "assistant", "content": "Shelf 3"},
        )
        result = ModularTargetRunner(run).run(
            task=TargetTask("catalog-1", "Find atlas", {}), bundle=bundle,
            harness_id="test", tools=[self.tool],
            budget=TargetBudget(4, 6, 10000, 1000, 30),
            model="test", reasoning_effort=None, seed=None)
        self.assertEqual(result.final_answer, "Shelf 3")
        skill = {"skill_id": "catalog_check", "interface_version": "modular-v2.0",
                 "when": "A lookup is needed", "provide": "Expose the lookup tool",
                 "use": "Check the observed entry", "evidence_ids": ["catalog-1"]}
        reflection = QueueProvider({"role": "assistant", "content": json.dumps(
            {"operation": "add", "skill": skill})})
        skills = reflect_skill(reflection, model="test", skills=(), bundle=bundle,
                               public_episode={"instance_id": "catalog-1", "stop_reason": result.stop_reason})
        self.assertEqual(len(skills), 1)
        changed = base.to_dict()
        changed["instruction"] += " Check lookup observations."
        refinement = QueueProvider({"role": "assistant", "content": json.dumps(changed)})
        next_bundle = generate_bundle(refinement, model="test", public_card={"name": "catalog"},
                                      tools=[self.tool], parent=bundle, skills=skills)
        self.assertIn("Check lookup", next_bundle.instruction)

    def test_official_tool_transcript_translation(self):
        history = [
            {"role": "system", "content": "Work carefully"},
            {"role": "user", "content": "Find atlas"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "lookup", "arguments": '{"title":"atlas"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": '{"shelf":3}'},
        ]
        instruction, openai_items = _openai_input(history)
        self.assertEqual(instruction, "Work carefully")
        self.assertEqual(openai_items[-1]["type"], "function_call_output")
        system, claude_items = _claude_messages(history)
        self.assertEqual(system, "Work carefully")
        self.assertEqual(claude_items[-1]["content"][0]["type"], "tool_result")

    def test_official_providers_normalize_tool_calls(self):
        messages = [{"role": "user", "content": "Find atlas"}]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}), patch(
            "meta_skill_ai4ai.api.official._request",
            return_value={"model": "gpt-test", "status": "completed", "usage": {"total_tokens": 12},
                          "output": [{"type": "function_call", "call_id": "c1",
                                      "name": "lookup", "arguments": "{}"}]}) as request:
            response = OpenAIProvider().create(model="gpt-test", messages=messages,
                                                max_completion_tokens=100)
            self.assertEqual(response.message["tool_calls"][0]["id"], "c1")
            self.assertEqual(request.call_args.args[0], "https://api.openai.com/v1/responses")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-only"}), patch(
            "meta_skill_ai4ai.api.official._request",
            return_value={"model": "claude-test", "stop_reason": "tool_use",
                          "usage": {"input_tokens": 5, "output_tokens": 7},
                          "content": [{"type": "tool_use", "id": "c2", "name": "lookup", "input": {}}]}) as request:
            response = ClaudeProvider().create(model="claude-test", messages=messages,
                                                max_completion_tokens=100)
            self.assertEqual(response.message["tool_calls"][0]["id"], "c2")
            self.assertEqual(response.metadata.usage.total_tokens, 12)
            self.assertEqual(request.call_args.args[0], "https://api.anthropic.com/v1/messages")


if __name__ == "__main__":
    unittest.main()
