"""Official HTTP APIs, with one normalized completion shape for the harness."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ChatCompletionError(RuntimeError):
    def __init__(self, message: str, *, request_metadata: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.request_metadata = dict(request_metadata or {})


class AuthenticationFailure(ChatCompletionError):
    pass


class InfrastructureFailure(ChatCompletionError):
    pass


class RequestCompatibilityFailure(ChatCompletionError):
    pass


class ModelContentFailure(ChatCompletionError):
    pass


@dataclass(frozen=True)
class Usage:
    total_tokens: int | None


@dataclass(frozen=True)
class CompletionMetadata:
    usage: Usage
    request: Mapping[str, Any]
    response_model: str
    finish_reason: str | None


@dataclass(frozen=True)
class Completion:
    message: Mapping[str, Any]
    metadata: CompletionMetadata
    is_model_content_failure: bool = False


def _request(url: str, key: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        code = error.code
        category = (
            AuthenticationFailure if code in (401, 403) else
            InfrastructureFailure if code in (408, 409, 429) or code >= 500 else
            RequestCompatibilityFailure
        )
        raise category(f"Provider returned HTTP {code}", request_metadata={"http_status": code}) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise InfrastructureFailure(f"Provider request failed: {type(error).__name__}") from None
    except (UnicodeError, ValueError):
        raise ModelContentFailure("Provider returned invalid JSON") from None
    if not isinstance(result, dict):
        raise ModelContentFailure("Provider response must be an object")
    if result.get("error"):
        raise ModelContentFailure("Provider response contains an error")
    return result


def _validated_call(model: str, messages: Sequence[Mapping[str, Any]], max_completion_tokens: int) -> None:
    if not isinstance(model, str) or not model:
        raise ValueError("model must be nonempty")
    if not isinstance(messages, Sequence) or not messages:
        raise ValueError("messages must be nonempty")
    if type(max_completion_tokens) is not int or max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")


def _openai_input(messages: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions, items = [], []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role in ("system", "developer"):
            if isinstance(content, str):
                instructions.append(content)
        elif role in ("user", "assistant"):
            if isinstance(content, str) and content:
                items.append({"role": role, "content": content})
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    items.append({"type": "function_call", "call_id": call["id"],
                                  "name": function["name"], "arguments": function["arguments"]})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": content if isinstance(content, str) else json.dumps(content)})
        else:
            raise ValueError(f"Unsupported message role: {role}")
    return "\n\n".join(instructions), items


def _openai_tools(fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for tool in fields.get("tools", []):
        function = tool["function"]
        result.append({"type": "function", "name": function["name"],
                       "description": function.get("description", ""),
                       "parameters": function["parameters"], "strict": False})
    return result


class OpenAIProvider:
    """Uses https://api.openai.com/v1/responses and OPENAI_API_KEY."""

    def __init__(self, *, timeout: float = 120.0):
        self.timeout = timeout

    def create(self, *, model: str, messages: Sequence[Mapping[str, Any]],
               max_completion_tokens: int, reasoning_effort: str | None = None,
               seed: int | None = None, temperature: float | None = None,
               additional_fields: Mapping[str, Any] | None = None) -> Completion:
        _validated_call(model, messages, max_completion_tokens)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise AuthenticationFailure("OPENAI_API_KEY is required")
        fields = dict(additional_fields or {})
        instructions, inputs = _openai_input(messages)
        body: dict[str, Any] = {"model": model, "input": inputs,
                                "max_output_tokens": max_completion_tokens, "store": False}
        if instructions:
            body["instructions"] = instructions
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        tools = _openai_tools(fields)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = fields.get("tool_choice", "auto")
        result = _request("https://api.openai.com/v1/responses", key,
                          {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          body, self.timeout)
        if result.get("status") in ("failed", "incomplete"):
            raise ModelContentFailure("OpenAI response did not complete")
        text, calls = [], []
        for item in result.get("output", []):
            if item.get("type") == "function_call":
                calls.append({"id": item["call_id"], "type": "function", "function":
                              {"name": item["name"], "arguments": item["arguments"]}})
            elif item.get("type") == "message":
                text.extend(part["text"] for part in item.get("content", [])
                            if part.get("type") == "output_text" and isinstance(part.get("text"), str))
        message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text) or None}
        if calls:
            message["tool_calls"] = calls
        usage = result.get("usage") or {}
        metadata = CompletionMetadata(Usage(usage.get("total_tokens")), {},
                                      result.get("model", model), result.get("status"))
        return Completion(message, metadata)


def _claude_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system, output = [], []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role in ("system", "developer"):
            if isinstance(content, str):
                system.append(content)
        elif role == "assistant":
            blocks = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": call["id"],
                               "name": call["function"]["name"],
                               "input": json.loads(call["function"]["arguments"])})
            if blocks:
                output.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": message["tool_call_id"],
                     "content": content if isinstance(content, str) else json.dumps(content)}
            if output and output[-1]["role"] == "user" and isinstance(output[-1]["content"], list) \
                    and output[-1]["content"] and output[-1]["content"][0].get("type") == "tool_result":
                output[-1]["content"].append(block)
            else:
                output.append({"role": "user", "content": [block]})
        elif role == "user":
            output.append({"role": "user", "content": content})
        else:
            raise ValueError(f"Unsupported message role: {role}")
    return "\n\n".join(system), output


class ClaudeProvider:
    """Uses https://api.anthropic.com/v1/messages and ANTHROPIC_API_KEY."""

    def __init__(self, *, timeout: float = 120.0):
        self.timeout = timeout

    def create(self, *, model: str, messages: Sequence[Mapping[str, Any]],
               max_completion_tokens: int, reasoning_effort: str | None = None,
               seed: int | None = None, temperature: float | None = None,
               additional_fields: Mapping[str, Any] | None = None) -> Completion:
        _validated_call(model, messages, max_completion_tokens)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise AuthenticationFailure("ANTHROPIC_API_KEY is required")
        fields = dict(additional_fields or {})
        system, transcript = _claude_messages(messages)
        body: dict[str, Any] = {"model": model, "messages": transcript,
                                "max_tokens": max_completion_tokens}
        if system:
            body["system"] = system
        if fields.get("tools"):
            body["tools"] = [
                {"name": x["function"]["name"], "description": x["function"].get("description", ""),
                 "input_schema": x["function"]["parameters"]} for x in fields["tools"]
            ]
            body["tool_choice"] = {"type": "auto"}
        result = _request("https://api.anthropic.com/v1/messages", key,
                          {"x-api-key": key, "anthropic-version": "2023-06-01",
                           "Content-Type": "application/json"}, body, self.timeout)
        text, calls = [], []
        for block in result.get("content", []):
            if block.get("type") == "text":
                text.append(block["text"])
            elif block.get("type") == "tool_use":
                calls.append({"id": block["id"], "type": "function", "function":
                              {"name": block["name"], "arguments": json.dumps(block["input"])}})
        message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text) or None}
        if calls:
            message["tool_calls"] = calls
        usage = result.get("usage") or {}
        total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        metadata = CompletionMetadata(Usage(total), {}, result.get("model", model),
                                      result.get("stop_reason"))
        return Completion(message, metadata)


def provider_for(name: str) -> OpenAIProvider | ClaudeProvider:
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return ClaudeProvider()
    raise ValueError("provider must be 'openai' or 'anthropic'")
