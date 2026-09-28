"""Tiny task adapter showing how native capabilities plug into the harness."""

from meta_skill_ai4ai import TargetTask, Tool, ToolResult

CATALOG = {"atlas": "A map collection on shelf 3", "botany": "A plant guide on shelf 7"}
SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}},
          "required": ["title"], "additionalProperties": False}


def public_tools():
    return [Tool("lookup", "Look up a title in the public catalog.", SCHEMA,
                 lambda args: ToolResult(True, {"entry": CATALOG.get(args["title"], "missing")}))]


def load_task(task_id):
    if task_id != "catalog-1":
        raise ValueError("unknown task")
    return (TargetTask(task_id, "Find the shelf for the atlas in the catalog.", {}),
            public_tools())


def public_feedback(result):
    return {"accepted": result.final_answer is not None and "3" in result.final_answer}
