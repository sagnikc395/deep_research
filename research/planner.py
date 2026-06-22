from __future__ import annotations

import json
import re

from pydantic import BaseModel

from .config import planner_model_id, planner_provider, planning_max_depth
from .memory import recall_relevant
from .models import hf_model
from .prompts import PLANNER_SYSTEM_PROMPT

MAX_PLANNING_DEPTH = planning_max_depth


class Subtask(BaseModel):
    id: str
    title: str
    description: str
    decompose: bool = False


class SubtaskList(BaseModel):
    subtasks: list[Subtask]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    raise ValueError(f"Could not extract JSON from model response: {text[:200]}")


def _memory_context(query: str) -> str:
    past = recall_relevant(query, limit=3)
    if not past:
        return ""
    lines = ["The following past research sessions may provide useful context:\n"]
    for i, s in enumerate(past, 1):
        lines.append(
            f"### Past Session {i} (from {s['created_at'][:10]})\n"
            f"**Query:** {s['query']}\n"
            f"**Plan excerpt:** {s['plan'][:500]}...\n"
        )
    return "\n".join(lines)


def _generate_subtasks(text: str, log) -> list[Subtask]:
    model = hf_model(planner_model_id, planner_provider)
    response = model(
        [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    )
    parsed = SubtaskList.model_validate(_extract_json(response.content))
    return parsed.subtasks


def plan_subtasks(query: str, log=print) -> list[dict]:
    ctx = _memory_context(query)
    message = f"{ctx}\n---\nNew research query:\n{query}" if ctx else query

    log("Planning research subtasks...")
    return _plan_recursive(message, prefix="", depth=0, log=log)


def _plan_recursive(
    text: str, prefix: str, depth: int, log
) -> list[dict]:
    subtasks = _generate_subtasks(text, log)

    result: list[dict] = []
    for task in subtasks:
        tid = f"{prefix}{task.id}" if not prefix else f"{prefix}.{task.id}"

        if task.decompose and depth < MAX_PLANNING_DEPTH:
            log(f"  Recursively decomposing [{tid}] {task.title} (depth {depth + 1})...")
            children = _plan_recursive(
                task.description, prefix=tid, depth=depth + 1, log=log,
            )
            result.extend(children)
        else:
            result.append({"id": tid, "title": task.title, "description": task.description})

    if depth == 0:
        log(f"Generated {len(result)} subtasks.")
        for t in result:
            log(f"  [{t['id']}] {t['title']}")

    return result
