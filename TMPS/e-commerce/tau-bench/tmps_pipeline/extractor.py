"""Extract PredictionEvents from tau-bench retail task definitions.

Each task defines a ground-truth action sequence. We treat each action position
as a prediction event: context = task instruction + previous actions, GT = next action.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionEvent:
    event_id: str
    event_type: str  # "hit" | "miss"
    task_instruction: str
    step_number: int
    total_steps: int
    context: str  # formatted prompt: instruction + previous actions
    predictions: list[str]  # k predicted tool calls (filled by speculator)
    actual_action: str  # ground truth: tool_name(kwargs_json)
    actual_action_type: str  # tool name
    prediction_rank: int  # 0-based hit rank, -1 = miss
    task_id: int
    user_id: str = ""
    token_usage: dict = field(default_factory=dict)  # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}


def load_tasks_from_module(module_path: str) -> list[Any]:
    """Load TASKS_TEST or TASKS_TRAIN from a Python module by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tasks_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ["TASKS_TEST", "TASKS_TRAIN", "TASKS_DEV"]:
        if hasattr(module, attr):
            return getattr(module, attr)
    raise ValueError(f"No TASKS_* found in {module_path}")


def action_to_str(action) -> str:
    """Convert Action to a compact string: tool_name({"key": "value"})"""
    kwargs = getattr(action, "kwargs", {}) or {}
    return f'{action.name}({json.dumps(kwargs, ensure_ascii=False)})'


def extract_events_from_tasks(
    tasks: list[Any],
    max_tasks: int = -1,
) -> list[PredictionEvent]:
    """Convert task definitions into prediction events."""
    events = []
    task_list = tasks[:max_tasks] if max_tasks > 0 else tasks

    for task in task_list:
        actions = task.actions
        if not actions:
            continue
        task_id = getattr(task, 'id', task_list.index(task))
        user_id = getattr(task, 'user_id', '')

        prev_actions: list[str] = []
        for step_idx, action in enumerate(actions):
            actual_str = action_to_str(action)
            # Build context: task instruction + previous actions
            context_parts = [f"Task: {task.instruction}"]
            if prev_actions:
                context_parts.append("Previous actions:")
                for pa in prev_actions:
                    context_parts.append(f"  {pa}")
            context_parts.append("Predict the next API call:")
            context = "\n".join(context_parts)

            events.append(PredictionEvent(
                event_id=f"task_{task_id}/step_{step_idx}",
                event_type="miss",  # filled after speculation
                task_instruction=task.instruction,
                step_number=step_idx,
                total_steps=len(actions),
                context=context,
                predictions=[],
                actual_action=actual_str,
                actual_action_type=action.name,
                prediction_rank=-1,
                task_id=task_id,
                user_id=user_id,
            ))
            prev_actions.append(actual_str)

    return events


def split_events(events: list[PredictionEvent]) -> tuple[list[PredictionEvent], list[PredictionEvent]]:
    """Will be filled: split into hits and misses after speculation."""
    hits = [e for e in events if e.event_type == "hit"]
    misses = [e for e in events if e.event_type == "miss"]
    return hits, misses


def chunk_events(events: list[PredictionEvent], chunk_size: int = 15) -> list[list[PredictionEvent]]:
    """Group events by action type, then chunk."""
    from collections import defaultdict
    by_type: dict[str, list[PredictionEvent]] = defaultdict(list)
    for e in events:
        by_type[e.actual_action_type].append(e)

    chunks = []
    for action_type, group in sorted(by_type.items()):
        for i in range(0, len(group), chunk_size):
            chunks.append(group[i:i + chunk_size])
    return chunks


def build_summary(events: list[PredictionEvent]) -> dict:
    """Compute aggregate statistics."""
    from collections import Counter
    if not events:
        return {"n_events": 0, "hit_rate": 0.0}
    n = len(events)
    hits = sum(1 for e in events if e.event_type == "hit")
    by_type = Counter()
    by_type_hits = Counter()
    for e in events:
        by_type[e.actual_action_type] += 1
        if e.event_type == "hit":
            by_type_hits[e.actual_action_type] += 1
    return {
        "n_events": n,
        "n_hits": hits,
        "hit_rate": hits / n if n > 0 else 0.0,
        "by_action_type": dict(by_type),
        "by_action_type_hit_rate": {
            t: by_type_hits[t] / by_type[t] if by_type[t] > 0 else 0.0
            for t in by_type
        },
    }
