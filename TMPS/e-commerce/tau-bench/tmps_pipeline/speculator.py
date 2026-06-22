"""E-commerce tool-call speculator: predicts next k API calls given task context."""

import json
import os
import re
import sys
from typing import Optional

# Use hotpotqa's LLM client (shared DeepSeek API wrapper)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "hotpotqa"))
from src.llm_client import LLMClient

from .extractor import PredictionEvent


TOOL_DESCRIPTIONS = {
    "find_user_id_by_name_zip": "Find user ID by first_name, last_name, zip",
    "find_user_id_by_email": "Find user ID by email",
    "get_user_details": "Get user details by user_id",
    "get_order_details": "Get order details by order_id",
    "get_product_details": "Get product details by product_id",
    "list_all_product_types": "List all product types",
    "calculate": "Calculate a mathematical expression",
    "modify_pending_order_items": "Modify items in a pending order (order_id, item_ids, new_item_ids, payment_method_id)",
    "modify_pending_order_address": "Modify shipping address of pending order (order_id, address_id)",
    "modify_pending_order_payment": "Modify payment method of pending order (order_id, payment_method_id)",
    "cancel_pending_order": "Cancel a pending order (order_id, payment_method_id)",
    "return_delivered_order_items": "Return delivered order items (order_id, item_ids, payment_method_id)",
    "exchange_delivered_order_items": "Exchange delivered order items (order_id, item_ids, new_item_ids, payment_method_id)",
    "modify_user_address": "Modify user address (user_id, address_id)",
    "transfer_to_human_agents": "Transfer to human agents with summary",
}


def _build_system_prompt() -> str:
    tool_lines = []
    for name, desc in sorted(TOOL_DESCRIPTIONS.items()):
        tool_lines.append(f"- {name}: {desc}")
    tools_text = "\n".join(tool_lines)

    return f"""You are a customer service AI agent for an e-commerce platform. Given a user request and the actions already taken, predict the NEXT API call the agent should make.

Available API calls:
{tools_text}

Rules:
- Predict the most likely next API call based on the task and previous actions
- Parameter values must come from the task description or previous action results
- Use exact values from the task (order IDs, product IDs, user info)
- If no more actions needed, predict: respond({{"content": "..."}})
"""


def format_prediction_prompt(event: PredictionEvent, k: int = 3) -> str:
    """Build user prompt for predicting the next action."""
    parts = [event.context]
    parts.append(f"\nReturn exactly {k} candidate API calls, one per line, in format: tool_name(JSON kwargs)")
    return "\n".join(parts)


def parse_predictions(text: str) -> list[str]:
    """Extract tool_name(JSON) predictions from LLM output."""
    pattern = r'(\w+)\(\s*(\{.+?\})\s*\)'
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for tool_name, kwargs_str in matches:
        try:
            kwargs = json.loads(kwargs_str)
            results.append(f'{tool_name}({json.dumps(kwargs, ensure_ascii=False)})')
        except json.JSONDecodeError:
            results.append(f'{tool_name}({kwargs_str.strip()})')
    return results


class EcommerceSpeculator:
    def __init__(self, model_name: str = "deepseek-chat", k: int = 3,
                 skill_file: Optional[str] = None):
        self.model_name = model_name
        self.k = k
        self.llm = LLMClient(
            model_name=model_name,
            temperature=0.0,
            max_tokens=512,
            top_p=0.9,
        )
        self.system_prompt = _build_system_prompt()
        self.skill_text = ""
        if skill_file:
            path = skill_file
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), "..", path)
            if os.path.exists(path):
                with open(path) as f:
                    self.skill_text = f.read().strip()

    def predict(self, event: PredictionEvent) -> list[str]:
        """Predict k candidate next actions for a single event."""
        preds, _ = self.predict_with_tokens(event)
        return preds

    def predict_with_tokens(self, event: PredictionEvent) -> tuple[list[str], dict]:
        """Predict k candidate next actions, returning (predictions, token_usage)."""
        user_prompt = format_prediction_prompt(event, self.k)
        if self.skill_text:
            user_prompt += "\n\n" + self.skill_text

        response_text, usage = self.llm.call_with_system_and_usage(
            self.system_prompt, user_prompt
        )
        predictions = parse_predictions(response_text or "")
        if not predictions:
            predictions = ["unknown()"]
        return predictions[:self.k], usage

    def evaluate_event(self, event: PredictionEvent) -> PredictionEvent:
        """Run prediction on one event and update hit/miss status."""
        preds, usage = self.predict_with_tokens(event)
        event.predictions = preds
        event.token_usage = usage

        # Check if ground truth is in predictions (strict match)
        actual = event.actual_action
        for rank, pred in enumerate(preds):
            if pred == actual:
                event.event_type = "hit"
                event.prediction_rank = rank
                break
        return event

    def evaluate_batch(self, events: list[PredictionEvent]) -> list[PredictionEvent]:
        """Evaluate prediction on a batch of events."""
        for i, event in enumerate(events):
            self.evaluate_event(event)
            if (i + 1) % 10 == 0:
                hits = sum(1 for e in events[:i + 1] if e.event_type == "hit")
                print(f"  [{i + 1}/{len(events)}] hit_rate={hits / (i + 1):.1%}")
        return events
