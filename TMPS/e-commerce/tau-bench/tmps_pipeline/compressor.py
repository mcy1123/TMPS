"""Latency-constrained compression for trajectory-evolved e-commerce skills.

Mirrors chess-game/trace2skill/compressor.py: rule-based selection → LLM refinement.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompressedSkill:
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CompressionRule:
    theme: str
    text: str
    triggers: tuple[str, ...]
    priority: int


# ── E-commerce prediction rules ────────────────────────────────────

RULES: tuple[CompressionRule, ...] = (
    # ── User Identification (CRITICAL) ──
    CompressionRule(
        "User Identification",
        "- Full name + zip → `find_user_id_by_name_zip` first. Email → `find_user_id_by_email` first. Both → prefer name+zip.",
        ("find_user_id_by_name_zip", "full name", "zip code", "most reliable"),
        100,
    ),
    CompressionRule(
        "User Identification",
        "- Capitalize first letter of first_name and last_name. Extract zip from CURRENT location, not old address.",
        ("capitalize the first letter", "current location", "not an old"),
        98,
    ),
    CompressionRule(
        "User Identification",
        "- Multiple emails → call `find_user_id_by_email` for each distinct email in sequence.",
        ("multiple email", "each distinct email"),
        92,
    ),
    CompressionRule(
        "User Identification",
        "- Zip code corrections → predict `find_user_id_by_name_zip` for each zip in order, one step at a time. After successful identification → `get_user_details` next.",
        ("zip code corrections", "zip code is later corrected", "one step at a time"),
        90,
    ),

    # ── User Identification Completion (CRITICAL) ──
    CompressionRule(
        "User Identification Completion",
        "- After `find_user_id_by_name_zip` or `find_user_id_by_email` → ALWAYS predict `get_user_details({\"user_id\": \"<returned_id>\"})` next.",
        ("get_user_details", "immediate next step", "mandatory before any order"),
        100,
    ),
    CompressionRule(
        "User Identification Completion",
        "- user_id format: `{first_name}_{last_name}_{4-digit_number}`. Never use placeholders like \"result_from_previous_call\".",
        ("first_name}_{last_name}_{4-digit_number", "result_from_previous_call", "not valid user"),
        98,
    ),
    CompressionRule(
        "User Identification Completion",
        "- If user_id not yet known, do NOT predict `get_user_details` — wait for identification to return the actual user_id.",
        ("user_id is not yet known", "wait until the identification"),
        94,
    ),

    # ── Known User ID (CRITICAL) ──
    CompressionRule(
        "Known User ID",
        "- Task says \"You are {user_id}\" → predict `get_user_details({\"user_id\": \"<explicit_id>\"})` as FIRST action. Bypass all identification calls.",
        ("you are", "explicitly stated", "bypass", "known user"),
        100,
    ),
    CompressionRule(
        "Known User ID",
        "- Both explicit user ID AND email provided → prioritize `get_user_details` with the user ID over `find_user_id_by_email`.",
        ("both an explicit user id and an email", "definitive identifier"),
        96,
    ),

    # ── Parameter Accuracy (CRITICAL) ──
    CompressionRule(
        "Parameter Accuracy",
        "- Order IDs ALWAYS include \"#\" prefix (e.g., \"#W2378156\"). Copy exact IDs from task or API responses. Never invent IDs.",
        ("#\" prefix", "copy them exactly", "never invent", "never use placeholder"),
        100,
    ),
    CompressionRule(
        "Parameter Accuracy",
        "- `get_user_details` takes exactly ONE parameter: `user_id`. Do not bundle order_id or other params.",
        ("get_user_details", "takes exactly one parameter", "do not bundle"),
        98,
    ),
    CompressionRule(
        "Parameter Accuracy",
        "- Product IDs are numeric strings (e.g., \"8310926033\"). Payment method IDs: \"credit_card_XXXXXXX\", \"paypal_XXXXXXX\", \"gift_card_XXXXXXX\".",
        ("numeric strings", "payment method ids follow", "credit_card_", "paypal_", "gift_card_"),
        94,
    ),

    # ── Order Operations (HIGH) ──
    CompressionRule(
        "Order Operations",
        "- After user identification + get_user_details → `get_order_details` for each order ID found, one at a time, in order.",
        ("get_order_details", "systematically", "one at a time", "in the order they appear"),
        96,
    ),
    CompressionRule(
        "Order Operations",
        "- Never skip to modification/return/exchange before retrieving ALL relevant order details.",
        ("never skip", "before retrieving all relevant order details", "modify_pending_order_items"),
        94,
    ),
    CompressionRule(
        "Order Operations",
        "- User \"guesses\" order number → still use that ID with \"#\" prefix as first `get_order_details` call.",
        ("guess", "not 100% sure", "still use that order id"),
        90,
    ),

    # ── Action Sequencing (HIGH) ──
    CompressionRule(
        "Action Sequencing",
        "- Standard workflow: identify user → get_user_details → get_order_details → get_product_details → modify/exchange/return/cancel.",
        ("standard workflow", "identify user", "get_user_details", "get_order_details"),
        96,
    ),
    CompressionRule(
        "Action Sequencing",
        "- At step 0: predict ONLY the first API call. Cancellation + return → complete all cancellations first, then returns.",
        ("step 0", "predict only the first", "complete all cancellations first"),
        92,
    ),

    # ── Cancel Pending Order (HIGH) ──
    CompressionRule(
        "Cancel Pending Order",
        "- `cancel_pending_order` takes exactly two params: `order_id` and `reason`. Do NOT include `payment_method_id`.",
        ("cancel_pending_order", "takes exactly two parameters", "order_id", "reason"),
        98,
    ),
    CompressionRule(
        "Cancel Pending Order",
        "- \"Cancel all pending orders\" → predict MULTIPLE `cancel_pending_order` calls matching pending order count. Same reason across all.",
        ("cancel all pending orders", "multiple", "one for each pending order"),
        94,
    ),
    CompressionRule(
        "Cancel Pending Order",
        "- Use `cancel_pending_order` ONLY for orders explicitly described as \"pending\" or \"just ordered\". NEVER at step 0.",
        ("only for orders", "pending", "just ordered", "never predict cancel_pending_order at step 0"),
        90,
    ),

    # ── Exchange Operations (HIGH) ──
    CompressionRule(
        "Exchange Operations",
        "- Before `exchange_delivered_order_items` → ALWAYS call `get_product_details` for each product ID. Never invent item_ids or new_item_ids.",
        ("exchange_delivered_order_items", "get_product_details", "never invent", "new_item_ids"),
        94,
    ),
    CompressionRule(
        "Exchange Operations",
        "- `payment_method_id` must be from actual user payment methods. `item_ids` = list of numeric strings from order data.",
        ("payment_method_id", "actual user", "item_ids", "list of numeric strings"),
        90,
    ),
    CompressionRule(
        "Exchange Operations",
        "- Check order status before deciding: `modify_pending_order_items` (pending) vs `exchange_delivered_order_items` (delivered).",
        ("check the order status", "modify_pending_order_items", "exchange_delivered_order_items"),
        88,
    ),

    # ── Return Operations (HIGH) ──
    CompressionRule(
        "Return Operations",
        "- Before `return_delivered_order_items` → ALWAYS call `get_order_details` for exact item_ids and payment_method_id.",
        ("return_delivered_order_items", "get_order_details", "exact item_ids"),
        96,
    ),
    CompressionRule(
        "Return Operations",
        "- \"Return everything\" → include ALL item_ids from that order. \"Return only [item]\" → include ONLY that item's ID.",
        ("return everything", "all item_ids", "return only", "only that item"),
        94,
    ),
    CompressionRule(
        "Return Operations",
        "- One `return_delivered_order_items` call per order. Payment method: user-specified > original order method. Never use \"original\" or \"default\" as ID.",
        ("one call per order", "payment method", "original", "never use"),
        92,
    ),
    CompressionRule(
        "Return Operations",
        "- Use `cancel_pending_order` for pending orders, `return_delivered_order_items` for delivered orders — never confuse the two.",
        ("cancel_pending_order", "return_delivered_order_items", "pending", "delivered"),
        90,
    ),

    # ── Address Modification (HIGH) ──
    CompressionRule(
        "Address Modification",
        "- `modify_pending_order_address` expects structured fields (address1, address2, city, state, country, zip). `modify_user_address` uses flat string fields.",
        ("modify_pending_order_address", "structured", "modify_user_address", "flat string"),
        94,
    ),
    CompressionRule(
        "Address Modification",
        "- \"Address wrong\" with pending order → prioritize `modify_pending_order_address`. Address changes do NOT require prior order/product lookups.",
        ("address wrong", "modify_pending_order_address", "do not require prior"),
        90,
    ),
    CompressionRule(
        "Address Modification",
        "- User provides full current address in task → use those exact values directly. Do not call `get_user_details` to \"verify\".",
        ("full current address", "use those exact values", "do not call get_user_details"),
        88,
    ),

    # ── Product Discovery (MEDIUM) ──
    CompressionRule(
        "Product Discovery",
        "- \"How many [type] options\" or price queries without specific product ID → `list_all_product_types({})` first. This is prerequisite before `get_product_details`.",
        ("list_all_product_types", "how many", "prerequisite", "catalog"),
        88,
    ),
    CompressionRule(
        "Product Discovery",
        "- After `list_all_product_types`, use returned numeric IDs — never descriptive names as product_id.",
        ("list_all_product_types", "only valid ids", "numeric", "descriptive names"),
        84,
    ),

    # ── Product Detail Retrieval (HIGH) ──
    CompressionRule(
        "Product Detail Retrieval",
        "- `get_product_details` ALWAYS uses exact numeric product ID string. Never descriptive names like \"jigsaw\" or \"t-shirt\".",
        ("get_product_details", "numeric product id", "never use descriptive", "exact numeric"),
        92,
    ),
    CompressionRule(
        "Product Detail Retrieval",
        "- Product described by description only → first retrieve numeric ID from order details via `get_order_details`.",
        ("by description only", "retrieve the numeric product id", "get_order_details"),
        88,
    ),
)

THEME_ORDER = (
    "User Identification",
    "User Identification Completion",
    "Known User ID",
    "Parameter Accuracy",
    "Order Operations",
    "Action Sequencing",
    "Cancel Pending Order",
    "Exchange Operations",
    "Return Operations",
    "Address Modification",
    "Product Discovery",
    "Product Detail Retrieval",
)

THEME_MIN_RULES = {
    "User Identification": 2,
    "User Identification Completion": 2,
    "Known User ID": 1,
    "Parameter Accuracy": 2,
    "Order Operations": 1,
    "Action Sequencing": 1,
    "Cancel Pending Order": 1,
    "Exchange Operations": 1,
    "Return Operations": 1,
    "Address Modification": 1,
    "Product Discovery": 1,
    "Product Detail Retrieval": 1,
}

COMPRESSOR_REFINE_SYSTEM_PROMPT = """\
You are a skill compression editor for e-commerce customer service action prediction.
You receive a rule-based compressed draft and the original full skill document.
Your job: polish the draft into a clean, deployment-ready SKILL.md under a strict
character budget, while preserving all critical prediction rules.

## Refinement Rules
1. **Preserve all rules from the draft**: Every bullet point in the draft encodes
   a critical prediction pattern. Do NOT drop any rule unless it duplicates another.
2. **Merge near-duplicates**: If two rules say essentially the same thing, combine them.
3. **Tighten wording**: Remove filler words. Make bullet points more concise while
   keeping the exact technical content (API names, parameter names, format specs).
4. **Keep section structure**: Use the same headings as the draft. Maintain priority
   labels (CRITICAL, HIGH, MEDIUM).
5. **Respect the budget**: Final output MUST be <= the character limit.
6. **No new rules**: Only polish what's in the draft. Do not invent rules.
7. **Output only the SKILL.md content** — no preamble, no commentary.
"""


def _refine_via_llm(rough_draft: str, original_skill: str, char_budget: int) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("  DEEPSEEK_API_KEY not set, skipping LLM refinement")
        return ""

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    user_msg = (
        f"Polish the following rule-based compressed draft into a clean SKILL.md.\n"
        f"Character budget: {char_budget} (strict).\n\n"
        f"## Rule-Based Draft (polish this):\n\n{rough_draft}\n\n"
        f"## Original Full Skill (reference for context):\n\n{original_skill[:3000]}\n\n"
        f"---\n"
        f"Output the polished SKILL.md now. Do NOT drop any rules from the draft."
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": COMPRESSOR_REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip()


# ── Core pipeline ───────────────────────────────────────────────────

def compress_skill(
    skill_text: str,
    char_budget: int = 2800,
    min_rules_per_theme: dict[str, int] | None = None,
    use_llm_refinement: bool = True,
) -> CompressedSkill:
    source = _normalize(skill_text)
    selected = _select_rules(source)
    selected = _fit_budget(selected, char_budget, min_rules_per_theme or THEME_MIN_RULES)
    rough_draft = _render(selected)

    llm_refined = False
    content = rough_draft
    if use_llm_refinement:
        try:
            print("  Running LLM refinement...")
            refined = _refine_via_llm(rough_draft, skill_text, char_budget)
            if refined:
                content = refined
                llm_refined = True
                print(f"  LLM refinement OK: {len(rough_draft)} → {len(content)} chars")
        except Exception as exc:
            print(f"  LLM refinement failed ({exc}), using rule-based draft")

    metadata = {
        "input_chars": len(skill_text),
        "output_chars": len(content),
        "char_budget": char_budget,
        "compression_ratio": len(content) / len(skill_text) if skill_text else 0.0,
        "themes": {theme: len(selected.get(theme, [])) for theme in THEME_ORDER if selected.get(theme)},
        "method": "rule-based selection + LLM refinement"
                  if llm_refined else "rule-based selection only",
        "llm_refined": llm_refined,
        "timestamp": datetime.now().isoformat(),
    }
    return CompressedSkill(content=content, metadata=metadata)


def save_compressed_skill(result: CompressedSkill, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    skill_path = output_path / "SKILL.md"
    skill_path.write_text(result.content, encoding="utf-8")
    (output_path / "compression_log.json").write_text(
        json.dumps(result.metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return skill_path


def _select_rules(source: str) -> dict[str, list[CompressionRule]]:
    selected: dict[str, list[CompressionRule]] = {theme: [] for theme in THEME_ORDER}
    for rule in RULES:
        if _matches(source, rule):
            selected[rule.theme].append(rule)
    for theme in selected:
        selected[theme] = sorted(selected[theme], key=lambda r: r.priority, reverse=True)
    return selected


def _fit_budget(
    selected: dict[str, list[CompressionRule]],
    char_budget: int,
    min_rules_per_theme: dict[str, int],
) -> dict[str, list[CompressionRule]]:
    compact = {theme: list(rules) for theme, rules in selected.items() if rules}
    while len(_render(compact)) > char_budget:
        removable = []
        for theme, rules in compact.items():
            min_count = min_rules_per_theme.get(theme, 0)
            if len(rules) > min_count:
                removable.append((rules[-1].priority, theme))
        if not removable:
            break
        _, theme = min(removable)
        compact[theme].pop()
        if not compact[theme]:
            del compact[theme]
    return compact


def _render(selected: dict[str, list[CompressionRule]]) -> str:
    priority_labels = {
        "User Identification": "CRITICAL",
        "User Identification Completion": "CRITICAL",
        "Known User ID": "CRITICAL",
        "Parameter Accuracy": "CRITICAL",
        "Order Operations": "HIGH",
        "Action Sequencing": "HIGH",
        "Cancel Pending Order": "HIGH",
        "Exchange Operations": "HIGH",
        "Return Operations": "HIGH",
        "Address Modification": "HIGH",
        "Product Discovery": "MEDIUM",
        "Product Detail Retrieval": "HIGH",
    }
    lines = [
        "# E-Commerce Customer Service Action Prediction Skill",
        "",
        "Compact guide for predicting API call sequences in e-commerce customer service.",
        "",
    ]
    for theme in THEME_ORDER:
        rules = selected.get(theme, [])
        if not rules:
            continue
        label = priority_labels.get(theme, "HIGH")
        lines.append(f"## {theme} ({label})")
        lines.append("")
        lines.extend(rule.text for rule in rules)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _matches(source: str, rule: CompressionRule) -> bool:
    return any(_normalize(trigger) in source for trigger in rule.triggers)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Compress e-commerce prediction skill")
    p.add_argument("--input", default="./skills/v2_fixed_20260602_012546/SKILL.md")
    p.add_argument("--budget", type=int, default=2800)
    p.add_argument("--no-llm", action="store_true", help="Skip LLM refinement")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    with open(args.input) as f:
        full_skill = f.read()
    print(f"Input: {len(full_skill)} chars")

    result = compress_skill(full_skill, args.budget, use_llm_refinement=not args.no_llm)
    print(f"Output: {len(result.content)} chars (budget: {args.budget})")

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./skills/compressed_{ts}"

    out = save_compressed_skill(result, args.output_dir)
    print(f"Saved to {out}")
    print(f"Themes: {result.metadata['themes']}")


if __name__ == "__main__":
    main()
