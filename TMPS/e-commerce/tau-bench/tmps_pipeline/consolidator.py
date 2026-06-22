"""E-commerce domain consolidator: merge analyst patches into unified SKILL.md."""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

from .prompts import CONSOLIDATOR_SYSTEM_PROMPT, format_consolidator_user_message


def _normalize_section(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _group_patches_by_section(patches: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    normalized_keys: dict[str, str] = {}

    for patch in patches:
        section = patch.get("section", "General")
        norm = _normalize_section(section)
        matched_key = None
        for existing_norm, existing_key in normalized_keys.items():
            if norm in existing_norm or existing_norm in norm:
                if abs(len(norm) - len(existing_norm)) <= 5:
                    matched_key = existing_key
                    break
            if _jaccard_similarity(norm, existing_norm) > 0.6:
                matched_key = existing_key
                break
        if matched_key:
            groups[matched_key].append(patch)
        else:
            normalized_keys[norm] = section
            groups[section].append(patch)
    return dict(groups)


def _deduplicate_section(patches: list[dict]) -> list[dict]:
    if len(patches) <= 1:
        return patches
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    patches = sorted(patches, key=lambda p: priority_order.get(p.get("priority", "MEDIUM"), 1))
    kept: list[dict] = []
    for patch in patches:
        is_dup = False
        for existing in kept:
            sim = _jaccard_similarity(patch.get("content", ""), existing.get("content", ""))
            if sim > 0.7:
                existing_words = set(existing.get("content", "").lower().split())
                patch_words = set(patch.get("content", "").lower().split())
                new_content = patch_words - existing_words
                if new_content:
                    existing["content"] += "\n" + " ".join(new_content)
                existing["evidence"] += ", " + patch.get("evidence", "")
                is_dup = True
                break
        if not is_dup:
            kept.append(patch)
    return kept


def _make_llm():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "hotpotqa"))
    from src.llm_client import LLMClient
    return LLMClient(
        model_name="deepseek-chat",
        temperature=0.0,
        max_tokens=4096,
        top_p=0.9,
    )


def _llm_consolidate(patches: list[dict], existing_skill: str) -> str:
    llm = _make_llm()
    user_msg = format_consolidator_user_message(patches, existing_skill)
    response = llm.call_with_system(CONSOLIDATOR_SYSTEM_PROMPT, user_msg)
    return (response or "").strip()


def _rule_consolidate(patches: list[dict], existing_skill: str = "") -> str:
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    groups = _group_patches_by_section(patches)

    lines: list[str] = ["# E-Commerce Customer Service Action Prediction Skill", ""]
    lines.append("Prediction heuristics distilled from e-commerce execution trajectories.")
    lines.append("")

    for section_name, section_patches in sorted(
        groups.items(),
        key=lambda item: min(
            priority_order.get(p.get("priority", "MEDIUM"), 1) for p in item[1]
        ),
    ):
        deduped = _deduplicate_section(section_patches)
        best_patch = max(
            deduped,
            key=lambda p: (
                -priority_order.get(p.get("priority", "MEDIUM"), 1),
                len(p.get("evidence", "").split(",")),
            ),
        )
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append(best_patch.get("content", ""))
        lines.append("")

    lines.append("## Evolution Notes")
    lines.append(f"- Patches consolidated: {len(patches)}")
    lines.append(f"- Sections: {len(groups)}")
    lines.append(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)


def consolidate(
    patches: list[dict],
    existing_skill: str = "",
    use_llm: bool = True,
) -> str:
    if not patches:
        return existing_skill or "# E-Commerce Customer Service Action Prediction Skill\n\nNo patches generated.\n"

    if use_llm:
        try:
            print("  Running LLM consolidation...")
            return _llm_consolidate(patches, existing_skill)
        except Exception as exc:
            print(f"  LLM consolidation failed ({exc}), falling back to rule-based")
    return _rule_consolidate(patches, existing_skill)


def save_skill(content: str, output_dir: str, metadata: dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    skill_path = os.path.join(output_dir, "SKILL.md")
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(content)
    log_path = os.path.join(output_dir, "evolution_log.json")
    metadata.setdefault("timestamp", datetime.now().isoformat())
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  SKILL.md -> {skill_path}")
    print(f"  evolution_log.json -> {log_path}")
    return skill_path
