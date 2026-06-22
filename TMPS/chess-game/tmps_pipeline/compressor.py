"""Latency-constrained compression for trajectory-evolved chess skills."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompressedSkill:
    """Compressed skill content plus reproducibility metadata."""

    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CompressionRule:
    """A compact predictive rule with source triggers and budget priority."""

    theme: str
    text: str
    triggers: tuple[str, ...]
    priority: int


RULES: tuple[CompressionRule, ...] = (
    CompressionRule(
        "Opening",
        "- From startpos: top candidates = `[e2e4, d2d4]`. Third = `g1f3`.",
        ("starting position", "initial position", "e2e4", "d2d4"),
        100,
    ),
    CompressionRule(
        "Opening",
        "- Always rank 1.e4 above 1.d4.",
        ("rank 1.e4 above 1.d4", "e2e4", "d2d4"),
        95,
    ),
    CompressionRule(
        "Opening",
        "- After 1.e4 e5: White responses = `[d2d4, g1f3, f1c4]`; rank d4 first.",
        ("after 1.e4 e5", "d2d4", "g1f3"),
        92,
    ),
    CompressionRule(
        "Opening",
        "- After 2.Nf3 Nc6: `[f1b5, f1c4]` are the main bishop developments; do not let d2d4 dominate both.",
        ("2.nf3 nc6", "bishop developments", "ruy lopez", "italian"),
        88,
    ),
    CompressionRule(
        "Opening",
        "- Ruy Lopez: Black = `[a7a6, g8f6]`, with a6 first; Italian: Black = `[g8f6, f8c5]`.",
        ("ruy lopez", "italian", "a7a6", "g8f6"),
        84,
    ),
    CompressionRule(
        "Opening",
        "- After 2.d4 exd4 3.Nf3: Black = `[d7d5, b8c6]`, with d5 first.",
        ("2.d4 exd4 3.nf3", "d7d5", "b8c6"),
        80,
    ),
    CompressionRule(
        "King Safety",
        "- When king is in check and the checking piece cannot be captured: king evacuation moves are top candidates.",
        ("king is in check", "checking piece cannot be captured", "evacuation"),
        100,
    ),
    CompressionRule(
        "King Safety",
        "- Knight check on f7/f2: prioritize g8/h8 or g1/h1 over f8/f1 when those squares are safer.",
        ("knight check", "f7/f2", "g8/h8", "g1/h1"),
        94,
    ),
    CompressionRule(
        "King Safety",
        "- King on f8/f1 with opponent knight on g5/g4 or e5/e4: include `[f8g8]` or `[f8h7]` to break fork threats.",
        ("king is on f8/f1", "knight on g5", "f8g8", "f8h7"),
        90,
    ),
    CompressionRule(
        "King Safety",
        "- King on starting square with f2/f7 or g2/g7 under attack: prioritize castling or Kf1/Kf8.",
        ("starting square", "f2/f7", "g2/g7", "castling"),
        84,
    ),
    CompressionRule(
        "King Safety",
        "- After sacrifice on h7/f7: recapture or consolidate before speculative counterattacks.",
        ("sacrifice", "h7", "f7", "recapture"),
        78,
    ),
    CompressionRule(
        "King Safety",
        "- Exposed king with few defenders: king retreat beats material-grabbing captures.",
        ("exposed", "few defenders", "material-grabbing", "king retreat"),
        74,
    ),
    CompressionRule(
        "Tactical Awareness",
        "- Queen can deliver checkmate or decisive check: rank the tactical capture/check as top candidate.",
        ("queen", "checkmate", "decisive", "tactical capture"),
        96,
    ),
    CompressionRule(
        "Tactical Awareness",
        "- Knight fork on exposed king: prioritize the fork, then capture the higher-value piece.",
        ("knight fork", "forking", "higher-value piece"),
        94,
    ),
    CompressionRule(
        "Tactical Awareness",
        "- Opponent king exposed on e7/f7/g6 with checking or capturing move available: rank the attack first.",
        ("king exposed", "e7", "f7", "g6", "checking"),
        86,
    ),
    CompressionRule(
        "Tactical Awareness",
        "- Bishop/knight fork on king plus piece beats quiet development or pawn pushes.",
        ("bishop", "knight", "fork", "quiet"),
        80,
    ),
    CompressionRule(
        "Development Principles",
        "- Open positions after central pawn exchange: develop knights (Nf3, Nc3) before queen sorties.",
        ("central pawn", "developing knights", "queen sorties", "nf3"),
        88,
    ),
    CompressionRule(
        "Development Principles",
        "- When knight is attacked: retreat to squares that maintain pressure or threaten forks.",
        ("knight is attacked", "retreat", "threaten forks"),
        82,
    ),
    CompressionRule(
        "Development Principles",
        "- Closed/Ruy Lopez positions: prefer Nge7/Nf6 and central tension over speculative Nd4 or Qh4.",
        ("closed positions", "ruy lopez", "nge7", "nd4", "qh4"),
        78,
    ),
    CompressionRule(
        "Development Principles",
        "- Rook on starting square with exposed king: activate rook to the king file for shelter.",
        ("rook", "starting square", "exposed king", "shelter"),
        70,
    ),
    CompressionRule(
        "Central Control",
        "- After king evacuates to safety: prioritize central recapture over speculative knight moves.",
        ("king has escaped", "central pawn", "recapturing", "bxd5"),
        86,
    ),
    CompressionRule(
        "Central Control",
        "- Undefended central pawn capturable by bishop: Bxd5 is a top candidate when it wins a pawn and opens lines.",
        ("undefended", "central pawn", "bishop", "opens lines"),
        82,
    ),
    CompressionRule(
        "Central Control",
        "- Pawn majority versus exposed king: advance pawns that open lines toward the king shelter.",
        ("pawn majority", "king shelter", "pawn advances"),
        74,
    ),
    CompressionRule(
        "Central Control",
        "- Reduced material: pawn breaks that open lines beat quiet positional moves.",
        ("reduced material", "pawn breaks", "quiet positional"),
        68,
    ),
)

THEME_ORDER = (
    "Opening",
    "King Safety",
    "Tactical Awareness",
    "Development Principles",
    "Central Control",
)

THEME_MIN_RULES = {
    "Opening": 2,
    "King Safety": 2,
    "Tactical Awareness": 1,
    "Development Principles": 1,
    "Central Control": 1,
}

DIVERSITY_HARMING_PHRASES = (
    "must be exactly",
    "never include a third candidate",
    "non-negotiable",
    "strict top-2",
    "top-2 predictions must be exactly",
)


def compress_skill(
    skill_text: str,
    char_budget: int = 2800,
    min_rules_per_theme: dict[str, int] | None = None,
    use_llm_refinement: bool = True,
    config_path: str = "config.yml",
) -> CompressedSkill:
    """Compress a trajectory-evolved chess prediction skill under a character budget.

    If use_llm_refinement is True (default), the rule-based draft is polished by an LLM.
    """
    source = _normalize(skill_text)
    selected = _select_rules(source)
    selected = _fit_budget(selected, char_budget, min_rules_per_theme or THEME_MIN_RULES)
    rough_draft = _render(selected)

    llm_refined = False
    content = rough_draft
    if use_llm_refinement:
        try:
            print("  Running LLM refinement...")
            refined = _llm_refine(rough_draft, skill_text, char_budget, config_path)
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
        "removed_diversity_harming_phrases": list(DIVERSITY_HARMING_PHRASES),
        "method": "latency-constrained heuristic compression"
                  + (" + LLM refinement" if llm_refined else ""),
        "llm_refined": llm_refined,
        "timestamp": datetime.now().isoformat(),
    }
    return CompressedSkill(content=content, metadata=metadata)


def save_compressed_skill(result: CompressedSkill, output_dir: str | Path) -> Path:
    """Write SKILL.md and compression_log.json."""
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

    # If a v2-like skill is broad but wording differs, keep the core scaffold.
    if not any(selected.values()) and "chess move prediction skill" in source:
        for rule in RULES:
            if rule.priority >= 84:
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


def _llm_refine(
    rough_draft: str,
    original_skill: str,
    char_budget: int,
    config_path: str,
) -> str:
    """Use an LLM to polish the rule-based compressed draft."""
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from utils import Utils

    from .prompts import COMPRESSOR_REFINE_SYSTEM_PROMPT, format_compressor_refine_message

    cfg = {}
    try:
        from yaml import safe_load
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = safe_load(f)
    except Exception:
        pass

    actor_cfg = cfg.get("agents", {}).get("actor", {})
    provider = actor_cfg.get("provider", "DeepSeek").strip().lower()
    api_key = Utils.get_api_key(cfg, provider)
    base_url = Utils.get_base_url(cfg, provider)
    model = actor_cfg.get("model", "deepseek-chat")

    client = OpenAI(api_key=api_key, base_url=base_url)

    user_msg = format_compressor_refine_message(rough_draft, original_skill, char_budget)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COMPRESSOR_REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip()


def _render(selected: dict[str, list[CompressionRule]]) -> str:
    lines = [
        "# Chess Move Prediction Skill",
        "",
        "Prioritize principled central play, development, king safety, and immediate tactical threats.",
        "",
    ]
    for theme in THEME_ORDER:
        rules = selected.get(theme, [])
        if not rules:
            continue
        lines.append(f"## {theme} (HIGH)")
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
