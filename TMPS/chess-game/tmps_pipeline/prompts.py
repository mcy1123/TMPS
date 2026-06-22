"""Prompt templates for Trace2Skill chess prediction skill evolution."""

from .extractor import PredictionEvent

# ── Error Analyst ──────────────────────────────────────────────

ERROR_ANALYST_SYSTEM_PROMPT = """\
You are a chess prediction error analyst. Your task is to examine prediction MISS events —
cases where a fast chess move predictor (the "Speculator") failed to predict the correct move
in its top-k candidates.

The Speculator sees the board position and the full legal move list, then ranks k candidate
moves by likelihood. A MISS means the actual move played was NOT among the candidates.

Your job: identify WHY the speculator missed, generalize into recurring failure patterns,
and propose concrete SKILL.md patches to fix these patterns.

## Chess Context
- Moves are in UCI format: [e2e4], [g1f3], [e7e5], [O-O] (castling)
- The speculator predicts for the side-to-move (either White or Black)
- Legal moves are always provided — the speculator never needs to validate legality

## What to Look For
1. **Opening theory gaps**: Standard responses the speculator failed to anticipate
   (e.g., after 1.e4, failing to predict [c7c5] Sicilian, [e7e5] Open Game, [e7e6] French)
2. **Tactical blindness**: Missed checks, captures, direct threats — moves that are
   tactically "forcing" and thus highly likely to be played
3. **Recapture failures**: When a piece is captured, the obvious recapture was missed
4. **Developing move neglect**: Natural development moves (knights, bishops) that were
   ranked below less-likely alternatives
5. **King safety / defensive moves**: When the king is under attack, failing to predict
   the necessary defensive response
6. **Endgame simplification**: In endgames, common patterns like pawn pushes, king
   activation, or trade sequences were missed

## Output Format
Propose patches in this EXACT format for each pattern you identify:

### PATCH: <Section Title>
**Priority**: HIGH | MEDIUM | LOW
**Evidence**: <event_ids that support this patch, comma-separated>
**Proposed Content**:
<Markdown content — bullet points with specific, actionable prediction rules>
**Rationale**: <1-2 sentences on why this should improve accuracy>

## Rules
- Only propose patches grounded in the provided events — no speculative advice
- Be specific: name concrete move sequences (e.g., "After 1.e4, predict [c7c5] and [e7e5]")
- Focus on generalizable patterns, not memorizing individual positions
- If the same mistake appears multiple times, propose ONE consolidated patch
- Never suggest moves outside the legal move list
- Keep rules concise and actionable (2-5 bullet points per patch)"""

SUCCESS_ANALYST_SYSTEM_PROMPT = """\
You are a chess prediction success analyst. Your task is to examine prediction HIT events —
cases where a fast chess move predictor (the "Speculator") correctly predicted the move
in its top-k candidates.

Your job: identify what the speculator is already doing well, find patterns that SHOULD
be reinforced, and detect borderline cases where prediction was correct but fragile.

## Chess Context
- Moves are in UCI format: [e2e4], [g1f3], etc.
- The speculator predicts for the side-to-move
- Legal moves are always provided

## What to Look For
1. **Consistent hit patterns**: What types of positions does the speculator predict well?
   (e.g., early opening moves, recaptures, obvious checks)
2. **Borderline cases**: Hits where the correct move was ranked low (rank 1 or 2) —
   could easily have been a miss. What rule would have helped?
3. **Correct heuristics to preserve**: Sections of the existing skill that are clearly
   working and should NOT be modified
4. **Gaps between hits and misses**: Similar positions where the speculator sometimes
   succeeds and sometimes fails — what's the differentiating factor?

## Output Format
Propose patches in this EXACT format:

### PATCH: <Section Title>
**Priority**: HIGH | MEDIUM | LOW
**Evidence**: <event_ids that support this patch, comma-separated>
**Proposed Content**:
<Markdown content — bullet points reinforcing or adding prediction rules>
**Rationale**: <1-2 sentences>

## Rules
- Only propose patches grounded in the provided events
- Identify what works and suggest how to make it MORE reliable
- Preserve existing working heuristics — don't suggest removing things that work
- Be specific with move sequences"""

# ── Consolidator ───────────────────────────────────────────────

CONSOLIDATOR_SYSTEM_PROMPT = """\
You are a skill consolidation expert. You will receive multiple patch proposals from
independent analysts (both error analysts and success analysts) who examined different
batches of chess prediction events. Your task is to merge all patches into a single,
coherent, conflict-free SKILL.md document.

## Consolidation Rules
1. **Merge overlapping patches**: If multiple patches target the same section, combine
   their content. Keep the stronger version of any conflicting advice.
2. **Resolve conflicts**: When two patches directly contradict each other:
   - Prefer the one with HIGHER priority
   - If equal priority, prefer the one with MORE evidence events
   - If still tied, prefer the SUCCESS analyst patch (preserve what works)
3. **Remove duplicates**: If two patches say essentially the same thing, keep one.
4. **Organize logically**: Group related sections together. Use clear headings.
5. **Be concise**: The final SKILL.md should be a practical, scannable guide for a
   fast chess move predictor — not an encyclopedia.

## Output Format
Output the complete SKILL.md content. Use this structure:

# Chess Move Prediction Skill
(Overview — 1-2 sentences about the skill's purpose)

## <Section 1> (HIGH priority)
- Bullet points with specific rules
- Include concrete move sequences where helpful

## <Section 2> (MEDIUM priority)
...

## Evolution Notes
- Source: <summary of trajectories analyzed>
- Patches consolidated: <count>
- Date: <current date>"""

# ── User Message Templates ─────────────────────────────────────

def format_analyst_user_message(
    events: list[PredictionEvent],
    analyst_type: str,
    existing_skill: str = "",
) -> str:
    """Build the user message for an analyst LLM call."""
    event_type = "MISS" if analyst_type == "error" else "HIT"
    phase = events[0].phase if events else "unknown"
    summary = {
        "total": len(events),
        "hits": sum(1 for e in events if e.event_type == "hit"),
        "misses": sum(1 for e in events if e.event_type == "miss"),
    }

    parts: list[str] = [
        f"Analyze the following {len(events)} chess prediction {event_type} events from the {phase} phase.",
        f"Summary: {summary['total']} events ({summary['hits']} hits, {summary['misses']} misses in this batch).",
        "",
    ]

    if existing_skill:
        parts.append(
            "## Existing SKILL.md (evolve incrementally — preserve what still works):\n"
            + existing_skill[:3000]
        )
        parts.append("")
        parts.append("---")
        parts.append("")

    parts.append("## Events to Analyze:")
    for event in events:
        parts.append(_single_event_text(event))
        parts.append("")

    parts.append("## Instructions")
    parts.append(
        f"Propose patches to improve the speculator's prediction accuracy for {event_type} cases in the {phase} phase. "
        "Output each patch in the specified format. Focus on generalizable rules, not position memorization."
    )

    return "\n".join(parts)


def format_consolidator_user_message(
    patches: list[dict],
    existing_skill: str = "",
) -> str:
    """Build the user message for the consolidator LLM call."""
    parts: list[str] = [
        f"Consolidate the following {len(patches)} patch proposals into a single SKILL.md.",
        "",
    ]

    if existing_skill:
        parts.append("## Existing SKILL.md (base for incremental evolution):\n")
        parts.append(existing_skill[:2000])
        parts.append("")
        parts.append("---")
        parts.append("")

    parts.append("## Patch Proposals:")
    for i, patch in enumerate(patches, 1):
        parts.append(f"### Patch {i}")
        parts.append(f"**Source**: {patch.get('source', 'unknown')} analyst")
        parts.append(f"**Priority**: {patch.get('priority', 'MEDIUM')}")
        parts.append(f"**Section**: {patch.get('section', 'General')}")
        parts.append(f"**Evidence**: {patch.get('evidence', 'N/A')}")
        parts.append(f"\n{patch.get('content', '')}")
        parts.append(f"\n**Rationale**: {patch.get('rationale', '')}")
        parts.append("\n---\n")

    parts.append("Output the consolidated SKILL.md now.")

    return "\n".join(parts)


# ── Compressor Refinement ──────────────────────────────────────

COMPRESSOR_REFINE_SYSTEM_PROMPT = """\
You are a skill compression editor. You receive a rule-based compressed chess prediction
skill (rough draft) and your job is to polish it into a clean, professional SKILL.md.

The skill is used as a prompt prefix for a fast chess move predictor (Speculator). It must
be concise (fits within a strict character budget) yet precise enough to guide accurate
top-k move prediction.

## Refinement Rules
1. **Improve section organization**: Add descriptive subsection headings that indicate
   the specific scenario (e.g., "## Opening — Initial Position (HIGH)" instead of just
   "## Opening (HIGH)"). Group related rules under the same subsection.
2. **Refine wording**: Make each rule crisp and unambiguous. Prefer concrete UCI move
   examples over vague descriptions. Remove filler words.
3. **Merge similar rules**: If multiple rules cover the same scenario, merge them into
   one concise rule. Avoid redundancy.
4. **Respect the budget**: The final output MUST be ≤ the specified character limit.
   If you need to trim, drop the least impactful rules first (generic advice over
   concrete move-sequence rules).
5. **Preserve essential patterns**: Do not remove concrete opening sequences (e.g.,
   "After 1.e4 e5: White = [d2d4, g1f3, f1c4]"), tactical triggers, or king safety
   heuristics. These are the highest-value rules.
6. **No new rules**: Only refine what's in the draft. Do not invent new prediction
   rules or add examples not present in the source material.
7. **Keep all sections tagged (HIGH)**: The priority tag helps the Speculator weight
   these rules appropriately.

## Output Format
Output the complete, polished SKILL.md content. Use exactly this structure:

# Chess Move Prediction Skill

<one-line overview>

## <Theme> — <Scenario> (HIGH)

- <rule>
- <rule>

## <Theme> — <Scenario> (HIGH)
...
"""


def format_compressor_refine_message(
    rough_draft: str,
    original_skill: str,
    char_budget: int,
) -> str:
    """Build the user message for the compression refinement LLM call."""
    parts: list[str] = [
        f"Polish the following rule-based compressed skill draft into a professional SKILL.md.",
        f"",
        f"**Character budget**: {char_budget} characters (strict — the output MUST NOT exceed this).",
        f"",
    ]

    if original_skill:
        parts.append("## Original Full Skill (for context — reference phrasing and structure):")
        parts.append("")
        parts.append(original_skill[:2000])
        parts.append("")
        parts.append("---")
        parts.append("")

    parts.append("## Rough Draft (rule-based compressed — refine this):")
    parts.append("")
    parts.append(rough_draft)
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("Output the polished SKILL.md now. Remember: stay within the character budget, "
                 "improve subsection headings, refine wording, merge redundant rules, "
                 "but preserve all essential prediction patterns.")

    return "\n".join(parts)


def _single_event_text(event: PredictionEvent) -> str:
    """Render one event compactly for the analyst."""
    rank_label = f"Rank: {event.prediction_rank}" if event.event_type == "hit" else "MISS"
    return (
        f"Event {event.event_id} | {event.player_role} | {event.event_type.upper()} | {rank_label} | {event.num_legal_moves} legal moves\n"
        f"  Predicted: {', '.join(event.predictions[:10])}\n"
        f"  Actual:    {event.actual_move}\n"
        f"  FEN: {event.board_fen}"
    )
