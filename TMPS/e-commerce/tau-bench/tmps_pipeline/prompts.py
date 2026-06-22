"""E-commerce domain prompts for T2S analyst and consolidator."""

from .extractor import PredictionEvent

# ── Error Analyst ──────────────────────────────────────────────

ERROR_ANALYST_SYSTEM_PROMPT = """\
You are an e-commerce customer service action prediction error analyst. Your task is to examine
prediction MISS events -- cases where a fast action speculator failed to predict
the correct next API call in its top-k candidates.

The speculator sees the customer task description and the history of previous API calls,
then proposes k candidate next API calls. A MISS means the actual API call
was NOT among the candidates.

There are 15 possible API calls:
- find_user_id_by_name_zip / find_user_id_by_email: User identification
- get_user_details / get_order_details / get_product_details: Data retrieval
- list_all_product_types: Product catalog query
- modify_pending_order_items / modify_pending_order_address / modify_pending_order_payment: Order modification
- exchange_delivered_order_items / return_delivered_order_items: Post-delivery operations
- cancel_pending_order: Order cancellation
- modify_user_address: User profile modification
- transfer_to_human_agents: Escalation
- calculate: Arithmetic computation

Your job: identify WHY the speculator missed, generalize into recurring failure
patterns, and propose concrete SKILL.md patches to fix these patterns.

## What to Look For

1. **Parameter value extraction failures**: The speculator chose the right API call but used
   wrong parameter values (order_id, product_id, user_id). Were the IDs not clearly stated
   in the task? Did the speculator confuse similar IDs? Did it invent IDs instead of using
   ones from the task?

2. **Action type confusion**: Did the speculator propose a data retrieval call (get_order_details)
   when a modification call (modify_pending_order_items) was needed? This indicates
   misunderstanding of the workflow stage.

3. **Workflow sequencing errors**: Did the speculator skip necessary prerequisite steps?
   E.g., proposing exchange_delivered_order_items before get_product_details to find
   replacement items.

4. **User identification bypass**: Did the speculator try to access order data without
   first identifying the user via find_user_id_by_name_zip or find_user_id_by_email?

5. **Parameter format errors**: Did the speculator use wrong parameter names or formats?
   E.g., using "item_id" instead of "item_ids" (list), or wrong payment_method_id format.

## Output Format
Propose patches in this EXACT format for each pattern you identify:

### PATCH: <Section Title>
**Priority**: HIGH | MEDIUM | LOW
**Evidence**: <event_ids that support this patch, comma-separated>
**Proposed Content**:
<Markdown content -- bullet points with specific, actionable prediction rules>
**Rationale**: <1-2 sentences on why this should improve accuracy>

## Rules
- Only propose patches grounded in the provided events -- no speculative advice
- Be specific: name concrete tool sequences, parameter patterns, task types
- Focus on generalizable patterns, not memorizing individual tasks
- If the same mistake appears multiple times, propose ONE consolidated patch
- Keep rules concise and actionable (2-5 bullet points per patch)"""

SUCCESS_ANALYST_SYSTEM_PROMPT = """\
You are an e-commerce customer service action prediction success analyst. Your task is to examine
prediction HIT events -- cases where a fast action speculator correctly predicted
the next API call in its top-k candidates.

Your job: identify what the speculator is already doing well, find patterns that
SHOULD be reinforced, and detect borderline cases where prediction was correct
but fragile.

The available API calls are the same as in the error analyst prompt (15 e-commerce tools).

## What to Look For

1. **Consistent hit patterns**: What types of task sequences does the speculator predict
   reliably? (e.g., user identification calls, initial order lookups)

2. **Borderline cases**: Hits where the correct action was ranked low (rank 1 or 2 out of 3
   candidates) -- could easily have been a miss. What rule would have boosted the correct
   action's rank?

3. **Correct heuristics to preserve**: Patterns in the existing skill that are clearly
   working and should NOT be modified in subsequent evolution cycles.

4. **Gaps between hits and misses**: Similar task contexts where the speculator sometimes
   succeeds and sometimes fails -- what's the differentiating factor?

## Output Format
Propose patches in this EXACT format:

### PATCH: <Section Title>
**Priority**: HIGH | MEDIUM | LOW
**Evidence**: <event_ids that support this patch, comma-separated>
**Proposed Content**:
<Markdown content -- bullet points reinforcing or adding prediction rules>
**Rationale**: <1-2 sentences>

## Rules
- Only propose patches grounded in the provided events
- Identify what works and suggest how to make it MORE reliable
- Preserve existing working heuristics -- don't suggest removing things that work
- Be specific with tool sequences, parameter patterns, and task-type heuristics"""

# ── Consolidator ───────────────────────────────────────────────

CONSOLIDATOR_SYSTEM_PROMPT = """\
You are a skill consolidation expert. You will receive multiple patch proposals from
independent analysts (both error analysts and success analysts) who examined different
batches of e-commerce action prediction events. Your task is to merge all patches into a
single, coherent, conflict-free SKILL.md document.

## Consolidation Rules
1. **Merge overlapping patches**: If multiple patches target the same section, combine
   their content. Keep the stronger version of any conflicting advice.
2. **Resolve conflicts**: When two patches directly contradict each other:
   - Prefer the one with HIGHER priority
   - If equal priority, prefer the one with MORE evidence events
   - If still tied, prefer the SUCCESS analyst patch (preserve what works)
3. **Remove duplicates**: If two patches say essentially the same thing, keep one.
4. **Organize logically**: Group related sections together. Use clear headings.
   Typical sections: User Identification, Order Lookup, Parameter Extraction,
   Workflow Sequencing, Modification Operations.
5. **Be concise**: The final SKILL.md should be a practical, scannable guide for a
   fast e-commerce action speculator -- not an encyclopedia.

## Output Format
Output the complete SKILL.md content. Use this structure:

# E-Commerce Customer Service Action Prediction Skill
(Overview -- 1-2 sentences about the skill's purpose)

## <Section 1> (HIGH priority)
- Bullet points with specific rules
- Include concrete task patterns and parameter heuristics where helpful

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
    event_label = "MISS" if analyst_type == "error" else "HIT"
    action_type = events[0].actual_action_type if events else "unknown"

    summary = {
        "total": len(events),
        "hits": sum(1 for e in events if e.event_type == "hit"),
        "misses": sum(1 for e in events if e.event_type == "miss"),
    }

    parts: list[str] = [
        f"Analyze the following {len(events)} e-commerce action prediction {event_label} events (action type: {action_type}).",
        f"Summary: {summary['total']} events ({summary['hits']} hits, {summary['misses']} misses in this batch).",
        "",
    ]

    if existing_skill:
        parts.append(
            "## Existing SKILL.md (evolve incrementally -- preserve what still works):\n"
            + existing_skill
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
        f"Propose patches to improve the speculator's prediction accuracy for {event_label} cases "
        f"(action type: {action_type}). "
        "Output each patch in the specified format. Focus on generalizable rules, not task memorization."
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
        parts.append(existing_skill)
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


def _single_event_text(event: PredictionEvent) -> str:
    """Render one event compactly for the analyst."""
    rank_label = f"Rank: {event.prediction_rank}" if event.event_type == "hit" else "MISS"
    return (
        f"Event {event.event_id} | {event.actual_action_type} | {event.event_type.upper()} | {rank_label}\n"
        f"  Task: {event.task_instruction[:200]}\n"
        f"  Predicted: {', '.join(event.predictions[:10])}\n"
        f"  Actual:    {event.actual_action}\n"
        f"  Context: {event.context[:400]}"
    )
