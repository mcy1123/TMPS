"""T2S evolution pipeline for e-commerce: extract → analyze → consolidate → evaluate."""

import json
import os
import sys
import time
from datetime import datetime

from .extractor import (
    PredictionEvent, extract_events_from_tasks, load_tasks_from_module,
    build_summary, split_events, chunk_events,
)
from .speculator import EcommerceSpeculator


def compute_hit_rates(events: list[PredictionEvent]) -> dict:
    summary = build_summary(events)
    return summary


def evolve_skill(
    tasks_path: str = "",
    model_name: str = "deepseek-chat",
    base_skill_path: str = "./skills/base_skill.md",
    output_dir: str = None,
    max_workers: int = 4,
    n_evolution_tasks: int = 80,
    k: int = 3,
    events: list[PredictionEvent] = None,
) -> str:
    """Full T2S evolution: extract events from first n tasks, analyze, consolidate.

    If `events` is provided (pre-evaluated), skip extraction and baseline evaluation.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./skills/v1_evolved_{timestamp}"

    if events is not None:
        print(f"[Evolve] Using {len(events)} pre-evaluated events")
        summary = build_summary(events)
        print(f"[Evolve] Baseline hit_rate: {summary['hit_rate']:.1%}")
        print(f"[Evolve] By action type: {json.dumps(summary['by_action_type_hit_rate'], indent=2)}")
    else:
        print(f"[Evolve] Loading tasks from {tasks_path}")
        tasks = load_tasks_from_module(tasks_path)
        evolution_tasks = tasks[:n_evolution_tasks]
        events = extract_events_from_tasks(evolution_tasks)
        print(f"[Evolve] {len(evolution_tasks)} tasks → {len(events)} prediction events")

        print(f"[Evolve] Running no-skill baseline on {len(events)} events...")
        spec = EcommerceSpeculator(model_name=model_name, k=k, skill_file=None)
        events = spec.evaluate_batch(events)
        summary = build_summary(events)
        print(f"[Evolve] Baseline hit_rate: {summary['hit_rate']:.1%}")
        print(f"[Evolve] By action type: {json.dumps(summary['by_action_type_hit_rate'], indent=2)}")

    hits, misses = split_events(events)
    print(f"[Evolve] {len(hits)} hits, {len(misses)} misses")

    # Read base skill
    base_skill = ""
    if base_skill_path and os.path.exists(base_skill_path):
        with open(base_skill_path) as f:
            base_skill = f.read()

    # Stage 2: Parallel analysis
    print(f"[Evolve] Stage 2: Parallel multi-agent analysis...")
    from .analyst import AnalystRunner
    from .consolidator import consolidate, save_skill

    analyst = AnalystRunner(model_name=model_name, max_workers=max_workers)
    success_patches, error_patches = analyst.run_parallel_analysis(hits, misses, base_skill)
    all_patches = success_patches + error_patches
    print(f"[Evolve] Generated {len(all_patches)} patches ({len(success_patches)} success, {len(error_patches)} error)")

    # Stage 3: Consolidate
    print(f"[Evolve] Stage 3: Consolidating patches into SKILL.md...")
    skill_content = consolidate(all_patches, base_skill, use_llm=True)
    skill_path = save_skill(skill_content, output_dir, {
        "n_events": len(events),
        "n_patches": len(all_patches),
        "baseline_hit_rate": summary["hit_rate"],
        "model": model_name,
        "k": k,
    })

    print(f"[Evolve] Skill saved to {skill_path}")
    return skill_path


def run_experiment(
    tasks_path: str,
    model_name: str = "deepseek-chat",
    skill_file: str = None,
    max_tasks: int = -1,
    k: int = 3,
    start_offset: int = 0,
) -> tuple[list[PredictionEvent], dict]:
    """Run prediction accuracy experiment on tasks."""
    tasks = load_tasks_from_module(tasks_path)
    if max_tasks > 0:
        tasks = tasks[start_offset:start_offset + max_tasks]
    events = extract_events_from_tasks(tasks)
    print(f"[Experiment] {len(tasks)} tasks, {len(events)} events")

    spec = EcommerceSpeculator(model_name=model_name, k=k, skill_file=skill_file)
    t0 = time.time()
    events = spec.evaluate_batch(events)
    elapsed = time.time() - t0

    summary = build_summary(events)
    print(f"[Experiment] Done in {elapsed:.0f}s")
    print(f"[Experiment] Overall hit_rate: {summary['hit_rate']:.1%}")
    print(f"[Experiment] By action type:")
    for t, rate in sorted(summary['by_action_type_hit_rate'].items(),
                          key=lambda x: -x[1]):
        n = summary['by_action_type'].get(t, 0)
        print(f"  {t}: {rate:.1%} (n={n})")

    return events, summary
