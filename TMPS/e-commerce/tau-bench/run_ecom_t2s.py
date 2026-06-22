"""E-commerce T2S experiment: baseline → evolve → evaluate.

Uses first 80 TEST tasks for evolution (skill training), last 35 for evaluation.
"""

import json
import os
import sys
import time
from datetime import datetime

from t2s.extractor import (
    load_tasks_from_module, extract_events_from_tasks,
    build_summary, split_events,
)
from t2s.speculator import EcommerceSpeculator
from t2s.pipeline import evolve_skill, run_experiment


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-evolve", action="store_true", help="Skip evolution, just evaluate")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--evolution-tasks", type=int, default=80)
    p.add_argument("--eval-tasks", type=int, default=35)
    p.add_argument("--tasks-path", default="tau_bench/envs/retail/tasks_test.py")
    p.add_argument("--base-skill", default="./skills/base_skill.md")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    tasks_path = args.tasks_path
    n_evolve = args.evolution_tasks
    n_eval = args.eval_tasks

    print("=" * 60)
    print("E-commerce T2S Experiment")
    print(f"  Model: {args.model}, k={args.k}")
    print(f"  Evolution tasks: first {n_evolve}")
    print(f"  Evaluation tasks: last {n_eval}")
    print("=" * 60)

    # Load all tasks
    all_tasks = load_tasks_from_module(tasks_path)
    print(f"\nLoaded {len(all_tasks)} total tasks")

    if not args.skip_evolve:
        # Step 1: Run no-skill baseline on evolution set
        print(f"\n{'=' * 60}")
        print(f"Step 1: No-Skill baseline on {n_evolve} evolution tasks")
        print(f"{'=' * 60}")

        events_base, summary_base = run_experiment(
            tasks_path=tasks_path,
            model_name=args.model,
            skill_file=None,
            max_tasks=n_evolve,
            k=args.k,
            start_offset=0,
        )

        # Save baseline summary
        os.makedirs("results", exist_ok=True)
        with open("results/baseline_evolution.json", "w") as f:
            json.dump(summary_base, f, indent=2)
        print(f"Baseline saved to results/baseline_evolution.json")

        # Step 2: Evolve skill
        print(f"\n{'=' * 60}")
        print(f"Step 2: T2S Skill Evolution")
        print(f"{'=' * 60}")

        skill_path = evolve_skill(
            tasks_path=tasks_path,
            model_name=args.model,
            base_skill_path=args.base_skill,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            n_evolution_tasks=n_evolve,
            k=args.k,
            events=events_base,
        )

        print(f"\nEvolved skill: {skill_path}")
    else:
        # Find latest evolved skill
        skill_dirs = sorted([
            d for d in os.listdir("skills")
            if d.startswith("v1_evolved_")
        ], reverse=True)
        if not skill_dirs:
            print("No evolved skill found! Run without --skip-evolve first.")
            sys.exit(1)
        skill_dir = skill_dirs[0]
        skill_path = f"skills/{skill_dir}/SKILL.md"
        print(f"Using existing skill: {skill_path}")

    # Step 3: Evaluate on held-out tasks
    print(f"\n{'=' * 60}")
    print(f"Step 3: Evaluation on last {n_eval} held-out tasks")
    print(f"{'=' * 60}")

    # No-Skill on evaluation set
    print("\n--- No-Skill Evaluation ---")
    events_noskill, summary_noskill = run_experiment(
        tasks_path=tasks_path,
        model_name=args.model,
        skill_file=None,
        max_tasks=n_eval,
        k=args.k,
        start_offset=n_evolve,
    )
    with open("results/eval_noskill.json", "w") as f:
        json.dump(summary_noskill, f, indent=2)

    # Base Skill on evaluation set
    print("\n--- Base Skill Evaluation ---")
    events_baseskill, summary_baseskill = run_experiment(
        tasks_path=tasks_path,
        model_name=args.model,
        skill_file=args.base_skill,
        max_tasks=n_eval,
        k=args.k,
        start_offset=n_evolve,
    )
    with open("results/eval_baseskill.json", "w") as f:
        json.dump(summary_baseskill, f, indent=2)

    # Evolved Skill on evaluation set
    print("\n--- Evolved Skill Evaluation ---")
    events_skill, summary_skill = run_experiment(
        tasks_path=tasks_path,
        model_name=args.model,
        skill_file=skill_path,
        max_tasks=n_eval,
        k=args.k,
        start_offset=n_evolve,
    )
    with open("results/eval_skill.json", "w") as f:
        json.dump(summary_skill, f, indent=2)

    # Print comparison
    print(f"\n{'=' * 60}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 60}")

    for name, s in [("No-Skill", summary_noskill), ("Base Skill", summary_baseskill), ("Evolved Skill", summary_skill)]:
        print(f"\n{name}:")
        print(f"  Overall hit_rate: {s['hit_rate']:.1%}")
        print(f"  n_events: {s['n_events']}")
        print(f"  By action type:")
        for t, rate in sorted(s['by_action_type_hit_rate'].items(), key=lambda x: -x[1]):
            n = s['by_action_type'].get(t, 0)
            print(f"    {t}: {rate:.1%} (n={n})")

    # Deltas
    delta_base = summary_baseskill['hit_rate'] - summary_noskill['hit_rate']
    delta_evolved = summary_skill['hit_rate'] - summary_noskill['hit_rate']
    print(f"\nΔ Base Skill vs No-Skill: {delta_base:+.1%}")
    print(f"Δ Evolved Skill vs No-Skill: {delta_evolved:+.1%}")

    # Save final comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model, "k": args.k,
            "evolution_tasks": n_evolve, "eval_tasks": n_eval,
            "skill_path": skill_path,
        },
        "no_skill": summary_noskill,
        "base_skill": summary_baseskill,
        "evolved_skill": summary_skill,
        "delta_base_vs_noskill": delta_base,
        "delta_evolved_vs_noskill": delta_evolved,
    }
    with open("results/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to results/comparison.json")
    return comparison


if __name__ == "__main__":
    main()
