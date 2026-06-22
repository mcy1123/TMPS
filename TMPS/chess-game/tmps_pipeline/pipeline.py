"""Trace2Skill pipeline orchestrator and CLI."""

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Optional

from .extractor import (
    PredictionEvent,
    build_summary,
    chunk_events,
    discover_run_dirs,
    extract_all_events,
    format_event_for_llm,
)
from .analyst import run_full_analysis
from .compressor import compress_skill, save_compressed_skill
from .consolidator import consolidate, save_skill


def _load_existing_skill(path: str) -> str:
    """Load SKILL.md content from a file path if it exists."""
    import os

    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def cmd_summary(args: argparse.Namespace) -> int:
    """Print trajectory statistics without evolving."""
    events = extract_all_events(args.traj_dir)
    if not events:
        print(f"No prediction events found in {args.traj_dir}")
        return 1

    summary = build_summary(events)
    print(f"Runs:         {summary['runs']}")
    print(f"Total events: {summary['total_events']}")
    print(f"Hits:         {summary['hits']}")
    print(f"Misses:       {summary['misses']}")
    print(f"Hit rate:     {summary['hit_rate']:.1%}")
    print(f"Forced moves: {summary['forced_moves']}")
    print(f"Rank dist:    {summary['rank_distribution']}")
    print()
    for phase, stats in summary["phases"].items():
        print(f"  {phase:>10}: {stats['total']:>3} events, {stats['hit_rate']:.1%} hit rate")

    chunks = chunk_events([e for e in events if e.event_type == "miss"])
    hits, misses = len([e for e in events if e.event_type == "hit"]), len([e for e in events if e.event_type == "miss"])
    print(f"\nMiss chunks: {len(chunks)} (from {misses} misses)")
    chunks_h = chunk_events([e for e in events if e.event_type == "hit"])
    print(f"Hit chunks:  {len(chunks_h)} (from {hits} hits)")

    if args.show_events:
        print("\n=== MISS Events ===")
        for e in [ev for ev in events if ev.event_type == "miss"]:
            print(format_event_for_llm(e))
            print()

    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    """Run the full 3-stage evolution pipeline."""
    print(f"=== Trace2Skill Evolution ===\nTrajectory dir: {args.traj_dir}")

    # Stage 1: Extract
    print("\n--- Stage 1: Extraction ---")
    events = extract_all_events(args.traj_dir)
    if not events:
        print("ERROR: No prediction events found")
        return 1
    summary = build_summary(events)
    print(f"  {summary['total_events']} events from {summary['runs']} runs")
    print(f"  Hit rate: {summary['hit_rate']:.1%} ({summary['hits']} hits / {summary['misses']} misses)")

    if args.dry_run:
        print("\n[Dry run] Extraction OK. Skipping LLM analysis.")
        return 0

    # Load existing skill
    existing_skill = _load_existing_skill(args.base_skill) if args.base_skill else ""
    if existing_skill:
        print(f"  Base skill loaded: {args.base_skill} ({len(existing_skill)} chars)")

    # Stage 2: Analyze
    error_patches, success_patches = run_full_analysis(
        events=events,
        config_path=args.config,
        existing_skill=existing_skill,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
    )

    # Stage 3: Consolidate
    print("\n--- Stage 3: Consolidation ---")
    all_patches = error_patches + success_patches
    print(f"  Total patches: {len(all_patches)} ({len(error_patches)} error + {len(success_patches)} success)")

    if args.save_intermediates and args.output:
        import os
        intermediate_dir = os.path.join(args.output, "intermediates")
        os.makedirs(intermediate_dir, exist_ok=True)
        with open(os.path.join(intermediate_dir, "error_patches.json"), "w", encoding="utf-8") as f:
            json.dump(error_patches, f, indent=2, ensure_ascii=False)
        with open(os.path.join(intermediate_dir, "success_patches.json"), "w", encoding="utf-8") as f:
            json.dump(success_patches, f, indent=2, ensure_ascii=False)
        print(f"  Intermediate patches saved to {intermediate_dir}")

    skill_content = consolidate(
        patches=all_patches,
        config_path=args.config,
        existing_skill=existing_skill,
        use_llm=not args.no_llm_consolidation,
    )

    # Save
    metadata = {
        "source_trajectories": args.traj_dir,
        "base_skill": args.base_skill,
        "events_analyzed": summary["total_events"],
        "hit_rate_at_evolution": summary["hit_rate"],
        "error_patches": len(error_patches),
        "success_patches": len(success_patches),
        "total_patches": len(all_patches),
        "config": args.config,
        "version": "1.0",
    }
    skill_path = save_skill(skill_content, args.output, metadata)
    print(f"\n=== Evolution complete ===")
    print(f"Skill: {skill_path}")
    print(f"Sections: {skill_content.count('## ')}")
    print(f"Size: {len(skill_content)} chars")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two SKILL.md versions."""
    a = _load_existing_skill(args.skill_a)
    b = _load_existing_skill(args.skill_b)
    if not a:
        print(f"ERROR: empty or missing file: {args.skill_a}")
        return 1
    if not b:
        print(f"ERROR: empty or missing file: {args.skill_b}")
        return 1

    print(f"Skill A: {args.skill_a} ({len(a)} chars)")
    print(f"Skill B: {args.skill_b} ({len(b)} chars)")
    print(f"\nA sections: {a.count('## ')}")
    print(f"B sections: {b.count('## ')}")

    # Simple line-level diff
    a_lines = set(a.split("\n"))
    b_lines = set(b.split("\n"))
    added = b_lines - a_lines
    removed = a_lines - b_lines
    if added:
        print(f"\nLines added in B ({len(added)}):")
        for line in sorted(added)[:20]:
            if line.strip():
                print(f"  + {line.strip()[:80]}")
    if removed:
        print(f"\nLines removed from A ({len(removed)}):")
        for line in sorted(removed)[:20]:
            if line.strip():
                print(f"  - {line.strip()[:80]}")
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    """Compress an evolved SKILL.md under a latency-oriented character budget."""
    skill_text = _load_existing_skill(args.input)
    if not skill_text:
        print(f"ERROR: empty or missing file: {args.input}")
        return 1

    result = compress_skill(
        skill_text,
        char_budget=args.char_budget,
        use_llm_refinement=not args.no_llm_refinement,
        config_path=args.config,
    )
    skill_path = save_compressed_skill(result, args.output)
    print("=== Compression complete ===")
    print(f"Input:  {args.input} ({result.metadata['input_chars']} chars)")
    print(f"Output: {skill_path} ({result.metadata['output_chars']} chars)")
    print(f"Budget: {args.char_budget} chars")
    print(f"Ratio:  {result.metadata['compression_ratio']:.2%}")
    print(f"Themes: {result.metadata['themes']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace2Skill: Evolve chess prediction skills from execution trajectories"
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # evolve
    ev = sub.add_parser("evolve", help="Run the full evolution pipeline")
    ev.add_argument("--traj-dir", required=True, help="Directory containing speculative run subdirectories")
    ev.add_argument("--output", required=True, help="Output directory for SKILL.md")
    ev.add_argument("--base-skill", default="", help="Path to existing SKILL.md for incremental evolution")
    ev.add_argument("--config", default="config.yml", help="Path to config YAML")
    ev.add_argument("--chunk-size", type=int, default=15, help="Events per analyst chunk")
    ev.add_argument("--max-workers", type=int, default=4, help="Max parallel analyst workers")
    ev.add_argument("--dry-run", action="store_true", help="Extract events without calling LLMs")
    ev.add_argument("--save-intermediates", action="store_true", help="Save intermediate patch proposals")
    ev.add_argument("--no-llm-consolidation", action="store_true", help="Use rule-based consolidation instead of LLM")

    # summary
    sm = sub.add_parser("summary", help="Show trajectory statistics")
    sm.add_argument("--traj-dir", required=True, help="Directory containing speculative run subdirectories")
    sm.add_argument("--show-events", action="store_true", help="Print individual miss events")

    # compare
    cp = sub.add_parser("compare", help="Compare two SKILL.md versions")
    cp.add_argument("--skill-a", required=True, help="First SKILL.md")
    cp.add_argument("--skill-b", required=True, help="Second SKILL.md")

    # compress
    cm = sub.add_parser("compress", help="Compress an evolved SKILL.md under a latency budget")
    cm.add_argument("--input", required=True, help="Input evolved SKILL.md")
    cm.add_argument("--output", required=True, help="Output directory for compressed SKILL.md")
    cm.add_argument("--char-budget", type=int, default=2800, help="Maximum output characters")
    cm.add_argument("--no-llm-refinement", action="store_true", help="Skip LLM refinement (use rule-based draft directly)")
    cm.add_argument("--config", default="config.yml", help="Path to config YAML")

    args = parser.parse_args()
    if args.command == "summary":
        sys.exit(cmd_summary(args))
    elif args.command == "evolve":
        sys.exit(cmd_evolve(args))
    elif args.command == "compare":
        sys.exit(cmd_compare(args))
    elif args.command == "compress":
        sys.exit(cmd_compress(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
