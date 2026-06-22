"""Stage 1: Extract prediction events from speculative chess trajectories."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PredictionEvent:
    """A single prediction attempt from one step of one chess game."""

    event_id: str
    event_type: str  # "hit" or "miss"
    player_role: str  # "White" or "Black"
    step_number: int
    board_fen: str
    compact_observation: str
    legal_moves: list[str]
    predictions: list[str]
    actual_move: str
    prediction_rank: int  # 0-based index of hit, -1 for miss
    is_forced: bool  # True if only 1 legal move

    @property
    def phase(self) -> str:
        if self.step_number <= 10:
            return "opening"
        elif self.step_number <= 40:
            return "middlegame"
        return "endgame"

    @property
    def num_legal_moves(self) -> int:
        return len(self.legal_moves)


def discover_run_dirs(base_dir: str) -> list[str]:
    """Find all run directories containing stepsinfo.json under base_dir."""
    if not os.path.exists(base_dir):
        return []
    run_dirs = []
    for root, _, files in os.walk(base_dir, followlinks=True):
        if "stepsinfo.json" in files:
            run_dirs.append(root)
    return sorted(run_dirs)


def load_events_from_run(run_dir: str) -> list[PredictionEvent]:
    """Parse one run's stepsinfo.json into prediction events."""
    path = os.path.join(run_dir, "stepsinfo.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        steps = json.load(f)

    run_name = os.path.basename(run_dir)
    events: list[PredictionEvent] = []
    for key in sorted(steps.keys(), key=lambda x: int(x)):
        step = steps[key]
        predictions = step.get("current_pred", [])
        if not predictions:
            continue
        actual = step.get("current_move")
        if not actual:
            continue

        is_hit = actual in predictions
        events.append(
            PredictionEvent(
                event_id=f"{run_name}/step_{key}",
                event_type="hit" if is_hit else "miss",
                player_role=step.get("player_role", "Unknown"),
                step_number=int(key),
                board_fen=step.get("board_fen", ""),
                compact_observation=step.get("compact_observation", ""),
                legal_moves=step.get("legal_moves", []),
                predictions=predictions,
                actual_move=actual,
                prediction_rank=predictions.index(actual) if is_hit else -1,
                is_forced=len(step.get("legal_moves", [])) == 1,
            )
        )
    return events


def extract_all_events(base_dir: str) -> list[PredictionEvent]:
    """Aggregate all prediction events from all runs under base_dir."""
    all_events: list[PredictionEvent] = []
    for run_dir in discover_run_dirs(base_dir):
        all_events.extend(load_events_from_run(run_dir))
    return sorted(all_events, key=lambda e: (e.step_number, e.event_id))


def split_events(
    events: list[PredictionEvent],
) -> tuple[list[PredictionEvent], list[PredictionEvent]]:
    """Split into hits and misses."""
    hits = [e for e in events if e.event_type == "hit"]
    misses = [e for e in events if e.event_type == "miss"]
    return hits, misses


def build_summary(events: list[PredictionEvent]) -> dict[str, Any]:
    """Compute aggregate statistics for a set of prediction events."""
    hits, misses = split_events(events)
    total = len(events)
    hit_rate = len(hits) / total if total else 0.0

    phase_stats: dict[str, dict[str, Any]] = {}
    for phase in ("opening", "middlegame", "endgame"):
        phase_events = [e for e in events if e.phase == phase]
        phase_hits = [e for e in phase_events if e.event_type == "hit"]
        phase_stats[phase] = {
            "total": len(phase_events),
            "hits": len(phase_hits),
            "misses": len(phase_events) - len(phase_hits),
            "hit_rate": len(phase_hits) / len(phase_events) if phase_events else 0.0,
        }

    # Rank distribution for hits
    rank_counts: dict[int, int] = {}
    for e in hits:
        rank_counts[e.prediction_rank] = rank_counts.get(e.prediction_rank, 0) + 1

    return {
        "total_events": total,
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": hit_rate,
        "runs": len({e.event_id.split("/")[0] for e in events}),
        "phases": phase_stats,
        "rank_distribution": rank_counts,
        "forced_moves": sum(1 for e in events if e.is_forced),
    }


def chunk_events(
    events: list[PredictionEvent], chunk_size: int = 15
) -> list[list[PredictionEvent]]:
    """Group events into batches, keeping same-phase events together."""
    phases: dict[str, list[PredictionEvent]] = {"opening": [], "middlegame": [], "endgame": []}
    for e in events:
        phases[e.phase].append(e)

    chunks: list[list[PredictionEvent]] = []
    for phase_events in phases.values():
        for i in range(0, len(phase_events), chunk_size):
            chunk = phase_events[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)
    return chunks


def format_event_for_llm(event: PredictionEvent) -> str:
    """Render a single PredictionEvent as text for analyst LLM consumption."""
    rank_label = f"Rank: {event.prediction_rank}" if event.event_type == "hit" else "MISS (not in predictions)"
    return (
        f"--- Event: {event.event_id} | {event.event_type.upper()} | {rank_label} ---\n"
        f"Player: {event.player_role} | Phase: {event.phase} | Legal moves: {event.num_legal_moves}\n"
        f"Predicted (ranked): {', '.join(event.predictions[:10])}\n"
        f"Actual move: {event.actual_move}  <-- {event.event_type.upper()}\n"
        f"Board FEN: {event.board_fen}"
    )
