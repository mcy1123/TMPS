"""Stage 2: Parallel multi-agent trajectory analysis for skill patch proposal."""

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import yaml
from openai import OpenAI

from .extractor import PredictionEvent, split_events
from .prompts import (
    ERROR_ANALYST_SYSTEM_PROMPT,
    SUCCESS_ANALYST_SYSTEM_PROMPT,
    format_analyst_user_message,
)

# ── Patch parsing ──────────────────────────────────────────────

_PATCH_RE = re.compile(
    r"###\s*PATCH:\s*(.+?)\n"
    r"\*\*Priority\*\*:\s*(HIGH|MEDIUM|LOW)\s*\n"
    r"\*\*Evidence\*\*:\s*(.*?)\s*\n"
    r"\*\*Proposed Content\*\*:\s*\n(.*?)\n"
    r"\*\*Rationale\*\*:\s*(.*?)(?=\n###\s*PATCH:|\n---|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_patch_response(text: str) -> list[dict[str, Any]]:
    """Parse structured patch proposals from LLM response."""
    patches: list[dict[str, Any]] = []
    for match in _PATCH_RE.finditer(text):
        patches.append(
            {
                "section": match.group(1).strip(),
                "priority": match.group(2).strip().upper(),
                "evidence": match.group(3).strip(),
                "content": match.group(4).strip(),
                "rationale": match.group(5).strip(),
            }
        )
    return patches


# ── Helpers ────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import Utils  # noqa: E402


class AnalystRunner:
    """Manages LLM calls for trajectory analysis."""

    SYSTEM_PROMPTS = {"error": ERROR_ANALYST_SYSTEM_PROMPT, "success": SUCCESS_ANALYST_SYSTEM_PROMPT}

    def __init__(self, config_path: str = "config.yml"):
        config = _load_config(config_path)
        actor_cfg = config.get("agents", {}).get("actor", {})
        provider = actor_cfg.get("provider", "DeepSeek").strip().lower()
        provider_key = Utils.normalize_provider(provider)
        api_key = Utils.get_api_key(config, provider)
        base_url = Utils.get_base_url(config, provider)
        self.model = actor_cfg.get("model", "deepseek-chat")
        self.temperature = actor_cfg.get("temperature", 0)
        self.max_tokens = actor_cfg.get("max_tokens", 2048)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def run_single_analysis(
        self,
        events: list[PredictionEvent],
        analyst_type: str,
        existing_skill: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run one analyst call on a chunk of events. Returns (patches, usage_info)."""
        system = self.SYSTEM_PROMPTS[analyst_type]
        user = format_analyst_user_message(events, analyst_type, existing_skill)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        usage_info = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        patches = parse_patch_response(text)
        for p in patches:
            p["source"] = analyst_type
        return patches, usage_info

    def run_parallel_analysis(
        self,
        events: list[PredictionEvent],
        analyst_type: str,
        chunk_size: int = 15,
        max_workers: int = 4,
        existing_skill: str = "",
    ) -> list[dict[str, Any]]:
        """Dispatch parallel analyst calls on chunked events."""
        from .extractor import chunk_events

        chunks = chunk_events(events, chunk_size)
        if not chunks:
            return []

        all_patches: list[dict[str, Any]] = []
        total_tokens = 0
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_single_analysis, chunk, analyst_type, existing_skill): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    patches, usage = future.result()
                    all_patches.extend(patches)
                    total_tokens += usage["total_tokens"]
                    print(f"  [{analyst_type}] chunk {i+1}/{len(chunks)}: {len(patches)} patches, {usage['total_tokens']} tokens")
                except Exception as exc:
                    print(f"  [{analyst_type}] chunk {i+1}/{len(chunks)} FAILED: {exc}")

        elapsed = time.perf_counter() - start
        print(f"  [{analyst_type}] done: {len(all_patches)} total patches in {elapsed:.1f}s ({total_tokens} tokens)")
        return all_patches


def run_full_analysis(
    events: list[PredictionEvent],
    config_path: str = "config.yml",
    existing_skill: str = "",
    chunk_size: int = 15,
    max_workers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run both error and success analysts in parallel. Returns (error_patches, success_patches)."""
    hits, misses = split_events(events)
    runner = AnalystRunner(config_path)

    print(f"\n=== Stage 2: Parallel Analysis ===")
    print(f"Hits: {len(hits)}, Misses: {len(misses)}")
    print(f"Chunk size: {chunk_size}, Workers: {max_workers}")

    error_patches: list[dict[str, Any]] = []
    success_patches: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        error_future = executor.submit(
            runner.run_parallel_analysis, misses, "error", chunk_size, max_workers, existing_skill
        ) if misses else None
        success_future = executor.submit(
            runner.run_parallel_analysis, hits, "success", chunk_size, max_workers, existing_skill
        ) if hits else None

        if error_future:
            error_patches = error_future.result()
        if success_future:
            success_patches = success_future.result()

    print(f"\nTotal: {len(error_patches)} error patches, {len(success_patches)} success patches")
    return error_patches, success_patches
