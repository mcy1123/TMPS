"""E-commerce domain analyst: parallel LLM analysis of prediction events.

Uses the same AnalystRunner pattern as hotpotqa but with e-commerce-specific
prompts and event format.
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from .extractor import PredictionEvent, chunk_events
from .prompts import (
    ERROR_ANALYST_SYSTEM_PROMPT,
    SUCCESS_ANALYST_SYSTEM_PROMPT,
    format_analyst_user_message,
)

_PATCH_RE = re.compile(
    r"### PATCH:\s*(.+?)\n"
    r"\*\*Priority\*\*:\s*(HIGH|MEDIUM|LOW)\s*\n"
    r"\*\*Evidence\*\*:\s*(.+?)\n"
    r"\*\*Proposed Content\*\*:\s*\n(.*?)\n"
    r"\*\*Rationale\*\*:\s*(.+?)(?=\n### PATCH:|\n\Z)",
    re.DOTALL,
)


def parse_patches(text: str, source: str = "unknown") -> list[dict]:
    """Extract structured patch proposals from analyst LLM output."""
    patches = []
    for match in _PATCH_RE.finditer(text):
        patches.append({
            "source": source,
            "section": match.group(1).strip(),
            "priority": match.group(2).strip(),
            "evidence": match.group(3).strip(),
            "content": match.group(4).strip(),
            "rationale": match.group(5).strip(),
        })
    if not patches:
        patches = _fallback_parse(text, source)
    return patches


def _fallback_parse(text: str, source: str) -> list[dict]:
    """Looser patch extraction when the strict regex fails."""
    patches = []
    sections = re.split(r"###\s*PATCH:", text, flags=re.IGNORECASE)
    for section in sections[1:]:
        patch = {"source": source, "section": "General", "priority": "MEDIUM",
                 "evidence": "", "content": "", "rationale": ""}
        header = section.strip().split("\n")[0] if section.strip() else "General"
        patch["section"] = header.strip()
        prio_match = re.search(r"\*\*Priority\*\*:\s*(HIGH|MEDIUM|LOW)", section)
        if prio_match:
            patch["priority"] = prio_match.group(1)
        ev_match = re.search(r"\*\*Evidence\*\*:\s*(.+?)(?:\n\*\*|\n\n|\Z)", section, re.DOTALL)
        if ev_match:
            patch["evidence"] = ev_match.group(1).strip()
        content_match = re.search(r"\*\*Proposed Content\*\*:\s*\n(.*?)(?:\n\*\*Rationale|\Z)", section, re.DOTALL)
        if content_match:
            patch["content"] = content_match.group(1).strip()
        rat_match = re.search(r"\*\*Rationale\*\*:\s*(.+?)(?:\n\Z|\Z)", section, re.DOTALL)
        if rat_match:
            patch["rationale"] = rat_match.group(1).strip()
        if patch["content"]:
            patches.append(patch)
    return patches


class AnalystRunner:
    """Runs parallel LLM analysis on batches of e-commerce prediction events."""

    def __init__(self, model_name: str = "deepseek-chat", max_workers: int = 4):
        self.model_name = model_name
        self.max_workers = max_workers

    @property
    def _llm_client_cls(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "hotpotqa"))
        from src.llm_client import LLMClient
        return LLMClient

    def _make_llm(self, temperature: float = 0.0, max_tokens: int = 2048):
        return self._llm_client_cls(
            model_name=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

    def run_single_analysis(
        self,
        events: list[PredictionEvent],
        analyst_type: str,
        existing_skill: str = "",
    ) -> list[dict]:
        """Run analysis on a single batch of events."""
        if analyst_type == "error":
            system = ERROR_ANALYST_SYSTEM_PROMPT
        else:
            system = SUCCESS_ANALYST_SYSTEM_PROMPT

        user_msg = format_analyst_user_message(events, analyst_type, existing_skill)
        llm = self._make_llm()
        response = llm.call_with_system(system, user_msg)
        return parse_patches(response, source=analyst_type)

    def run_parallel_analysis(
        self,
        hit_events: list[PredictionEvent],
        miss_events: list[PredictionEvent],
        existing_skill: str = "",
    ) -> tuple[list[dict], list[dict]]:
        """Run error and success analysts in parallel across event chunks."""
        hit_chunks = chunk_events(hit_events) if hit_events else []
        miss_chunks = chunk_events(miss_events) if miss_events else []

        all_success_patches: list[dict] = []
        all_error_patches: list[dict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for chunk in hit_chunks:
                f = executor.submit(
                    self.run_single_analysis, chunk, "success", existing_skill
                )
                futures[f] = "success"
            for chunk in miss_chunks:
                f = executor.submit(
                    self.run_single_analysis, chunk, "error", existing_skill
                )
                futures[f] = "error"

            for future in as_completed(futures):
                try:
                    patches = future.result()
                except Exception as e:
                    print(f"[Analyst] Task failed: {e}", file=sys.stderr)
                    continue
                if not patches:
                    continue
                if futures[future] == "success":
                    all_success_patches.extend(patches)
                else:
                    all_error_patches.extend(patches)

        return all_success_patches, all_error_patches
