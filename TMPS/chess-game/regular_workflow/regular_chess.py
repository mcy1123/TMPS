"""
Chess Regular Game Runner

A chess game runner that implements regular sequential execution using the same
actor configuration as the speculative runner, without speculation.
"""

import re
import time
import uuid
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import textarena as ta
from utils import Utils
import yaml


class Config:
    """Configuration management with YAML support."""

    def __init__(self, config_path: Optional[str] = "./config.yml"):
        if config_path and config_path.endswith(".yml"):
            self._load_from_yaml(config_path)

    def _load_from_yaml(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        Utils.load_env()

        self.raw_config = config
        self.openai_api_key = Utils.get_api_key(config, "openai")
        self.openrouter_api_key = Utils.get_api_key(config, "openrouter")
        self.deepseek_api_key = Utils.get_api_key(config, "deepseek")
        self.deepseek_base_url = Utils.get_base_url(config, "deepseek")

        self.num_chess_players = config["game"]["num_players"]
        self.stop_after = config["game"]["stop_after"]
        self.agent_name0 = config["game"]["agent_name0"]
        self.agent_name1 = config["game"]["agent_name1"]
        self.num_guesses = config["game"]["num_guesses"]

        logging_config = config.get("logging", {})
        self.logging_schema_version = logging_config.get("schema_version", "spec_v3")
        self.include_compact_observation = logging_config.get("include_compact_observation", True)
        self.include_legal_moves = logging_config.get("include_legal_moves", True)

        self.trajectories_path = config["paths"]["trajectories"]

        prompts = config.get("prompts", {})
        self.actor_system_prompt = prompts.get("actor_system", prompts.get("standard_game", ""))
        self.retry_prompt = prompts["retry"]

        default_actor_provider = Utils.provider_from_agent_name(self.agent_name0)
        self.actor_role = Utils.get_role_config(config, "actor", default_actor_provider, "main")


class ChessActionCleaner:
    """Utility class for cleaning and validating chess actions."""

    UCI_PATTERN = re.compile(r"\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]")

    @classmethod
    def clean_action(cls, action: Optional[str]) -> Optional[str]:
        if action is None:
            return None
        matches = cls.UCI_PATTERN.findall(action)
        if matches:
            return f"[{matches[-1]}]"
        return None

    @classmethod
    def clean_actions(cls, action: Optional[str]) -> List[str]:
        if action is None:
            return []
        matches = cls.UCI_PATTERN.findall(action)
        return [f"[{move}]" for move in matches]


class GameLogger:
    def __init__(self, base_path: str, run_id: str):
        self.base_path = base_path
        self.run_id = run_id
        self.log_path = join(base_path, str(run_id), "log.txt")

    def log(self, level: str, *args, save_log: bool = True) -> None:
        message = f"{level.upper()} {' '.join(str(arg) for arg in args)}\n"
        print(message, end="")
        if save_log:
            Utils.append_file(message.rstrip("\n"), self.log_path)


class AgentManager:
    def __init__(self, config: Config):
        self.config = config
        self.clients = {}
        if config.deepseek_api_key:
            self.clients["deepseek"] = OpenAI(api_key=config.deepseek_api_key, base_url=config.deepseek_base_url)
        if config.openai_api_key:
            self.clients["openai"] = OpenAI(api_key=config.openai_api_key, base_url=Utils.DEFAULT_BASE_URLS["openai"])
        if config.openrouter_api_key:
            self.clients["openrouter"] = OpenAI(api_key=config.openrouter_api_key, base_url=Utils.DEFAULT_BASE_URLS["openrouter"])

    def call_llm(
        self,
        *,
        provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        thinking: Optional[dict] = None,
    ) -> Tuple[str, int, int, int]:
        provider_key = Utils.normalize_provider(provider)
        response = self.clients[provider_key].chat.completions.create(
            **Utils.chat_completion_kwargs(
                provider=provider_key,
                model_name=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                thinking=thinking,
            )
        )
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        return response.choices[0].message.content.strip(), input_tokens, output_tokens, total_tokens


class RegularChessRunner:
    def __init__(self, config: Config):
        self.config = config
        self.agent_manager = AgentManager(config)
        self.agent0_name = config.agent_name0
        self.agent1_name = config.agent_name1
        self.current_run_id: Optional[str] = None
        actor_model = self.config.actor_role["model"].replace("/", "_")
        self.base_traj_path = f"{self.config.trajectories_path.rstrip('/')}/{self.agent0_name}_vs_{self.agent1_name}_actor_{actor_model}_regular"
        self.logger: Optional[GameLogger] = None
        self.env = self._create_environment()
        print("Chess environment initialized successfully")

    def _create_environment(self) -> ta.Env:
        return ta.make(env_id="Chess-v0")

    def _build_compact_observation(self, player_id: int) -> Tuple[str, List[str]]:
        board = self.env.state.game_state["board"]
        role = "White" if player_id == 0 else "Black"
        valid_moves = [f"[{move.uci()}]" for move in board.legal_moves]
        compact_observation = (
            f"[GAME] You are playing as {role} in a game of Chess. "
            "Make your moves in UCI format enclosed in square brackets (e.g., [e2e4]).\n"
            f"[GAME] The current board is:\n{Utils.board_with_coords(board)}\n"
            f"[GAME] The valid moves are: {Utils.serialize_valid_moves(valid_moves)}."
        )
        return compact_observation, valid_moves

    def _build_step_metadata(self, player_id: int, compact_observation: str, valid_moves: List[str]) -> Dict[str, Any]:
        board = self.env.state.game_state["board"]
        return {
            "schema_version": self.config.logging_schema_version,
            "player_role": "White" if player_id == 0 else "Black",
            "board_fen": board.fen(),
            "legal_moves": valid_moves if self.config.include_legal_moves else [],
            "compact_observation": compact_observation if self.config.include_compact_observation else "",
            "guess_prompt_mode": "off",
            "prompt_role": "actor",
        }

    def _log_retry(self, role_name: str, attempt: int, detail: str) -> None:
        if self.logger:
            self.logger.log("RETRY", f"{role_name} attempt {attempt} failed: {detail}")

    def _call_actor_move(self, observation: str, player_id: int, valid_moves: List[str], retries: int = 3) -> Dict[str, Any]:
        role = "White" if player_id == 0 else "Black"
        prompt = observation
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_all_attempts = 0
        retry_count = 0
        final_input_tokens = 0
        final_output_tokens = 0
        final_total_tokens = 0
        raw_action = None
        start_time = time.perf_counter()

        for attempt in range(1, retries + 1):
            try:
                raw_action, input_tokens, output_tokens, total_tokens = self.agent_manager.call_llm(
                    provider=self.config.actor_role["provider"],
                    model_name=self.config.actor_role["model"],
                    system_prompt=self.config.actor_system_prompt,
                    user_prompt=prompt,
                    temperature=self.config.actor_role["temperature"],
                    max_tokens=self.config.actor_role["max_tokens"],
                    reasoning_effort=self.config.actor_role.get("reasoning_effort"),
                    thinking=self.config.actor_role.get("thinking"),
                )
            except Exception as exc:
                raw_action = None
                input_tokens = output_tokens = total_tokens = 0
                self._log_retry(f"{role} actor", attempt, str(exc))
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_tokens_all_attempts += total_tokens
            final_input_tokens = input_tokens
            final_output_tokens = output_tokens
            final_total_tokens = total_tokens
            cleaned = ChessActionCleaner.clean_action(raw_action)
            if cleaned and cleaned in valid_moves:
                return {
                    "action": cleaned,
                    "wall_time": time.perf_counter() - start_time,
                    "input_tokens": final_input_tokens,
                    "output_tokens": final_output_tokens,
                    "total_tokens": final_total_tokens,
                    "input_tokens_all_attempts": total_input_tokens,
                    "output_tokens_all_attempts": total_output_tokens,
                    "total_tokens_all_attempts": total_tokens_all_attempts,
                    "retry_count": retry_count,
                    "fallback_used": False,
                }
            retry_count += 1
            self._log_retry(f"{role} actor", attempt, raw_action or "empty response")
            prompt += "\n" + self.config.retry_prompt.format(attempt=attempt, role=role)

        fallback_action = valid_moves[0] if valid_moves else None
        if fallback_action is not None:
            self._log_retry(f"{role} actor", retries, f"using fallback legal move {fallback_action}")
        return {
            "action": fallback_action,
            "wall_time": time.perf_counter() - start_time,
            "input_tokens": final_input_tokens,
            "output_tokens": final_output_tokens,
            "total_tokens": final_total_tokens,
            "input_tokens_all_attempts": total_input_tokens,
            "output_tokens_all_attempts": total_output_tokens,
            "total_tokens_all_attempts": total_tokens_all_attempts,
            "retry_count": retry_count,
            "fallback_used": fallback_action is not None,
        }

    def _execute_game_loop(self, stop_after: Optional[int] = None) -> Tuple[Dict[int, Any], Any, Any, Dict[str, Any]]:
        self.env.reset(num_players=self.config.num_chess_players)
        steps_info: Dict[int, Dict[str, Any]] = {}
        step_count = 0
        done = False
        total_effective_wall_time = 0.0
        total_tokens_effective = 0

        while not done:
            player_id, observation = self.env.get_observation()
            compact_observation, valid_moves = self._build_compact_observation(player_id)
            step_metadata = self._build_step_metadata(player_id, compact_observation, valid_moves)
            actor_result = self._call_actor_move(compact_observation, player_id, valid_moves)
            total_effective_wall_time += actor_result["wall_time"]
            total_tokens_effective += actor_result["total_tokens_all_attempts"]

            steps_info[step_count] = {
                **step_metadata,
                "player_id": player_id,
                "current_observation": observation,
                "current_move": actor_result["action"],
                "time_taken_current_agent": actor_result["wall_time"],
                "wall_time_step_actor_only": actor_result["wall_time"],
                "wall_time_step_effective": actor_result["wall_time"],
                "wall_time_speculation_task": 0.0,
                "tokens_step_effective": actor_result["total_tokens_all_attempts"],
                "input_tokens_current_agent": actor_result["input_tokens"],
                "output_tokens_current_agent": actor_result["output_tokens"],
                "total_tokens_current_agent": actor_result["total_tokens"],
                "input_tokens_current_agent_all_attempts": actor_result["input_tokens_all_attempts"],
                "output_tokens_current_agent_all_attempts": actor_result["output_tokens_all_attempts"],
                "total_tokens_current_agent_all_attempts": actor_result["total_tokens_all_attempts"],
                "retry_count_current_agent": actor_result["retry_count"],
                "fallback_used_current_agent": actor_result.get("fallback_used", False),
            }

            if self.logger:
                self.logger.log("INFO", f"STEP {step_count}:", Utils.dict_to_str(steps_info[step_count]))
                self.logger.log("-" * 100)

            done, _ = self.env.step(actor_result["action"])
            step_count += 1
            if stop_after and step_count >= stop_after:
                break

        rewards, game_info = self.env.close()
        run_metrics = {
            "target_stop_after": stop_after,
            "steps_completed": step_count,
            "complete_target_steps": bool(stop_after and step_count >= stop_after),
            "total_effective_wall_time": total_effective_wall_time,
            "total_tokens_effective": total_tokens_effective,
        }
        return steps_info, rewards, game_info, run_metrics

    def run(self, stop_after: int = 20) -> None:
        self.current_run_id = str(uuid.uuid4())
        self.logger = GameLogger(self.base_traj_path, self.current_run_id)
        current_dir_path = join(self.base_traj_path, self.current_run_id)
        self.logger.log("INFO", f"Starting run {self.current_run_id} actor={self.config.actor_role['model']}")

        steps_info, rewards, game_info, run_metrics = self._execute_game_loop(stop_after=stop_after)
        Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
        Utils.save_json(rewards, join(current_dir_path, "rewards.json"))
        Utils.save_json(game_info, join(current_dir_path, "game_info.json"))
        Utils.save_json(run_metrics, join(current_dir_path, "run_metrics.json"))
        Utils.save_json(run_metrics["total_effective_wall_time"], join(current_dir_path, "time_checker_regular.json"))
        self.logger.log("INFO", f"Run completed for {self.current_run_id}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run regular chess (paper-like DeepSeek approximation).")
    p.add_argument("--config", default="config.yml", help="Path to config YAML (default: config.yml)")
    p.add_argument("--trajectories-dir", default=None, help="Output directory for trajectories (overrides config)")
    p.add_argument("--stop-after", type=int, default=None, help="Stop after N steps (default: from config)")
    args = p.parse_args()

    config = Config(args.config)
    if args.trajectories_dir is not None:
        config.trajectories_path = args.trajectories_dir.rstrip("/")

    runner = RegularChessRunner(config=config)
    stop_after = args.stop_after if args.stop_after is not None else config.stop_after
    start_time = time.time()
    runner.run(stop_after=stop_after)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
