"""
Chess Speculative Game Runner

A chess game runner that implements speculative execution where a fast
speculator predicts the current side's move while the actor selects its move,
then prepares the opponent response in parallel.
"""

import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

import chess
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
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            Utils.load_env()

            self.raw_config = config

            self.openai_api_key = Utils.get_api_key(config, "openai")
            self.openrouter_api_key = Utils.get_api_key(config, "openrouter")
            self.deepseek_api_key = Utils.get_api_key(config, "deepseek")
            self.deepseek_base_url = Utils.get_base_url(config, "deepseek")

            self.num_chess_players = config["game"]["num_players"]
            self.client_error_sleep_time = config["game"]["error_sleep_time"]
            self.server_error_sleep_time = config["game"]["error_sleep_time"]
            self.stop_after = config["game"]["stop_after"]
            self.agent_name0 = config["game"]["agent_name0"]
            self.agent_name1 = config["game"]["agent_name1"]
            self.num_guesses = config["game"]["num_guesses"]

            guess_cfg = config.get("guess", {})
            self.guess_skill_enabled = guess_cfg.get("skill_enabled", False)
            self.guess_skill_mode = guess_cfg.get("skill_mode", "paper_like_fast_predictor")
            self.guess_skill_prompt = guess_cfg.get("skill_prompt", "")
            skill_file = guess_cfg.get("skill_file", "")
            if self.guess_skill_enabled and skill_file:
                skill_path = os.path.join(os.path.dirname(__file__), "..", skill_file)
                skill_path = os.path.normpath(skill_path)
                if os.path.exists(skill_path):
                    with open(skill_path, "r", encoding="utf-8") as _sf:
                        self.guess_skill_prompt = _sf.read().strip()
                    self.guess_skill_mode = "trace2skill"

            self.gating_enabled = config.get("gating", {}).get("enabled", False)
            self.gating_mode = config.get("gating", {}).get("mode", "rule")
            self.gating_rule_max_legal_moves = config.get("gating", {}).get("rule_max_legal_moves")
            self.speculation_branch_use_fast_mode = config.get("speculation", {}).get("branch_use_fast_mode", True)

            logging_config = config.get("logging", {})
            self.logging_schema_version = logging_config.get("schema_version", "spec_v3")
            self.include_compact_observation = logging_config.get("include_compact_observation", True)
            self.include_legal_moves = logging_config.get("include_legal_moves", True)

            self.trajectories_path = config["paths"]["trajectories"]

            prompts = config.get("prompts", {})
            self.actor_system_prompt = prompts.get("actor_system", prompts.get("standard_game", ""))
            self.speculator_system_prompt = prompts.get("speculator_system", prompts.get("standard_game", ""))
            self.speculator_user_prompt = prompts.get("speculator_user", prompts.get("guess", ""))
            self.retry_prompt = prompts["retry"]

            self.analysis_target_steps = config.get("analysis", {}).get("target_steps", [30])
            self.analysis_require_full_target_steps = config.get("analysis", {}).get("require_full_target_steps", True)

            default_actor_provider = Utils.provider_from_agent_name(self.agent_name0)
            self.actor_role = Utils.get_role_config(config, "actor", default_actor_provider, "main")
            self.speculator_role = Utils.get_role_config(
                config,
                "speculator",
                config.get("guess", {}).get("provider", default_actor_provider),
                "guess",
            )
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            raise


class ChessActionCleaner:
    """Utility class for cleaning and validating chess actions."""

    UCI_PATTERN = re.compile(r"\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]")
    UCI_RAW_PATTERN = re.compile(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b')

    @classmethod
    def clean_action(cls, action: Optional[str]) -> Optional[str]:
        if action is None:
            return None
        matches = cls.UCI_PATTERN.findall(action)
        if matches:
            return f"[{matches[-1]}]"
        json_moves = cls._try_parse_json_moves(action)
        if json_moves:
            return json_moves[0]
        return None

    @classmethod
    def clean_actions(cls, action: Optional[str]) -> List[str]:
        if action is None:
            return []
        matches = cls.UCI_PATTERN.findall(action)
        if matches:
            return [f"[{move}]" for move in matches]
        json_moves = cls._try_parse_json_moves(action)
        if json_moves:
            return json_moves
        raw_matches = cls.UCI_RAW_PATTERN.findall(action)
        return [f"[{move}]" for move in raw_matches]

    @classmethod
    def _try_parse_json_moves(cls, text: str) -> List[str]:
        try:
            data = json.loads(text)
            moves = data.get("moves", [])
            validated = []
            for m in moves:
                m_str = str(m).strip()
                if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', m_str):
                    validated.append(f"[{m_str}]")
            return validated
        except (json.JSONDecodeError, TypeError, AttributeError):
            return []


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
        response_format: Optional[dict] = None,
    ) -> Tuple[str, int, int, int]:
        provider_key = Utils.normalize_provider(provider)
        if provider_key not in self.clients:
            raise ValueError(f"Unknown provider: {provider}")
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
                response_format=response_format,
            )
        )
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        return response.choices[0].message.content.strip(), input_tokens, output_tokens, total_tokens


class SpeculativeChessRunner:
    def __init__(self, config: Config):
        self.config = config
        self.agent_manager = AgentManager(config)
        self.agent0_name = config.agent_name0
        self.agent1_name = config.agent_name1
        self.num_guesses = config.num_guesses
        self.current_run_id: Optional[str] = None
        actor_model = self.config.actor_role["model"].replace("/", "_")
        spec_model = self.config.speculator_role["model"].replace("/", "_")
        base = self.config.trajectories_path.rstrip("/")
        self.base_traj_path = f"{base}/{self.agent0_name}_vs_{self.agent1_name}_actor_{actor_model}_spec_{spec_model}"
        self.logger: Optional[GameLogger] = None
        self.env = self._create_environment()
        print("Chess environment initialized successfully")

    def _create_environment(self) -> ta.Env:
        return ta.make(env_id="Chess-v0")

    def _get_guess_prompt_mode(self) -> str:
        if self.config.speculator_user_prompt.strip():
            if self.config.guess_skill_mode and self.config.guess_skill_mode != "off":
                return self.config.guess_skill_mode
            return "paper_like_fast_predictor"
        return "off"

    def _build_compact_observation(self, player_id: int, board: Optional[chess.Board] = None) -> Tuple[str, List[str]]:
        board = board or self.env.state.game_state["board"]
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
            "guess_prompt_mode": self._get_guess_prompt_mode(),
            "prompt_role": "actor",
        }

    def _should_run_speculation(self, valid_moves: List[str]) -> Tuple[bool, str, str]:
        if not self.config.gating_enabled:
            return True, "run", "gating_disabled"
        if self.config.gating_mode != "rule":
            return True, "run", f"unsupported_gating_mode:{self.config.gating_mode}"
        threshold = self.config.gating_rule_max_legal_moves
        if threshold is not None and len(valid_moves) > threshold:
            return False, "skip", f"legal_moves>{threshold}"
        return True, "run", "rule_pass"

    def _log_retry(self, role_name: str, attempt: int, detail: str) -> None:
        if self.logger:
            self.logger.log("RETRY", f"{role_name} attempt {attempt} failed: {detail}")

    def _branch_role_source(self) -> str:
        return "speculator_role" if self.config.speculation_branch_use_fast_mode else "actor_role"

    def _call_actor_move(
        self,
        observation: str,
        player_id: int,
        valid_moves: List[str],
        retries: int = 3,
        use_fast_mode: bool = False,
    ) -> Dict[str, Any]:
        role_cfg = self.config.speculator_role if use_fast_mode else self.config.actor_role
        system_prompt = self.config.speculator_system_prompt if use_fast_mode else self.config.actor_system_prompt
        role = "White" if player_id == 0 else "Black"
        prompt = observation
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_all_attempts = 0
        retry_count = 0
        last_raw_action = None
        final_input_tokens = 0
        final_output_tokens = 0
        final_total_tokens = 0
        start_time = time.perf_counter()

        for attempt in range(1, retries + 1):
            try:
                raw_action, input_tokens, output_tokens, total_tokens = self.agent_manager.call_llm(
                    provider=role_cfg["provider"],
                    model_name=role_cfg["model"],
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    temperature=role_cfg["temperature"],
                    max_tokens=role_cfg["max_tokens"],
                    reasoning_effort=role_cfg.get("reasoning_effort"),
                    thinking=role_cfg.get("thinking"),
                    response_format=role_cfg.get("response_format"),
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
            last_raw_action = raw_action
            cleaned_action = ChessActionCleaner.clean_action(raw_action)
            if cleaned_action and cleaned_action in valid_moves:
                return {
                    "action": cleaned_action,
                    "wall_time": time.perf_counter() - start_time,
                    "input_tokens": final_input_tokens,
                    "output_tokens": final_output_tokens,
                    "total_tokens": final_total_tokens,
                    "input_tokens_all_attempts": total_input_tokens,
                    "output_tokens_all_attempts": total_output_tokens,
                    "total_tokens_all_attempts": total_tokens_all_attempts,
                    "retry_count": retry_count,
                    "raw_action": raw_action,
                    "fallback_used": False,
                }
            retry_count += 1
            detail = raw_action if raw_action else "empty response"
            self._log_retry(f"{role} actor", attempt, detail)
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
            "raw_action": last_raw_action,
            "fallback_used": fallback_action is not None,
        }

    def _guess_actions(self, observation: str, retries: int = 3) -> Dict[str, Any]:
        prompt = Utils.build_guess_prompt(
            observation=observation,
            guess_prompt_template=self.config.speculator_user_prompt,
            num_guesses=self.num_guesses,
            skill_enabled=self.config.guess_skill_enabled,
            skill_prompt=self.config.guess_skill_prompt,
            skill_position="suffix",
        )
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens_all_attempts = 0
        retry_count = 0
        final_input_tokens = 0
        final_output_tokens = 0
        final_total_tokens = 0
        raw_output = None
        start_time = time.perf_counter()

        for attempt in range(1, retries + 1):
            try:
                raw_output, input_tokens, output_tokens, total_tokens = self.agent_manager.call_llm(
                    provider=self.config.speculator_role["provider"],
                    model_name=self.config.speculator_role["model"],
                    system_prompt=self.config.speculator_system_prompt,
                    user_prompt=prompt,
                    temperature=self.config.speculator_role["temperature"],
                    max_tokens=self.config.speculator_role["max_tokens"],
                    reasoning_effort=self.config.speculator_role.get("reasoning_effort"),
                    thinking=self.config.speculator_role.get("thinking"),
                    response_format=self.config.speculator_role.get("response_format"),
                )
            except Exception as exc:
                raw_output = None
                input_tokens = output_tokens = total_tokens = 0
                self._log_retry("speculator", attempt, str(exc))
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_tokens_all_attempts += total_tokens
            final_input_tokens = input_tokens
            final_output_tokens = output_tokens
            final_total_tokens = total_tokens
            predictions = Utils.dedupe_preserve_order(ChessActionCleaner.clean_actions(raw_output))[: self.num_guesses]
            if len(predictions) == self.num_guesses:
                return {
                    "predictions": predictions,
                    "wall_time": time.perf_counter() - start_time,
                    "input_tokens": final_input_tokens,
                    "output_tokens": final_output_tokens,
                    "total_tokens": final_total_tokens,
                    "input_tokens_all_attempts": total_input_tokens,
                    "output_tokens_all_attempts": total_output_tokens,
                    "total_tokens_all_attempts": total_tokens_all_attempts,
                    "retry_count": retry_count,
                    "raw_output": raw_output,
                }
            retry_count += 1
            self._log_retry("speculator", attempt, raw_output or "insufficient predictions")
            prompt += (
                f"\nReturn exactly {self.num_guesses} unique legal moves in [UCI_MOVE] format only. "
                "Do not explain."
            )

        return {
            "predictions": Utils.dedupe_preserve_order(ChessActionCleaner.clean_actions(raw_output))[: self.num_guesses],
            "wall_time": time.perf_counter() - start_time,
            "input_tokens": final_input_tokens,
            "output_tokens": final_output_tokens,
            "total_tokens": final_total_tokens,
            "input_tokens_all_attempts": total_input_tokens,
            "output_tokens_all_attempts": total_output_tokens,
            "total_tokens_all_attempts": total_tokens_all_attempts,
            "retry_count": retry_count,
            "raw_output": raw_output,
        }

    def _simulate_and_speculate(self, player_id: int, predicted_move: str) -> Dict[str, Any]:
        move_uci = predicted_move.lower().replace("[", "").replace("]", "")
        predicted_chess_move = chess.Move.from_uci(move_uci)
        board_copy = self.env.state.game_state["board"].copy()
        board_copy.push(predicted_chess_move)
        spec_player_id = 1 - player_id
        new_observation, valid_moves = self._build_compact_observation(spec_player_id, board=board_copy)
        use_fast_mode = self.config.speculation_branch_use_fast_mode
        actor_result = self._call_actor_move(
            new_observation,
            spec_player_id,
            valid_moves,
            use_fast_mode=use_fast_mode,
        )
        actor_result["observation"] = new_observation
        actor_result["use_fast_mode"] = use_fast_mode
        actor_result["role_source"] = "speculator_role" if use_fast_mode else "actor_role"
        return actor_result

    def _speculation_task(self, player_id: int, compact_observation: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        guess_result = self._guess_actions(compact_observation, retries=3)
        predictions = guess_result["predictions"]
        branch_results: List[Dict[str, Any]] = []
        if predictions:
            with ThreadPoolExecutor(max_workers=len(predictions)) as executor:
                branch_results = list(executor.map(lambda move: self._simulate_and_speculate(player_id, move), predictions))
        branch_times = [result.get("wall_time", 0.0) for result in branch_results]
        branch_actions = [result.get("action") for result in branch_results]
        total_times = [guess_result["wall_time"] + branch_time for branch_time in branch_times]
        return {
            "predictions": predictions,
            "speculations": branch_actions,
            "branch_results": branch_results,
            "prediction_wall_times": [guess_result["wall_time"]] * len(predictions),
            "branch_wall_times": branch_times,
            "combined_branch_times": total_times,
            "guess_result": guess_result,
            "wall_time_task": time.perf_counter() - start_time,
            "tokens_step_effective": guess_result["total_tokens_all_attempts"] + sum(
                result.get("total_tokens_all_attempts", 0) for result in branch_results
            ),
        }

    def _execute_game_loop(
        self,
        stop_after: Optional[int] = None,
    ) -> Tuple[Dict[int, Any], Any, Any, Dict[str, Any]]:
        self.env.reset(num_players=self.config.num_chess_players)
        steps_info: Dict[int, Dict[str, Any]] = {}
        step_count = 0
        done = False
        is_initial_step = True
        total_effective_wall_time = 0.0
        total_actor_wall_time_executed = 0.0
        total_speculation_wall_time = 0.0
        total_tokens_effective = 0
        reused_steps = 0
        player_id, observation = self.env.get_observation()

        while not done:
            if not is_initial_step:
                prev_step = steps_info[step_count - 1]
                prev_predictions = prev_step["current_pred"]
                prev_speculations = prev_step["current_spec"]
                prev_move = prev_step["current_move"]
                prev_cached_results = prev_step.get("speculation_branch_results", [])
            else:
                prev_predictions = []
                prev_speculations = []
                prev_move = None
                prev_cached_results = []

            player_id, observation = self.env.get_observation()
            compact_observation, valid_moves = self._build_compact_observation(player_id)
            step_metadata = self._build_step_metadata(player_id, compact_observation, valid_moves)

            speculation_hit = False
            current_move = None
            current_predictions: List[str] = []
            current_speculations: List[Optional[str]] = []
            prediction_times: List[float] = []
            speculation_times: List[float] = []
            combined_times: List[float] = []
            input_prediction_tokens: List[int] = []
            output_prediction_tokens: List[int] = []
            total_prediction_tokens: List[int] = []
            input_prediction_tokens_all_attempts: List[int] = []
            output_prediction_tokens_all_attempts: List[int] = []
            total_prediction_tokens_all_attempts: List[int] = []
            input_speculation_tokens: List[int] = []
            output_speculation_tokens: List[int] = []
            total_speculation_tokens: List[int] = []
            input_speculation_tokens_all_attempts: List[int] = []
            output_speculation_tokens_all_attempts: List[int] = []
            total_speculation_tokens_all_attempts: List[int] = []
            retry_count_speculation: List[int] = []
            input_tokens1 = output_tokens1 = total_tokens1 = 0
            input_tokens1_all_attempts = output_tokens1_all_attempts = total_tokens1_all_attempts = 0
            retry_count_current_agent = 0
            actor_wall_time = 0.0
            wall_time_step_effective = 0.0
            wall_time_speculation_task = 0.0
            tokens_step_effective = 0
            gate_should_run, gate_decision, gate_reason = self._should_run_speculation(valid_moves)
            prediction_hit_index = -1
            prediction_hit_move = None
            used_speculation_index = -1
            used_speculation_source_step = None
            speculation_branch_results: List[Dict[str, Any]] = []

            if not is_initial_step and prev_predictions:
                for i, pred in enumerate(prev_predictions):
                    if pred == prev_move and i < len(prev_speculations) and prev_speculations[i]:
                        current_move = prev_speculations[i]
                        speculation_hit = True
                        used_speculation_index = i
                        used_speculation_source_step = step_count - 1
                        cached = prev_cached_results[i] if i < len(prev_cached_results) else {}
                        actor_wall_time = cached.get("wall_time", 0.0)
                        input_tokens1 = cached.get("input_tokens", 0)
                        output_tokens1 = cached.get("output_tokens", 0)
                        total_tokens1 = cached.get("total_tokens", 0)
                        input_tokens1_all_attempts = cached.get("input_tokens_all_attempts", 0)
                        output_tokens1_all_attempts = cached.get("output_tokens_all_attempts", 0)
                        total_tokens1_all_attempts = cached.get("total_tokens_all_attempts", 0)
                        retry_count_current_agent = cached.get("retry_count", 0)
                        fallback_used_current_agent = cached.get("fallback_used", False)
                        gate_decision = "reuse"
                        gate_reason = "speculation_hit_from_previous_step"
                        reused_steps += 1
                        break
            else:
                fallback_used_current_agent = False

            if is_initial_step:
                fallback_used_current_agent = False

            if not speculation_hit:
                if gate_should_run:
                    step_start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        current_future = executor.submit(self._call_actor_move, compact_observation, player_id, valid_moves)
                        speculation_future = executor.submit(self._speculation_task, player_id, compact_observation)
                        current_result = current_future.result()
                        speculation_result = speculation_future.result()
                    wall_time_step_effective = time.perf_counter() - step_start
                    wall_time_speculation_task = speculation_result["wall_time_task"]
                    current_move = current_result["action"]
                    actor_wall_time = current_result["wall_time"]
                    input_tokens1 = current_result["input_tokens"]
                    output_tokens1 = current_result["output_tokens"]
                    total_tokens1 = current_result["total_tokens"]
                    input_tokens1_all_attempts = current_result["input_tokens_all_attempts"]
                    output_tokens1_all_attempts = current_result["output_tokens_all_attempts"]
                    total_tokens1_all_attempts = current_result["total_tokens_all_attempts"]
                    retry_count_current_agent = current_result["retry_count"]
                    fallback_used_current_agent = current_result.get("fallback_used", False)
                    current_predictions = speculation_result["predictions"]
                    current_speculations = speculation_result["speculations"]
                    prediction_times = speculation_result["prediction_wall_times"]
                    speculation_times = speculation_result["branch_wall_times"]
                    combined_times = speculation_result["combined_branch_times"]
                    speculation_branch_results = speculation_result["branch_results"]
                    guess_result = speculation_result["guess_result"]
                    input_prediction_tokens = [guess_result["input_tokens"]] * len(current_predictions)
                    output_prediction_tokens = [guess_result["output_tokens"]] * len(current_predictions)
                    total_prediction_tokens = [guess_result["total_tokens"]] * len(current_predictions)
                    input_prediction_tokens_all_attempts = [guess_result["input_tokens_all_attempts"]] * len(current_predictions)
                    output_prediction_tokens_all_attempts = [guess_result["output_tokens_all_attempts"]] * len(current_predictions)
                    total_prediction_tokens_all_attempts = [guess_result["total_tokens_all_attempts"]] * len(current_predictions)
                    input_speculation_tokens = [result.get("input_tokens", 0) for result in speculation_branch_results]
                    output_speculation_tokens = [result.get("output_tokens", 0) for result in speculation_branch_results]
                    total_speculation_tokens = [result.get("total_tokens", 0) for result in speculation_branch_results]
                    input_speculation_tokens_all_attempts = [result.get("input_tokens_all_attempts", 0) for result in speculation_branch_results]
                    output_speculation_tokens_all_attempts = [result.get("output_tokens_all_attempts", 0) for result in speculation_branch_results]
                    total_speculation_tokens_all_attempts = [result.get("total_tokens_all_attempts", 0) for result in speculation_branch_results]
                    retry_count_speculation = [result.get("retry_count", 0) for result in speculation_branch_results]
                    tokens_step_effective = (
                        current_result["total_tokens_all_attempts"] + speculation_result["tokens_step_effective"]
                    )
                else:
                    current_result = self._call_actor_move(compact_observation, player_id, valid_moves)
                    current_move = current_result["action"]
                    actor_wall_time = current_result["wall_time"]
                    wall_time_step_effective = actor_wall_time
                    input_tokens1 = current_result["input_tokens"]
                    output_tokens1 = current_result["output_tokens"]
                    total_tokens1 = current_result["total_tokens"]
                    input_tokens1_all_attempts = current_result["input_tokens_all_attempts"]
                    output_tokens1_all_attempts = current_result["output_tokens_all_attempts"]
                    total_tokens1_all_attempts = current_result["total_tokens_all_attempts"]
                    retry_count_current_agent = current_result["retry_count"]
                    fallback_used_current_agent = current_result.get("fallback_used", False)
                    tokens_step_effective = current_result["total_tokens_all_attempts"]
                is_initial_step = False
                total_effective_wall_time += wall_time_step_effective
                total_actor_wall_time_executed += actor_wall_time
                total_speculation_wall_time += wall_time_speculation_task
                total_tokens_effective += tokens_step_effective
                if current_move in current_predictions:
                    prediction_hit_index = current_predictions.index(current_move)
                    prediction_hit_move = current_move
            else:
                wall_time_step_effective = 0.0
                wall_time_speculation_task = 0.0
                tokens_step_effective = 0

            steps_info[step_count] = {
                **step_metadata,
                "player_id": player_id,
                "current_observation": observation,
                "current_move": current_move,
                "current_pred": current_predictions,
                "current_spec": current_speculations,
                "time_taken_current_agent": actor_wall_time,
                "time_taken_other_agent": combined_times,
                "time_taken_prediction": prediction_times,
                "time_taken_speculation": speculation_times,
                "speculation_hit": speculation_hit,
                "prediction_hit_index": prediction_hit_index,
                "prediction_hit_move": prediction_hit_move,
                "used_speculation_index": used_speculation_index,
                "used_speculation_source_step": used_speculation_source_step,
                "gate_decision": gate_decision,
                "gate_reason": gate_reason,
                "num_predictions": len(current_predictions),
                "speculation_branch_use_fast_mode": self.config.speculation_branch_use_fast_mode,
                "speculation_branch_role_source": self._branch_role_source(),
                "wall_time_step_actor_only": actor_wall_time,
                "wall_time_speculation_task": wall_time_speculation_task,
                "wall_time_step_effective": wall_time_step_effective,
                "tokens_step_effective": tokens_step_effective,
                "input_tokens_current_agent": input_tokens1,
                "output_tokens_current_agent": output_tokens1,
                "total_tokens_current_agent": total_tokens1,
                "input_tokens_current_agent_all_attempts": input_tokens1_all_attempts,
                "output_tokens_current_agent_all_attempts": output_tokens1_all_attempts,
                "total_tokens_current_agent_all_attempts": total_tokens1_all_attempts,
                "retry_count_current_agent": retry_count_current_agent,
                "fallback_used_current_agent": fallback_used_current_agent,
                "input_tokens_prediction": input_prediction_tokens,
                "output_tokens_prediction": output_prediction_tokens,
                "total_tokens_prediction": total_prediction_tokens,
                "input_tokens_prediction_all_attempts": input_prediction_tokens_all_attempts,
                "output_tokens_prediction_all_attempts": output_prediction_tokens_all_attempts,
                "total_tokens_prediction_all_attempts": total_prediction_tokens_all_attempts,
                "input_tokens_speculation": input_speculation_tokens,
                "output_tokens_speculation": output_speculation_tokens,
                "total_tokens_speculation": total_speculation_tokens,
                "input_tokens_speculation_all_attempts": input_speculation_tokens_all_attempts,
                "output_tokens_speculation_all_attempts": output_speculation_tokens_all_attempts,
                "total_tokens_speculation_all_attempts": total_speculation_tokens_all_attempts,
                "retry_count_speculation": retry_count_speculation,
                "speculation_branch_results": speculation_branch_results,
            }

            if self.logger:
                self.logger.log("INFO", f"STEP {step_count}:", Utils.dict_to_str(steps_info[step_count]))
                self.logger.log("-" * 100)

            done, _ = self.env.step(current_move)
            step_count += 1
            if stop_after and step_count >= stop_after:
                break

        rewards, game_info = self.env.close()
        run_metrics = {
            "target_stop_after": stop_after,
            "steps_completed": step_count,
            "complete_target_steps": bool(stop_after and step_count >= stop_after),
            "total_effective_wall_time": total_effective_wall_time,
            "total_actor_wall_time_executed": total_actor_wall_time_executed,
            "total_speculation_wall_time": total_speculation_wall_time,
            "total_tokens_effective": total_tokens_effective,
            "reused_steps": reused_steps,
        }
        return steps_info, rewards, game_info, run_metrics

    def run(self, stop_after: int = 20) -> None:
        self.current_run_id = str(uuid.uuid4())
        self.logger = GameLogger(self.base_traj_path, self.current_run_id)
        current_dir_path = join(self.base_traj_path, self.current_run_id)
        self.logger.log(
            "INFO",
            f"Starting run {self.current_run_id} actor={self.config.actor_role['model']} speculator={self.config.speculator_role['model']}",
        )
        try:
            steps_info, rewards, game_info, run_metrics = self._execute_game_loop(stop_after=stop_after)
            Utils.save_json(steps_info, join(current_dir_path, "stepsinfo.json"))
            Utils.save_json(rewards, join(current_dir_path, "rewards.json"))
            Utils.save_json(game_info, join(current_dir_path, "game_info.json"))
            Utils.save_json(run_metrics, join(current_dir_path, "run_metrics.json"))
            Utils.save_json(run_metrics["total_effective_wall_time"], join(current_dir_path, "time_checker_speculate.json"))
            Utils.save_json(run_metrics["total_actor_wall_time_executed"], join(current_dir_path, "time_checker_actor_only.json"))
            self.logger.log("INFO", f"Run completed for {self.current_run_id}")
        except Exception as e:
            if self.logger:
                self.logger.log("ERROR", str(e))
            raise


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run speculative chess (paper-like DeepSeek approximation).")
    p.add_argument("--config", default="config.yml", help="Path to config YAML (default: config.yml)")
    p.add_argument("--trajectories-dir", default=None, help="Output directory for trajectories (overrides config)")
    p.add_argument("--stop-after", type=int, default=None, help="Stop after N steps (default: from config)")
    args = p.parse_args()

    config = Config(args.config)
    if args.trajectories_dir is not None:
        config.trajectories_path = args.trajectories_dir.rstrip("/")

    runner = SpeculativeChessRunner(config=config)
    stop_after = args.stop_after if args.stop_after is not None else config.stop_after
    runner.run(stop_after=stop_after)
    print("Run completed")


if __name__ == "__main__":
    main()
