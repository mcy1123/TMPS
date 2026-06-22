import os
import json
import shutil
import re
import chess
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

class Utils:
    DEFAULT_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "deepseek": "https://api.deepseek.com",
    }
    API_KEY_ENV_VARS = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    AGENT_NAME_TO_PROVIDER = {
        "openai": "openai",
        "openrouter": "openrouter",
        "deepseek": "deepseek",
    }

    @staticmethod
    def load_env():
        if load_dotenv is None:
            return
        project_env = Path(__file__).resolve().parent / ".env"
        load_dotenv(project_env)
        load_dotenv()

    @staticmethod
    def normalize_provider(provider: str) -> str:
        return (provider or "").strip().lower()

    @staticmethod
    def provider_from_agent_name(name: str) -> str:
        normalized = Utils.normalize_provider(name)
        return Utils.AGENT_NAME_TO_PROVIDER.get(normalized, normalized)

    @staticmethod
    def _expand_config_value(value: str | None) -> str:
        if value is None:
            return ""
        return os.path.expandvars(str(value)).strip()

    @staticmethod
    def get_api_key(config: dict, provider: str) -> str:
        Utils.load_env()
        provider_key = Utils.normalize_provider(provider)
        api_config = config.get("api", {}).get(provider_key, {})
        configured_key = Utils._expand_config_value(api_config.get("key", ""))
        env_key = os.getenv(Utils.API_KEY_ENV_VARS.get(provider_key, ""), "")
        return configured_key or env_key

    @staticmethod
    def get_base_url(config: dict, provider: str) -> str:
        provider_key = Utils.normalize_provider(provider)
        api_config = config.get("api", {}).get(provider_key, {})
        return (
            Utils._expand_config_value(api_config.get("base_url", ""))
            or Utils.DEFAULT_BASE_URLS.get(provider_key, "")
        )

    @staticmethod
    def get_model_name(config: dict, provider: str, model_kind: str) -> str:
        provider_key = Utils.normalize_provider(provider)
        return config.get("models", {}).get(provider_key, {}).get(model_kind, "")

    @staticmethod
    def chat_completion_kwargs(
        provider: str,
        model_name: str,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        thinking: dict | None = None,
        response_format: dict | None = None,
    ) -> dict:
        provider_key = Utils.normalize_provider(provider)
        kwargs = {"model": model_name, "messages": messages, "n": 1}
        if provider_key in {"openai", "openrouter"} and reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        elif provider_key in {"openai", "openrouter"} and reasoning_effort is None:
            kwargs["reasoning_effort"] = "low"
        if provider_key == "deepseek" and thinking is not None:
            kwargs["extra_body"] = {"thinking": thinking}
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            kwargs["response_format"] = response_format
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif provider_key == "deepseek":
            kwargs["temperature"] = 0
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif provider_key == "deepseek":
            kwargs["max_tokens"] = 96
        return kwargs

    @staticmethod
    def get_role_config(config: dict, role: str, default_provider: str, default_model_kind: str) -> dict[str, Any]:
        role_cfg = config.get("agents", {}).get(role, {})
        provider = Utils.provider_from_agent_name(role_cfg.get("provider", default_provider))
        return {
            "provider": provider,
            "model": role_cfg.get("model") or Utils.get_model_name(config, provider, default_model_kind),
            "temperature": role_cfg.get("temperature"),
            "max_tokens": role_cfg.get("max_tokens"),
            "reasoning_effort": role_cfg.get("reasoning_effort"),
            "thinking": role_cfg.get("thinking"),
            "response_format": role_cfg.get("response_format"),
        }

    @staticmethod
    def read_json(path):
        f = open(path, "r")
        output = json.load(f)   
        f.close()
        return output
    
    @staticmethod
    def save_json(obj, path, delete_prev_file=False):
        if os.path.exists(path) and delete_prev_file:
            os.remove(path)
        f = open(path, "w")
        json.dump(obj, f, indent=4)
        f.close()
    
    @staticmethod
    def read_file(path):
        f = open(path, "r", encoding="utf-8")
        output = f.read()
        f.close()
        return output
    
    @staticmethod
    def save_file(string, path, delete_prev_file=False):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

        if os.path.exists(path) and delete_prev_file:
            os.remove(path)
        f = open(path, "w",  encoding="utf-8")
        f.write(string)
        f.close()

    @staticmethod
    def join_prompt(*args):
        output = ""
        for arg in args:
            output += arg
        return output

    @staticmethod
    def append_file(string, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        f = open(path, "a",  encoding="utf-8")
        f.write(string+"\n")
        f.close()
    
    @staticmethod
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
    
    @staticmethod
    def delete_dir(path, nested=False):
        if os.path.isdir(path):
            try:
                os.rmdir(path)
            except OSError as e:
                if nested:
                    shutil.rmtree(path)
                else:
                    raise e
    
    @staticmethod
    def check_all_dirs(base_path):
        for direc in os.listdir(base_path):
            Utils.check_dir(os.path.join(base_path, direc))

    @staticmethod
    def extract_ith_step_info(i, text, trajectory_type):
        pattern = rf"{trajectory_type.upper()} TRAJECTORY:\n Thought {i}: (.*?)\nAction {i}: (.*?)\nObservation {i}: (.*?)\n"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
            observation = match.group(3).strip()
            return {"thought": thought, "action": action, "observation": observation}
        else:
            return {"thought": None, "action": None, "observation": None}

    @staticmethod
    def process_obs(base_path):
        not_processed = []
        for direc in os.listdir(base_path):
            log_path = os.path.join(base_path, direc, "log.txt")
            normal_obs_path = os.path.join(base_path, direc, "normalobs.json")
            sim_obs_path = os.path.join(base_path, direc, "simobs.json")
            logs = Utils.read_file(log_path)
            normal_obs = Utils.read_json(normal_obs_path)
            sim_obs = Utils.read_json(sim_obs_path)
            new_normal_obs = {}
            new_normal_obs["prompt"] = normal_obs["prompt"]
            new_sim_obs = {}
            new_sim_obs["prompt"] = sim_obs["prompt"]
            try:
                normal_first_steps = Utils.extract_ith_step_info(1, logs, "normal")
                sim_first_steps = Utils.extract_ith_step_info(1, logs, "simulation")
                new_normal_obs["actions"] = normal_obs["actions"][normal_obs["actions"].index(normal_first_steps["action"]):]
                new_normal_obs["thoughts"] = normal_obs["thoughts"][normal_obs["thoughts"].index(normal_first_steps["thought"]):]
                new_normal_obs["observations"] = normal_obs["observations"][normal_obs["observations"].index(normal_first_steps["observation"]):]
                new_sim_obs["actions"] = sim_obs["actions"][sim_obs["actions"].index(sim_first_steps["action"]):]
                new_sim_obs["thoughts"] = sim_obs["thoughts"][sim_obs["thoughts"].index(sim_first_steps["thought"]):]
                new_sim_obs["observations"] = sim_obs["observations"][sim_obs["observations"].index(sim_first_steps["observation"]):]
                assert len(new_normal_obs["thoughts"]) == len(new_normal_obs["actions"]) == len(new_normal_obs["observations"]), "lengths of thoughts, actions and observations are unequal for normal obs"
                assert len(new_sim_obs["thoughts"]) == len(new_sim_obs["actions"]) == len(new_sim_obs["observations"]), "lengths of thoughts, actions and observations are unequal for sim obs"
            except Exception as e:
                print(e)
                not_processed.append(direc)                
                continue
            
            Utils.save_json(new_normal_obs, normal_obs_path, delete_prev_file=True)
            Utils.save_json(new_sim_obs, sim_obs_path, delete_prev_file=True)
        print(len(not_processed))
        Utils.save_file(str(not_processed), "./not_processed.txt")

    @staticmethod
    def cleanup_trajs(trajs_folder):
        for direc in os.listdir(trajs_folder):
            datapoint_path = os.path.join(trajs_folder, direc)
            if Utils.is_dirty_traj(datapoint_path):
                print(f"Cleaning up {datapoint_path}")
                Utils.delete_dir(datapoint_path, nested=True)

    @staticmethod
    def is_dirty_traj(datapoint_path):
        files = ["metrics.json", "normalobs.json", "simobs.json", "log.txt"]
        try:
            for file_name in files:
                assert os.path.exists(os.path.join(datapoint_path, file_name)), f"{file_name} is missing"
        except AssertionError as e:
            return True
        return False
    
    @staticmethod
    def dict_to_str(d):
        return ' | '.join([f"{k}: {v}" for k, v in d.items()])

    @staticmethod
    def board_with_coords(board: chess.Board) -> str:
        inner_width = len(str(board).splitlines()[0])
        top = bottom = f"   +{'-' * (inner_width + 2)}+"
        body = [f" {rank} | {row} |" for rank, row in zip(range(8, 0, -1), str(board).splitlines())]
        files = "   " + " ".join("a b c d e f g h".split()).center(inner_width + 2)
        return "\n".join([top, *body, bottom, files])

    @staticmethod
    def build_guess_prompt(
        observation: str,
        guess_prompt_template: str,
        num_guesses: int,
        skill_enabled: bool = False,
        skill_prompt: str = "",
        skill_position: str = "suffix",
    ) -> str:
        """Build the guess-module prompt without changing authority prompts.

        skill_position:
          - "inline": observation + skill + base_prompt (legacy)
          - "suffix": observation + base_prompt + skill (default, keeps legal-move constraint prominent)
        """
        base_prompt = guess_prompt_template.format(num_guesses=num_guesses)
        if not skill_enabled or not skill_prompt.strip():
            return observation + base_prompt
        if skill_position == "inline":
            return observation + "\n\n" + skill_prompt.strip() + "\n\n" + base_prompt
        return observation + base_prompt + "\n\n" + skill_prompt.strip()

    @staticmethod
    def dedupe_preserve_order(items: list[str]) -> list[str]:
        seen = set()
        output: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output

    @staticmethod
    def serialize_valid_moves(valid_moves: list[str]) -> str:
        return ", ".join(valid_moves)



def truncate_chess_observation(observation_string: str) -> str:
    """
    Extract the last board state and valid moves from a chess observation string.
    """
    lines = observation_string.strip().split('\n')
    
    # Find the last board representation
    last_board_start = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Current board:" in lines[i]:
            last_board_start = i
            break
    
    if last_board_start == -1:
        return "No board found in observation"
    
    # Find where the board ends (after the coordinate line "a b c d e f g h")
    board_end = -1
    for i in range(last_board_start + 1, len(lines)):
        if "a b c d e f g h" in lines[i]:
            board_end = i
            break
    
    if board_end == -1:
        return "Could not find end of board"
    
    valid_moves_line = ""
    for i in range(board_end + 1, len(lines)):
        if "Valid moves:" in lines[i]:
            valid_moves_line = lines[i]
            break
    
    truncated_lines = []
    
    if lines[0].startswith("[GAME] You are playing"):
        truncated_lines.append(lines[0])
        truncated_lines.append(lines[1])  # Move instruction line
    
    for i in range(last_board_start, board_end + 1):
        truncated_lines.append(lines[i])
    
    if valid_moves_line:
        truncated_lines.append(valid_moves_line)
    
    return '\n'.join(truncated_lines)
