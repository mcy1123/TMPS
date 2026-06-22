# TMPS: Trajectory-Mined Predictive Skills for Lossless Agent Acceleration

## Quick Reproduction

### Environment

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- API key in `TMPS/chess-game/.env` (e.g., `DEEPSEEK_API_KEY=sk-xxx`)

### Setup

```bash
git clone https://github.com/mcy1123/TMPS.git
cd TMPS/chess-game
uv sync
```

### Run

```bash
# Speculative pipeline (with TMPS skill)
uv run speculative-chess

# Sequential baseline
uv run regular-chess

# TMPS skill evolution pipeline
uv run -m tmps_pipeline.analyst --trajectories <traj_dir> --base-skill skills/v4/SKILL.md
uv run -m tmps_pipeline.consolidator --patches <patches_dir> --output skills/v5/SKILL.md
```
