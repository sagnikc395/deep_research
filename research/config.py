import tomllib
from pathlib import Path

_config_path = Path(__file__).resolve().parent.parent / "deep-research-config.toml"

with open(_config_path, "rb") as f:
    _raw = tomllib.load(f)

_cfg = _raw["app"]
_rlm = _raw.get("rlm", {})

planner_model_id    = _cfg["planner_model_id"]
planner_provider    = _cfg["planner_provider"]
agent_model_id      = _cfg["agent_model_id"]
agent_provider      = _cfg["agent_provider"]

rlm_max_iterations  = _rlm.get("max_iterations", 15)
rlm_max_depth       = _rlm.get("max_depth", 2)
planning_max_depth  = _rlm.get("planning_max_depth", 2)
