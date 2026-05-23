import tomllib
import os
from pathlib import Path

_config_path = Path(__file__).resolve().parent.parent / "deep-research-config.toml"

with open(_config_path, "rb") as f:
    _cfg = tomllib.load(f)["app"]

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")


def firecrawl_mcp_url() -> str:
    """Return the Firecrawl MCP URL, or fail with setup guidance."""
    api_key = os.environ.get("FIRECRAWL_API_KEY") or FIRECRAWL_API_KEY
    if not api_key:
        raise RuntimeError(
            "FIRECRAWL_API_KEY is required for web research tools. "
            "Set it in your environment or .env file."
        )
    return f"https://mcp.firecrawl.dev/{api_key}/v2/mcp/"


MCP_URL = firecrawl_mcp_url() if FIRECRAWL_API_KEY else ""

model_id                 = _cfg["model_id"]
model_provider           = _cfg["model_provider"]
task_planner_model_id    = _cfg["task_planner_model_id"]
task_planner_provider    = _cfg["task_planner_provider"]
coordinator_model_id     = _cfg["coordinator_model_id"]
coordinator_provider     = _cfg.get("coordinator_provider", "novita")
subagent_model_id        = _cfg["subagent_model_id"]
subagent_provider        = _cfg.get("subagent_provider", "novita")
