from __future__ import annotations

from smolagents import DuckDuckGoSearchTool, VisitWebpageTool

from .config import agent_model_id, agent_provider, rlm_max_iterations, rlm_max_depth
from .models import hf_model
from .prompts import SUBAGENT_PROMPT_TEMPLATE
from .rlm import RLM

_search = DuckDuckGoSearchTool()
_visit = VisitWebpageTool()


def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Returns a results summary."""
    return _search(query)


def visit_page(url: str) -> str:
    """Fetch a web page and return its content as markdown."""
    return _visit(url)


def research_subtask(query: str, subtask: dict, log=print) -> str:
    sid = subtask["id"]
    log(f"Researcher [{sid}] starting: {subtask['title']}...")

    task = SUBAGENT_PROMPT_TEMPLATE.format(
        user_query=query,
        subtask_id=sid,
        subtask_title=subtask["title"],
        subtask_description=subtask["description"],
    )

    rlm = RLM(
        model=hf_model(agent_model_id, agent_provider),
        max_iterations=rlm_max_iterations,
        max_depth=rlm_max_depth,
        log=log,
    )
    result = rlm.run(
        context=subtask["description"],
        task=task,
        extra_tools={"web_search": web_search, "visit_page": visit_page},
    )

    log(f"Researcher [{sid}] done.")
    return result
