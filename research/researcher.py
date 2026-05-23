"""Stage 3 — Single-subtask researcher.

Spawns an Agno Agent with Firecrawl MCP tools to research one subtask.

MCP requires an async context (websocket / SSE transport). To stay
compatible with sync callers (including Textual thread workers that
may already have a running event loop), each subtask is executed in a
fresh OS thread via ThreadPoolExecutor so that asyncio.run() always
gets a clean loop.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Optional

from agno.agent import Agent
from agno.agent._run import RunOutput
from agno.tools.mcp import MCPTools

from .config import firecrawl_mcp_url, subagent_model_id, subagent_provider
from .models import hf_model
from .prompts import SUBAGENT_PROMPT_TEMPLATE
from .telemetry import RESEARCHER_TOOL_CALL_LIMIT, PipelineMetrics, check_tool_loop


async def _run_async(prompt: str) -> RunOutput:
    """Run the research agent inside an async MCP context.

    Returns the full RunOutput so callers can inspect metrics and tool
    executions for token-waste and loop detection.
    """
    async with MCPTools(url=firecrawl_mcp_url(), transport="streamable-http") as mcp:
        agent = Agent(
            model=hf_model(subagent_model_id, subagent_provider),
            tools=[mcp],
            markdown=True,
            # Hard cap on tool calls per run to prevent infinite tool loops.
            tool_call_limit=RESEARCHER_TOOL_CALL_LIMIT,
        )
        return await agent.arun(prompt)


def research_subtask(
    query: str,
    plan: str,
    subtask: dict,
    log=print,
    metrics: Optional[PipelineMetrics] = None,
) -> str:
    """Research *subtask* and return a markdown report.

    Runs asynchronous MCP operations in an isolated thread so the
    caller can be either sync or inside an existing event loop.
    """
    sid = subtask["id"]
    log(f"Researcher starting [{sid}] {subtask['title']}...")

    prompt = SUBAGENT_PROMPT_TEMPLATE.format(
        user_query=query,
        research_plan=plan,
        subtask_id=sid,
        subtask_title=subtask["title"],
        subtask_description=subtask["description"],
    )

    # ThreadPoolExecutor gives us a thread with no running event loop,
    # so asyncio.run() always works regardless of the caller context.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, _run_async(prompt))
        response: RunOutput = future.result()

    # ── Post-run checks ───────────────────────────────────────────────────
    check_tool_loop(response.tools, label=f"researcher[{sid}]", log=log)
    if metrics is not None:
        metrics.record(f"researcher[{sid}]", response.metrics, log)

    log(f"Researcher [{sid}] done.")
    return response.content
