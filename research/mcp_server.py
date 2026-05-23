"""MCP server exposing deep-research workflows.

The default transport is stdio for Claude Desktop and other local MCP
clients. SSE and streamable HTTP are available for development and
self-hosted deployments.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
from typing import Any

from agno.agent import Agent
from agno.agent._run import RunOutput
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from .config import (
    coordinator_model_id,
    coordinator_provider,
    firecrawl_mcp_url,
    subagent_model_id,
    subagent_provider,
)
from .coordinator import run_deep_research
from .memory import get_session, list_sessions, recall_relevant
from .models import hf_model
from .planner import generate_research_plan
from .task_splitter import split_task_into_subtasks
from .telemetry import RESEARCHER_TOOL_CALL_LIMIT, PipelineMetrics, check_tool_loop

mcp = FastMCP(
    "deep-research-mcp",
    instructions=(
        "Research workflow server for planning, web-backed literature review, "
        "synthesis, source evaluation, and access to saved research sessions."
    ),
)


def _require_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is required for model-backed research tools. "
            "Set it in your environment or .env file."
        )


def _run_model_prompt(prompt: str, *, name: str) -> str:
    _require_hf_token()
    agent = Agent(
        model=hf_model(coordinator_model_id, coordinator_provider),
        name=name,
        markdown=True,
    )
    response = agent.run(prompt)
    return str(response.content)


async def _run_firecrawl_async(prompt: str, *, name: str) -> RunOutput:
    _require_hf_token()
    async with MCPTools(url=firecrawl_mcp_url(), transport="streamable-http") as tools:
        agent = Agent(
            model=hf_model(subagent_model_id, subagent_provider),
            name=name,
            tools=[tools],
            markdown=True,
            tool_call_limit=RESEARCHER_TOOL_CALL_LIMIT,
        )
        return await agent.arun(prompt)


def _run_firecrawl_prompt(prompt: str, *, name: str) -> str:
    """Run async MCP web tools from sync MCP handlers."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        response = pool.submit(asyncio.run, _run_firecrawl_async(prompt, name=name)).result()
    check_tool_loop(response.tools, label=name, log=lambda _message: None)
    return str(response.content)


def _clip(text: str, limit: int = 900) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _json_resource(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.resource(
    "research://sessions",
    name="recent_research_sessions",
    description="Recent saved deep-research sessions from the local memory database.",
    mime_type="application/json",
)
def recent_research_sessions() -> str:
    return _json_resource(list_sessions(limit=20))


@mcp.resource(
    "research://session/{session_id}",
    name="research_session",
    description="Full saved research session by numeric session ID.",
    mime_type="application/json",
)
def research_session(session_id: str) -> str:
    session = get_session(int(session_id))
    if session is None:
        raise ValueError(f"No research session found for id={session_id}")
    return _json_resource(session)


@mcp.resource(
    "research://server",
    name="deep_research_server_profile",
    description="Server capabilities and deployment notes for this MCP integration.",
    mime_type="application/json",
)
def server_profile() -> str:
    return _json_resource(
        {
            "name": "deep-research-mcp",
            "default_transport": "stdio",
            "optional_transports": ["sse", "streamable-http"],
            "primary_tools": [
                "plan_research",
                "run_literature_review",
                "search_past_research",
                "synthesize_findings",
                "evaluate_source",
                "extract_data_from_paper",
            ],
            "required_environment": {
                "HF_TOKEN": "Required for model-backed tools.",
                "FIRECRAWL_API_KEY": (
                    "Required for web search, retrieval, and URL inspection tools."
                ),
            },
        }
    )


@mcp.tool()
def search_past_research(query: str, limit: int = 5) -> dict[str, Any]:
    """Search local saved research sessions using lightweight keyword overlap."""
    matches = recall_relevant(query, limit=limit)
    return {
        "query": query,
        "matches": [
            {
                "id": item["id"],
                "query": item["query"],
                "created_at": item["created_at"],
                "plan_excerpt": _clip(item["plan"]),
                "report_excerpt": _clip(item["report"], 1200),
            }
            for item in matches
        ],
    }


@mcp.tool()
def get_research_session(session_id: int) -> dict[str, Any]:
    """Return a complete saved research session by ID."""
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"No research session found for id={session_id}")
    return session


@mcp.tool()
def plan_research(topic: str) -> dict[str, Any]:
    """Create a structured research plan without running web researchers."""
    _require_hf_token()
    metrics = PipelineMetrics()
    logs: list[str] = []
    plan = generate_research_plan(topic, metrics=metrics, log=logs.append)
    return {"topic": topic, "plan": plan, "logs": logs}


@mcp.tool()
def split_research_plan(research_plan: str) -> dict[str, Any]:
    """Split an existing research plan into non-overlapping research subtasks."""
    _require_hf_token()
    metrics = PipelineMetrics()
    logs: list[str] = []
    subtasks = split_task_into_subtasks(research_plan, metrics=metrics, log=logs.append)
    return {"subtasks": subtasks, "logs": logs}


@mcp.tool()
def run_literature_review(
    topic: str,
    scope: str = "focused",
    depth: str = "standard",
) -> dict[str, Any]:
    """Run the full deep-research pipeline and return a synthesized Markdown report."""
    _require_hf_token()
    firecrawl_mcp_url()
    query = (
        f"Run a {depth} literature review with {scope} scope on this topic:\n\n"
        f"{topic}"
    )
    logs: list[str] = []
    report = run_deep_research(query=query, log=logs.append)
    return {"topic": topic, "scope": scope, "depth": depth, "report": report, "logs": logs}


@mcp.tool()
def synthesize_findings(
    topic: str,
    source_notes: list[str],
    instructions: str = "",
) -> dict[str, Any]:
    """Synthesize supplied notes or paper summaries into one coherent Markdown brief."""
    prompt = f"""
Synthesize the following research notes into a concise, citation-aware Markdown brief.

Topic:
{topic}

Additional instructions:
{instructions or "No additional instructions."}

Source notes:
{chr(10).join(f"--- SOURCE {i + 1} ---{chr(10)}{note}" for i, note in enumerate(source_notes))}

Return:
- Executive summary
- Main findings
- Agreements and contradictions
- Evidence gaps
- Suggested next searches
"""
    return {"topic": topic, "synthesis": _run_model_prompt(prompt, name="mcp_synthesizer")}


@mcp.tool()
def analyze_research_gap(
    field: str,
    known_findings: str = "",
    constraints: str = "",
) -> dict[str, Any]:
    """Identify plausible research gaps and testable opportunities in a field."""
    prompt = f"""
Analyze research gaps in the following field:
{field}

Known findings or prior notes:
{known_findings or "None supplied."}

Constraints:
{constraints or "None supplied."}

Return a Markdown analysis with:
- Current consensus or likely baseline
- Underserved questions
- Why each gap matters
- Feasible study designs or experiments
- Data/source requirements
- Risks and disconfirming evidence to look for
"""
    return {"field": field, "analysis": _run_model_prompt(prompt, name="mcp_gap_analyzer")}


@mcp.tool()
def evaluate_source(
    url: str,
    context: str = "",
    criteria: str = "authority, evidence quality, recency, bias, methodological rigor",
) -> dict[str, Any]:
    """Evaluate a source URL for credibility, using Firecrawl retrieval when needed."""
    if context:
        prompt = f"""
Evaluate this source for research credibility.

URL:
{url}

Criteria:
{criteria}

Provided source context:
{context}

Return a Markdown scorecard with strengths, weaknesses, missing information,
credibility rating, and recommended use in a literature review.
"""
        evaluation = _run_model_prompt(prompt, name="mcp_source_evaluator")
    else:
        prompt = f"""
Open and inspect this source, then evaluate it for research credibility.

URL:
{url}

Criteria:
{criteria}

Return a Markdown scorecard with strengths, weaknesses, missing information,
credibility rating, and recommended use in a literature review. Cite what you
observed from the page rather than guessing.
"""
        evaluation = _run_firecrawl_prompt(prompt, name="mcp_source_evaluator")
    return {"url": url, "criteria": criteria, "evaluation": evaluation}


@mcp.tool()
def extract_data_from_paper(
    source: str,
    fields: list[str],
    source_is_url: bool = True,
) -> dict[str, Any]:
    """Extract requested structured fields from a paper URL or supplied paper text."""
    field_list = "\n".join(f"- {field}" for field in fields)
    if source_is_url:
        prompt = f"""
Retrieve the paper or article at this URL and extract the requested fields.

URL:
{source}

Fields:
{field_list}

Return JSON-compatible Markdown with each field, extracted value, confidence,
and the evidence snippet or page section used.
"""
        extraction = _run_firecrawl_prompt(prompt, name="mcp_data_extractor")
    else:
        prompt = f"""
Extract the requested fields from the supplied paper text.

Fields:
{field_list}

Paper text:
{source}

Return JSON-compatible Markdown with each field, extracted value, confidence,
and the evidence snippet used.
"""
        extraction = _run_model_prompt(prompt, name="mcp_data_extractor")
    return {"fields": fields, "source_is_url": source_is_url, "extraction": extraction}


@mcp.tool()
def generate_literature_map(topic: str, max_sources: int = 12) -> dict[str, Any]:
    """Search a topic and produce a relationship map of papers, claims, and themes."""
    prompt = f"""
Search for up to {max_sources} important sources about this topic:
{topic}

Build a literature map in Markdown:
- Cluster sources by theme or method
- Name the central claims in each cluster
- Show relationships, dependencies, contradictions, and missing links
- Include a Mermaid graph if the relationships are clear enough
- Include URLs for sources you used
"""
    return {
        "topic": topic,
        "max_sources": max_sources,
        "literature_map": _run_firecrawl_prompt(prompt, name="mcp_literature_mapper"),
    }


@mcp.tool()
def generate_research_proposal(
    idea: str,
    constraints: str = "",
    audience: str = "academic reviewer",
) -> dict[str, Any]:
    """Turn a research idea into a structured proposal draft."""
    prompt = f"""
Draft a research proposal for this idea:
{idea}

Audience:
{audience}

Constraints:
{constraints or "None supplied."}

Return Markdown with:
- Title
- Research question
- Motivation
- Related-work positioning
- Hypotheses
- Methodology
- Data and evaluation plan
- Timeline
- Risks and mitigations
"""
    return {"idea": idea, "proposal": _run_model_prompt(prompt, name="mcp_proposal_writer")}


@mcp.tool()
def create_research_timeline(
    project_scope: str,
    horizon: str = "12 weeks",
    milestones: str = "",
) -> dict[str, Any]:
    """Create a milestone timeline for a research project."""
    prompt = f"""
Create a practical research timeline.

Project scope:
{project_scope}

Horizon:
{horizon}

Required or preferred milestones:
{milestones or "None supplied."}

Return a Markdown timeline with phases, milestones, deliverables,
dependencies, decision points, and review checkpoints.
"""
    return {
        "project_scope": project_scope,
        "timeline": _run_model_prompt(prompt, name="mcp_timeline_planner"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deep-research MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=os.environ.get("DEEP_RESEARCH_MCP_TRANSPORT", "stdio"),
        help="MCP transport to use. Defaults to stdio.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("DEEP_RESEARCH_MCP_HOST", "127.0.0.1"),
        help="Host for SSE or streamable HTTP transports.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DEEP_RESEARCH_MCP_PORT", "8000")),
        help="Port for SSE or streamable HTTP transports.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
