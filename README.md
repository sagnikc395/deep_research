# deep-research

A multi-agent deep research system built on open-source models. Give it a question; it plans the investigation, fans out parallel web-research agents, and produces a single polished Markdown report.

Like OpenAI's Deep Research or Perplexity, but local, inspectable, and running entirely on open-source inference.

## Why

Researching a topic usually means opening thirty tabs, skimming half of them, and ending up with scattered notes. Existing "deep research" products solve this but are black boxes.

`deep-research` rebuilds the core loop (**plan, search, read, synthesize**) as a transparent multi-agent system on open-source models, following [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) and [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The core insight: multiple specialized agents working in parallel outperform a single agent trying to do everything.

## Stack

| Component       | Tool                                                    |
| --------------- | ------------------------------------------------------- |
| Inference       | Hugging Face Inference API (Novita, Together)           |
| Agent framework | [smolagents](https://github.com/huggingface/smolagents) |
| Web search      | DuckDuckGo via smolagents `DuckDuckGoSearchTool`        |
| Web scraping    | smolagents `VisitWebpageTool`                           |
| Core engine     | RLM (Recursive Language Model) REPL loop                |
| CLI             | [Click](https://click.palletsprojects.com/)             |
| Package manager | [uv](https://docs.astral.sh/uv/)                       |

Default models (configurable in `config.py`):

- **Planner / Synthesizer**: `Qwen/Qwen3.6-27B`
- **Researchers**: `MiniMaxAI/MiniMax-M1-80k` (via Novita)

## Architecture

```
User Query (CLI)
      |
      v
+------------------+
|     Planner      |  LLM call -> recursive subtask decomposition (Pydantic structured output)
+------------------+
      |
      v
+------------------------------+
|        Coordinator           |  ThreadPoolExecutor -> spawns N RLM researchers in parallel
|  +----------+ +----------+   |
|  | RLM      | | RLM      | ..|  Each: RLM REPL loop + web_search + visit_page + llm_query
|  +----------+ +----------+   |
+------------------------------+
      |
      v
+------------------+
|   Synthesizer    |  Direct LLM call (< 50k chars) or hierarchical RLM synthesis
+------------------+
      |
      v
  results.md + SQLite memory
```

### 1. Planning (`deep_research/planner.py`)

Takes the user question and produces **non-overlapping subtasks** via Pydantic structured output (`SubtaskList`). Past sessions from memory are prepended as context.

Supports **recursive decomposition**: subtasks marked `decompose=true` are split into finer children, up to `planning_max_depth`. Planning and splitting happen in a single recursive pass.

### 2. Research (`deep_research/researcher.py`)

One researcher per subtask, all running in parallel via `ThreadPoolExecutor`. Each researcher is an **RLM** (Recursive Language Model): a REPL loop where the LLM writes Python code executed in a persistent sandbox. The REPL exposes:

- `web_search()` for DuckDuckGo queries
- `visit_page()` for scraping web pages
- `llm_query()` for recursive sub-calls on constructed prompts

This lets researchers handle inputs exceeding the context window by chunking, summarizing, and aggregating across REPL steps. See [arXiv:2512.24601](https://arxiv.org/abs/2512.24601).

### 3. Synthesis (`deep_research/coordinator.py`)

Once all researchers report back:

- **Direct**: if combined sub-reports fit within 50k chars, a single LLM call produces the report.
- **Hierarchical**: if they exceed the limit, an RLM instance synthesizes recursively.

Output is saved to `results.md` and the session is persisted to SQLite memory for future context.

### Evolution from v1

The original architecture (documented in `docs/v1.md`) used smolagents' `ToolCallingAgent` with JSON tool calls for each sub-agent, a separate `task_splitter.py` stage, and a `collector.py` for final synthesis. The current version replaces all of that with the RLM engine — a persistent Python REPL loop that gives each researcher programmatic control over chunking, aggregation, and recursive sub-queries, enabling work over inputs far exceeding the context window.

## Repository layout

```
deep-research/
├── cli.py                           # Click CLI entrypoint
├── config.py                        # Default model / provider constants
├── deep_research/
│   ├── __init__.py
│   ├── config.py                    # Config loader (reads deep-research-config.toml if present)
│   ├── coordinator.py               # Pipeline orchestrator + synthesis
│   ├── manager.py                   # Multi-agent management (placeholder)
│   ├── memory.py                    # Persistent session memory (SQLite)
│   ├── models.py                    # smolagents model factory
│   ├── planner.py                   # Recursive planning and splitting
│   ├── prompts.py                   # Prompt template loader
│   ├── researcher.py                # Per-subtask RLM research agent
│   └── rlm.py                       # Recursive Language Model engine
├── prompts/
│   ├── planner_system_instructions.md
│   ├── subagent_prompt.md
│   └── synthesis_prompt.md
├── docs/
│   └── v1.md                        # Original v1 architecture documentation
└── static/
    └── deep-research-v0.1.png
```

## Setup

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)
- A [Hugging Face](https://huggingface.co/) API token

### Install

```bash
git clone https://github.com/sagnikc395/deep-research.git
cd deep-research
uv sync
cp .env.sample .env
```

Fill in `.env`:

```
HF_TOKEN="your-huggingface-token"
```

### Run

```bash
uv run python cli.py "your research query"
uv run python cli.py "your query" -o report.md   # custom output path
```

Progress logs print to stderr. If no query argument is given, the CLI prompts for one.

## Configuration

Model and RLM parameters are set in `config.py` at the project root:

```python
# Planning stage
PLANNER_MODE_ID  = "Qwen/Qwen3.6-27B"
PLANNER_PROVIDER = ""

# Research sub-agents
AGENT_MODEL_ID   = "MiniMaxAI/MiniMax-M1-80k"
AGENT_PROVIDER   = ""

# RLM depth config
MAX_ITERATIONS     = 15
MAX_DEPTH          = 2
PLANNING_MAX_DEPTH = 2
```

If a `deep-research-config.toml` file is present, `deep_research/config.py` reads from it instead, using the `[app]` and `[rlm]` sections.

| Key | Purpose |
| --- | ------- |
| `PLANNER_MODE_ID`, `PLANNER_PROVIDER` | Model for planning and synthesis |
| `AGENT_MODEL_ID`, `AGENT_PROVIDER` | Model for research sub-agents |
| `MAX_ITERATIONS` | REPL steps per RLM run |
| `MAX_DEPTH` | Recursive sub-RLM levels allowed |
| `PLANNING_MAX_DEPTH` | Recursive planner decomposition depth |

Provider values: `""` (HF default routing), `"novita"`, `"together"`.

## RLM engine

`deep_research/rlm.py` implements the Recursive Language Model pattern ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)). Instead of a single LLM call, the model operates in a persistent Python REPL:

1. The LLM sees the task, a context preview, and available tools.
2. Each step, it emits a `python` code block executed in-process.
3. Variables persist between steps for inspection, chunking, and aggregation.
4. `llm_query(prompt)` spawns a sub-RLM at the next depth level.
5. `FINAL(answer)` returns the result and exits.
6. A hard cap of `max_iterations` prevents runaway loops.

This lets a single researcher process inputs far exceeding the context window by programmatically chunking and summarizing.

## Design notes

- **The planner determines everything.** Overlapping subtasks cause redundancy; too-broad ones produce shallow summaries; too-narrow ones miss the bigger picture. Recursive decomposition with a depth cap is the main quality lever.
- **RLM over single-shot agents.** The REPL loop lets the model inspect context size, chunk strategically, and aggregate across steps. Tradeoff: more LLM round-trips per researcher.
- **Hierarchical synthesis.** When combined sub-reports exceed the context window, the RLM engine synthesizes recursively rather than truncating.
- **Parallel agents need guardrails.** Without constraints, sub-agents chase the same sources. URL-level dedup helps; semantic overlap remains an open problem.
- **Open-source models struggle most at synthesis.** Planning, searching, and extraction all work well. Weaving ten mini-reports into a cohesive narrative is where the gap with frontier models is most visible.
- **Coordinator as plain Python.** A Python loop is simpler and more deterministic than an LLM-as-coordinator. The LLM is only used where it adds real value.
- **Watch thread counts.** Each researcher gets its own thread. `max_workers=len(subtasks)` works for typical splits (4-8 subtasks) but should be capped for larger lists.
- **Session memory.** Past research sessions are stored in SQLite and recalled during planning via keyword overlap, giving the planner context from prior runs.

## References

- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (Anthropic)
- [Recursive LLMs](https://arxiv.org/abs/2512.24601) (arXiv:2512.24601)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [smolagents](https://github.com/huggingface/smolagents) (Hugging Face)

## License

MIT. See [LICENSE](./LICENSE).
