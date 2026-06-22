# deep-research

A multi-agent deep research system built on open-source models. Give it a question; it plans the investigation, fans out parallel web-research agents, and produces a single polished Markdown report.

Like OpenAI's Deep Research or Perplexity, but local, inspectable, and running entirely on open-source inference.

## Why

Researching a topic usually means opening thirty tabs, skimming half of them, and ending up with scattered notes. Existing "deep research" products solve this but are black boxes.

`deep-research` rebuilds the core loop (**plan, search, read, synthesize**) as a transparent multi-agent system on open-source models, following [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) and [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The core insight: multiple specialized agents working in parallel outperform a single agent trying to do everything.

## Stack

| Component          | Tool                                                      |
| ------------------ | --------------------------------------------------------- |
| Inference          | Hugging Face Inference API (Novita, Together)             |
| Agent framework    | [smolagents](https://github.com/huggingface/smolagents)   |
| Web search         | DuckDuckGo via smolagents `DuckDuckGoSearchTool`          |
| Web scraping       | smolagents `VisitWebpageTool`                             |
| Core engine        | RLM (Recursive Language Model) REPL loop                  |
| UI                 | [Textual](https://textual.textualize.io/) TUI / [Click](https://click.palletsprojects.com/) CLI |
| Package manager    | [uv](https://docs.astral.sh/uv/)                         |

Default models (configurable in `deep-research-config.toml`):

- **Planner / Synthesizer**: `deepseek-ai/DeepSeek-V3.2-Exp` (via Together)
- **Researchers**: `MiniMaxAI/MiniMax-M1-80k` (via Novita)

## Architecture

![deep-research-architecture](./static/deep-research-v0.1.png)

Three stages, each a separate module. All LLM calls go through smolagents' `InferenceClientModel`, routed to open-source models via the Hugging Face Inference API.

### 1. Planning (`research/planner.py`)

Takes the user question and produces **non-overlapping subtasks** via Pydantic structured output (`SubtaskList`). Past sessions from memory are prepended as context.

Supports **recursive decomposition**: subtasks marked `decompose=true` are split into finer children, up to `planning_max_depth`. Planning and splitting happen in a single recursive pass.

### 2. Research (`research/researcher.py`)

One researcher per subtask, all running in parallel via `ThreadPoolExecutor`. Each researcher is an **RLM** (Recursive Language Model): a REPL loop where the LLM writes Python code executed in a persistent sandbox. The REPL exposes:

- `web_search()` for DuckDuckGo queries
- `visit_page()` for scraping web pages
- `llm_query()` for recursive sub-calls on constructed prompts

This lets researchers handle inputs exceeding the context window by chunking, summarizing, and aggregating across REPL steps. See [arXiv:2512.24601](https://arxiv.org/abs/2512.24601).

### 3. Synthesis (`research/coordinator.py`)

Once all researchers report back:

- **Direct**: if combined sub-reports fit within 50k chars, a single LLM call produces the report.
- **Hierarchical**: if they exceed the limit, an RLM instance synthesizes recursively.

Output is saved to `results.md`.

## Repository layout

```
deep-research/
├── main.py                          # Textual TUI entrypoint
├── cli.py                           # Click CLI entrypoint
├── deep-research-config.toml        # Model / provider configuration
├── research/
│   ├── models.py                    # smolagents model factory
│   ├── coordinator.py               # Pipeline orchestrator + synthesis
│   ├── planner.py                   # Recursive planning and splitting
│   ├── researcher.py                # Per-subtask RLM research agent
│   ├── rlm.py                       # Recursive Language Model engine
│   ├── prompts.py                   # Prompt template loader
│   ├── config.py                    # Config loader
│   └── memory.py                    # Persistent session memory (SQLite)
├── prompts/
│   ├── planner_system_instructions.md
│   ├── subagent_prompt.md
│   └── synthesis_prompt.md
└── static/
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

**TUI** (interactive):

```bash
uv run python main.py
```

**CLI** (scriptable):

```bash
uv run python cli.py "your research query"
uv run python cli.py "your query" -o report.md   # custom output path
```

Progress logs print to stderr. If no query argument is given, the CLI prompts for one.

## Configuration

`deep-research-config.toml`:

```toml
[app]
name = "deep-research"
planner_model_id   = "deepseek-ai/DeepSeek-V3.2-Exp"
planner_provider   = "together"
agent_model_id     = "MiniMaxAI/MiniMax-M1-80k"
agent_provider     = "novita"

[rlm]
max_iterations     = 15
max_depth          = 2
planning_max_depth = 2
```

| Section | Key | Purpose |
| ------- | --- | ------- |
| `[app]` | `planner_model_id`, `planner_provider` | Model for planning and synthesis |
| `[app]` | `agent_model_id`, `agent_provider` | Model for research sub-agents |
| `[rlm]` | `max_iterations` | REPL steps per RLM run |
| `[rlm]` | `max_depth` | Recursive sub-RLM levels allowed |
| `[rlm]` | `planning_max_depth` | Recursive planner decomposition depth |

Provider values: `"novita"`, `"together"`, `"auto"` (HF default routing).

## RLM engine

`research/rlm.py` implements the Recursive Language Model pattern ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)). Instead of a single LLM call, the model operates in a persistent Python REPL:

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

## Roadmap

- [x] Memory support
- [x] Parallel researchers
- [x] RLM engine
- [x] Hierarchical synthesis
- [x] Recursive planning
- [x] CLI interface
- [ ] [Obsidian integration](https://github.com/sagnikc395/obscure) for autonomous knowledge-gap filling
- [ ] Cost tracking per run
- [ ] Token-usage telemetry

## References

- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (Anthropic)
- [Recursive LLMs](https://arxiv.org/abs/2512.24601) (arXiv:2512.24601)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [smolagents](https://github.com/huggingface/smolagents) (Hugging Face)

## License

MIT. See [LICENSE](./LICENSE).
