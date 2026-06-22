import concurrent.futures

from .config import planner_model_id, planner_provider, rlm_max_iterations, rlm_max_depth
from .memory import save_session
from .models import hf_model
from .planner import plan_subtasks
from .prompts import SYNTHESIS_PROMPT_TEMPLATE
from .researcher import research_subtask
from .rlm import RLM

DIRECT_SYNTHESIS_LIMIT = 50_000


def _synthesize_direct(query: str, sub_reports: str) -> str:
    model = hf_model(planner_model_id, planner_provider)
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(user_query=query, sub_reports=sub_reports)
    return model([{"role": "user", "content": prompt}]).content


def _synthesize_rlm(query: str, sub_reports: str, log) -> str:
    log("Reports exceed direct synthesis limit — using RLM for hierarchical synthesis.")
    rlm = RLM(
        model=hf_model(planner_model_id, planner_provider),
        max_iterations=rlm_max_iterations,
        max_depth=rlm_max_depth,
        log=log,
    )
    task = (
        "Synthesize these research sub-reports into a single cohesive markdown document. "
        "Write a title and executive summary, integrate findings into a logical narrative, "
        "highlight key insights and contradictions, and end with a deduplicated Sources section. "
        f"The original user query was: {query}"
    )
    return rlm.run(context=sub_reports, task=task)


def run_deep_research(query: str, log=print) -> str:
    subtasks = plan_subtasks(query, log=log)

    reports: dict[str, str] = {}
    log(f"Starting {len(subtasks)} researchers in parallel...")

    def _research_one(subtask: dict) -> tuple[str, str]:
        return subtask["id"], research_subtask(query, subtask, log=log)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        futures = {pool.submit(_research_one, t): t for t in subtasks}
        for future in concurrent.futures.as_completed(futures):
            subtask_id, report = future.result()
            reports[subtask_id] = report

    log("Synthesizing final report...")
    sub_reports = "\n\n---\n\n".join(
        f"## {t['title']}\n\n{reports[t['id']]}" for t in subtasks
    )

    if len(sub_reports) <= DIRECT_SYNTHESIS_LIMIT:
        final = _synthesize_direct(query, sub_reports)
    else:
        final = _synthesize_rlm(query, sub_reports, log)

    save_session(query, subtasks, final)
    log("Research complete.")
    return final
