"""Recursive Language Model — REPL-based inference loop.

Based on: "Recursive LLMs" (arXiv:2512.24601).
The LLM generates Python code executed in a persistent REPL. It can call
llm_query() to invoke itself (or a sub-RLM) on programmatically constructed
prompts, enabling Ω(|P|) semantic work over inputs far exceeding the context
window.
"""

from __future__ import annotations

import io
import re
import traceback
from contextlib import redirect_stdout
from typing import Callable

REPL_INSTRUCTIONS = """\
You are operating inside a Python REPL. You write code, it gets executed, and
you see a short summary of stdout. Variables persist between steps.

Available:
{available}

Strategy:
1. Inspect the context — check its length, preview sections.
2. If manageable, process directly with llm_query().
3. If large, chunk it, call llm_query() per chunk, and aggregate.
4. Store intermediate results in variables.
5. When done, call FINAL(your_answer_string).

Rules:
- Wrap ALL code in a single ```python ... ``` block per step.
- Do NOT print the full context — it may be huge. Use slicing.
- Use llm_query() for any reasoning over text.
- Call FINAL(result) exactly once when finished.
"""


class _FinalSignal(Exception):
    """Raised inside the REPL to break out of the exec loop."""

    def __init__(self, value: str):
        self.value = value


def _extract_code(text: str) -> str | None:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _metadata(text: str, max_preview: int = 300) -> str:
    if len(text) <= max_preview:
        return text
    return f"[{len(text)} chars] {text[:max_preview]}…"


class RLM:
    """Recursive Language Model engine.

    Parameters
    ----------
    model : InferenceClientModel
        smolagents model used for LLM calls.
    max_iterations : int
        Max REPL steps before forced termination.
    max_depth : int
        Max recursion depth for llm_query sub-calls.
    log : callable
        Logging function (default: print).
    """

    def __init__(
        self,
        model,
        max_iterations: int = 12,
        max_depth: int = 1,
        log: Callable = print,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.log = log

    def run(
        self,
        context: str,
        task: str,
        extra_tools: dict[str, Callable] | None = None,
    ) -> str:
        """Run the RLM loop.

        Parameters
        ----------
        context : str
            The (potentially very large) input text.
        task : str
            A short description of what to do with the context.
        extra_tools : dict
            Additional callables exposed in the REPL (e.g. web_search).
        """
        return self._run_at_depth(context, task, extra_tools or {}, depth=0)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _make_llm_query(self, depth: int, extra_tools: dict) -> Callable:
        def llm_query(prompt: str) -> str:
            """Query an LLM. At depth < max_depth this spawns a sub-RLM."""
            if depth < self.max_depth:
                sub = RLM(
                    self.model,
                    max_iterations=self.max_iterations,
                    max_depth=self.max_depth,
                    log=self.log,
                )
                return sub._run_at_depth(prompt, "Process this.", extra_tools, depth + 1)
            resp = self.model([{"role": "user", "content": prompt}])
            return resp.content

        return llm_query

    def _run_at_depth(
        self,
        context: str,
        task: str,
        extra_tools: dict,
        depth: int,
    ) -> str:
        final_value: str | None = None

        def FINAL(value):  # noqa: N802
            nonlocal final_value
            final_value = str(value)
            raise _FinalSignal(final_value)

        repl_globals: dict = {
            "__builtins__": __builtins__,
            "context": context,
            "llm_query": self._make_llm_query(depth, extra_tools),
            "FINAL": FINAL,
            **extra_tools,
        }

        tool_lines = [
            "- `context` (str): The full input text.",
            "- `llm_query(prompt: str) -> str`: Query an LLM on arbitrary text.",
            "- `FINAL(answer: str)`: Return your final answer and stop.",
        ]
        for name, func in extra_tools.items():
            doc = (getattr(func, "__doc__", "") or "").strip().split("\n")[0]
            tool_lines.append(f"- `{name}(...)`: {doc}")

        system = REPL_INSTRUCTIONS.format(available="\n".join(tool_lines))
        prefix = f"[RLM d={depth}]" if depth > 0 else "[RLM]"

        history: list[dict] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n\n"
                    f"Context is in variable `context` ({len(context)} chars). "
                    f"Preview:\n{context[:500]}"
                ),
            },
        ]

        for i in range(self.max_iterations):
            self.log(f"{prefix} step {i + 1}/{self.max_iterations}")

            resp = self.model(history)
            code = _extract_code(resp.content)

            if code is None:
                return resp.content

            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    exec(code, repl_globals)  # noqa: S102
            except _FinalSignal:
                self.log(f"{prefix} done.")
                return final_value  # type: ignore[return-value]
            except Exception:
                buf.write(traceback.format_exc())

            stdout = buf.getvalue()
            history.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            history.append(
                {"role": "user", "content": f"[Executed] stdout:\n{_metadata(stdout)}"}
            )

        self.log(f"{prefix} max iterations reached.")
        for key in ("answer", "result", "output", "final_answer"):
            val = repl_globals.get(key)
            if isinstance(val, str) and val:
                return val
        return "RLM: max iterations reached without producing a final answer."
