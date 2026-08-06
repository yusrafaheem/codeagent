"""The observe -> think -> act loop.

Each turn: build a prompt from the task + prior steps, ask the LLM for one
JSON-encoded action, dispatch it to the matching tool, record what
happened, repeat -- until the model calls finish, a max step budget is
hit, or the model can't produce a parseable action even after a couple of
retries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .history import History
from .llm import LLMClient
from .prompts import build_messages
from .safety import SafetyPolicy
from .tools import TOOLS, ToolResult
from .workspace import Workspace

MUTATING_TOOLS = frozenset({"write_file", "edit_file", "run_shell"})


@dataclass
class AgentResult:
    success: bool
    summary: str
    steps_taken: int
    history: History


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        workspace: Workspace,
        safety: SafetyPolicy | None = None,
        max_parse_retries: int = 2,
    ):
        self.llm = llm
        self.workspace = workspace
        self.safety = safety or SafetyPolicy()
        self.max_parse_retries = max_parse_retries

    def run(self, task: str, max_steps: int = 20) -> AgentResult:
        history = History()

        for step_index in range(max_steps):
            action = self._get_action(task, history)
            if action is None:
                return AgentResult(
                    success=False,
                    summary=(
                        "agent gave up: could not produce a parseable action "
                        f"after {self.max_parse_retries} retries"
                    ),
                    steps_taken=step_index,
                    history=history,
                )

            thought = action.get("thought", "")
            tool_name = action.get("tool")
            args = action.get("args") or {}

            if tool_name == "finish":
                summary = args.get("summary", "")
                history.record(thought, "finish", args, True, summary)
                return AgentResult(
                    success=True, summary=summary, steps_taken=step_index + 1, history=history
                )

            tool_fn = TOOLS.get(tool_name)
            if tool_fn is None:
                observation = f"unknown tool: {tool_name!r}"
                history.record(thought, str(tool_name), args, False, observation)
                continue

            if tool_name in MUTATING_TOOLS:
                history.snapshot_before(step_index, self.workspace.root)

            try:
                result: ToolResult = tool_fn(self.workspace, self.safety, **args)
            except TypeError as e:
                result = ToolResult(success=False, output=f"bad arguments for {tool_name}: {e}")

            history.record(thought, tool_name, args, result.success, result.output)

        return AgentResult(
            success=False,
            summary=f"stopped after reaching max_steps={max_steps} without calling finish",
            steps_taken=max_steps,
            history=history,
        )

    def _get_action(self, task: str, history: History) -> dict | None:
        last_error: str | None = None
        for _ in range(self.max_parse_retries + 1):
            messages = build_messages(task, history)
            if last_error:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your last response could not be parsed: {last_error}. "
                            "Respond with a single valid JSON object only."
                        ),
                    }
                )
            raw = self.llm.complete(messages)
            try:
                return _parse_action(raw)
            except ValueError as e:
                last_error = str(e)
        return None


def _parse_action(raw: str) -> dict:
    text = raw.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        action = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from e

    if not isinstance(action, dict):
        raise ValueError("expected a JSON object")
    if "tool" not in action:
        raise ValueError("missing required 'tool' key")

    return action
