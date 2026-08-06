"""Prompt templates: how the task and tool vocabulary are described to the
model, and how prior steps get folded back in as conversation turns.
"""

from __future__ import annotations

from .history import History

TOOL_DESCRIPTIONS = """\
read_file(path)                    -- read a text file's full contents
write_file(path, content)          -- create or overwrite a text file
edit_file(path, old, new)          -- replace one unique occurrence of `old` with `new`
list_dir(path=".")                 -- list a directory's immediate contents
run_shell(command)                 -- run a shell command in the workspace root
search(pattern, path=".")          -- regex search across files under path
finish(summary)                    -- declare the task complete, with a summary"""

SYSTEM_PROMPT = f"""\
You are codeagent, an autonomous coding agent. You are given a task and a \
workspace directory. On each turn you choose exactly one tool call to make \
progress, then you're shown the result and choose the next one, until you \
call finish.

Available tools:
{TOOL_DESCRIPTIONS}

Respond with a single JSON object and nothing else, in this exact shape:
{{"thought": "<why you're taking this action>", "tool": "<tool name>", "args": {{...}}}}

Do not wrap the JSON in markdown code fences. Do not include any text \
before or after the JSON object."""


def build_messages(task: str, history: History) -> list[dict[str, str]]:
    """Assemble the full message list for the next LLM call: a system
    prompt, the task as the first user turn, and every prior step
    rendered as an assistant/user exchange so the model can see what it
    already tried."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]

    for step in history.steps:
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f'{{"thought": {step.thought!r}, "tool": "{step.tool}", '
                    f'"args": {step.args!r}}}'
                ),
            }
        )
        status = "ok" if step.success else "error"
        messages.append(
            {"role": "user", "content": f"Observation ({status}): {step.observation}"}
        )

    return messages
