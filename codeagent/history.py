"""Step log and checkpoint/undo support for a single agent run.

Every step the agent takes -- what it decided to do, and what happened --
is recorded as a Step. Before any mutating tool call (write_file,
edit_file, run_shell), History snapshots the current text-file contents of
the workspace, so a run can be rolled back with undo() if a step turns out
to be wrong.

Known limitation, documented rather than hidden: undo() restores file
*contents* it had already seen, but it won't delete a file that a step
newly created (the snapshot taken before that step simply didn't know the
file would exist). For an agent whose main failure mode is "edited the
wrong thing" rather than "created a stray file", that's the trade-off this
makes -- a full restore would mean diffing the whole directory tree on
every single step, which is a lot of I/O for a rarely-needed safety net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MUTATING_TOOLS = frozenset({"write_file", "edit_file", "run_shell"})


@dataclass
class Step:
    """One turn of the agent loop, as actually recorded (not what the LLM
    was asked for, but what it did and what happened as a result).

    ``index`` doubles as this step's key into History's checkpoint dict --
    it's set to ``len(self.steps)`` at record time, so it's stable and
    matches the step's position in the transcript.
    """

    index: int
    thought: str
    tool: str
    args: dict
    success: bool
    observation: str


@dataclass
class History:
    steps: list[Step] = field(default_factory=list)
    _checkpoints: dict[int, dict[str, str]] = field(default_factory=dict)

    def snapshot_before(self, step_index: int, workspace_root: Path) -> None:
        """Capture all readable text files under the workspace root, keyed
        by step index, before that step runs (if it's a mutating tool)."""
        snapshot: dict[str, str] = {}
        for f in workspace_root.rglob("*"):
            if not f.is_file():
                continue
            try:
                snapshot[str(f.relative_to(workspace_root))] = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
        self._checkpoints[step_index] = snapshot

    def record(self, thought: str, tool: str, args: dict, success: bool, observation: str) -> Step:
        step = Step(
            index=len(self.steps),
            thought=thought,
            tool=tool,
            args=args,
            success=success,
            observation=observation,
        )
        self.steps.append(step)
        return step

    def undo(self, workspace_root: Path, n: int = 1) -> list[str]:
        """Restore file contents to how they were before the last ``n``
        mutating steps. Returns the list of relative paths that were
        restored."""
        if n < 1:
            raise ValueError("n must be >= 1")

        mutating_indices = [s.index for s in self.steps if s.tool in MUTATING_TOOLS]
        to_undo = mutating_indices[-n:]
        if not to_undo:
            return []

        earliest = min(to_undo)
        if earliest not in self._checkpoints:
            raise KeyError(f"no checkpoint recorded for step {earliest}")

        snapshot = self._checkpoints[earliest]
        restored: list[str] = []
        for rel_path, content in snapshot.items():
            target = workspace_root / rel_path
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                restored.append(rel_path)

        # Drop the undone steps and their checkpoints from history so a
        # second undo() doesn't re-target already-reverted steps.
        self.steps = self.steps[:earliest]
        self._checkpoints = {k: v for k, v in self._checkpoints.items() if k < earliest}

        return restored

    def transcript(self) -> str:
        """Render the run so far as plain text, for logging or inclusion
        in a follow-up prompt."""
        lines = []
        for step in self.steps:
            lines.append(f"[step {step.index}] thought: {step.thought}")
            lines.append(f"  action: {step.tool}({step.args})")
            status = "ok" if step.success else "error"
            lines.append(f"  observation ({status}): {step.observation}")
        return "\n".join(lines)
