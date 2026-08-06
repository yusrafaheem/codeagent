"""The agent's action vocabulary.

Every tool is a plain function of the form
    (workspace, safety, **args) -> ToolResult
so the agent loop can dispatch by name without any tool needing to know
about the agent, the LLM, or the history -- they only ever see a Workspace
(for the file tools) and a SafetyPolicy (for run_shell).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from .safety import CommandNotAllowedError, SafetyPolicy
from .workspace import PathEscapesWorkspaceError, Workspace

MAX_FILE_BYTES = 1_000_000  # refuse to write pathologically large files


@dataclass
class ToolResult:
    success: bool
    output: str

    def __str__(self) -> str:  # convenient for logging/prompts
        return self.output


def _err(message: str) -> ToolResult:
    return ToolResult(success=False, output=message)


def read_file(workspace: Workspace, safety: SafetyPolicy, path: str) -> ToolResult:
    try:
        target = workspace.resolve(path)
    except PathEscapesWorkspaceError as e:
        return _err(str(e))

    if not target.exists():
        return _err(f"no such file: {path}")
    if target.is_dir():
        return _err(f"{path} is a directory, not a file")

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _err(f"{path} is not valid UTF-8 text (binary file?)")

    return ToolResult(success=True, output=content)


def write_file(workspace: Workspace, safety: SafetyPolicy, path: str, content: str) -> ToolResult:
    try:
        target = workspace.resolve(path)
    except PathEscapesWorkspaceError as e:
        return _err(str(e))

    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        return _err(
            f"refusing to write {path}: content exceeds the "
            f"{MAX_FILE_BYTES}-byte limit"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ToolResult(success=True, output=f"wrote {len(content)} chars to {path}")


def edit_file(
    workspace: Workspace, safety: SafetyPolicy, path: str, old: str, new: str
) -> ToolResult:
    """Replace a single, unique occurrence of ``old`` with ``new``.

    Mirrors the discipline of a good find-and-replace tool: if ``old``
    doesn't appear, or appears more than once, this fails loudly instead
    of guessing -- an agent editing the wrong occurrence of a common
    string is a real, easy-to-hit failure mode worth designing against.
    """
    try:
        target = workspace.resolve(path)
    except PathEscapesWorkspaceError as e:
        return _err(str(e))

    if not target.exists():
        return _err(f"no such file: {path}")

    content = target.read_text(encoding="utf-8")
    count = content.count(old)

    if count == 0:
        return _err(f"old text not found in {path}")
    if count > 1:
        return _err(
            f"old text is not unique in {path} ({count} occurrences) -- "
            "include more surrounding context so it matches exactly once"
        )

    target.write_text(content.replace(old, new, 1), encoding="utf-8")
    return ToolResult(success=True, output=f"edited {path}")


def list_dir(workspace: Workspace, safety: SafetyPolicy, path: str = ".") -> ToolResult:
    try:
        target = workspace.resolve(path)
    except PathEscapesWorkspaceError as e:
        return _err(str(e))

    if not target.exists():
        return _err(f"no such directory: {path}")
    if not target.is_dir():
        return _err(f"{path} is a file, not a directory")

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = [f"{'d' if e.is_dir() else 'f'} {e.name}" for e in entries]
    return ToolResult(success=True, output="\n".join(lines) if lines else "(empty directory)")


def run_shell(workspace: Workspace, safety: SafetyPolicy, command: str) -> ToolResult:
    try:
        safety.check(command)
    except CommandNotAllowedError as e:
        return _err(f"command rejected by safety policy: {e}")

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=safety.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return _err(f"command timed out after {safety.timeout_seconds}s: {command}")

    output = proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + "[stderr]\n" + proc.stderr
    output += f"\n[exit code {proc.returncode}]"

    return ToolResult(success=(proc.returncode == 0), output=output)


def search(
    workspace: Workspace, safety: SafetyPolicy, pattern: str, path: str = "."
) -> ToolResult:
    try:
        target = workspace.resolve(path)
    except PathEscapesWorkspaceError as e:
        return _err(str(e))

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return _err(f"invalid regex {pattern!r}: {e}")

    files = [target] if target.is_file() else sorted(target.rglob("*"))
    matches: list[str] = []

    for f in files:
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel = workspace.relative(f)
                matches.append(f"{rel}:{lineno}: {line}")

    if not matches:
        return ToolResult(success=True, output="no matches")
    return ToolResult(success=True, output="\n".join(matches))


def finish(workspace: Workspace, safety: SafetyPolicy, summary: str) -> ToolResult:
    """Terminal action: the agent believes the task is complete."""
    return ToolResult(success=True, output=summary)


TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_dir": list_dir,
    "run_shell": run_shell,
    "search": search,
    "finish": finish,
}
