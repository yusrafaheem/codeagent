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
    """The uniform return type every tool function produces.

    ``success`` is what the agent loop checks to decide how to log the
    step; ``output`` is always a string (never structured data) because it
    goes straight back into the next LLM prompt as an observation.
    """

    success: bool
    output: str

    def __str__(self) -> str:  # convenient for logging/prompts
        return self.output


def _err(message: str) -> ToolResult:
    """Shorthand for the ``ToolResult(success=False, ...)`` every tool
    returns on a handled failure (bad path, missing file, etc.) -- as
    opposed to an unhandled exception, which the agent loop catches
    separately as a malformed tool call."""
    return ToolResult(success=False, output=message)


def read_file(workspace: Workspace, safety: SafetyPolicy, path: str) -> ToolResult:
    """Return the full UTF-8 text contents of ``path``.

    Fails cleanly (rather than raising) on the three predictable ways this
    can go wrong: the path escapes the workspace, the path doesn't exist,
    or it exists but isn't valid UTF-8 text (e.g. a binary file).
    """
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
    """Create ``path`` (and any missing parent directories) or overwrite it
    entirely with ``content``. Unlike edit_file, this has no notion of
    "existing content" -- it's the right tool for a brand-new file, and the
    wrong one for a small change to a file the agent hasn't fully re-read,
    since it silently discards whatever was there before.
    """
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
    """List the immediate (non-recursive) contents of ``path``, one entry
    per line prefixed with ``d`` or ``f``. Directories sort before files at
    each name so an agent scanning the output can spot subdirectories
    worth exploring at a glance.
    """
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
    """Run ``command`` in a subprocess rooted at the workspace directory,
    after ``safety.check()`` clears it against the allow/deny policy.

    stdout and stderr are both captured and folded into a single output
    string (stderr labeled separately) along with the exit code, so the
    agent sees everything a human running the command in a terminal would.
    A command that exceeds ``safety.timeout_seconds`` is treated as a
    failure rather than left to hang the loop.
    """
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
    """Regex-search (Python ``re`` syntax) for ``pattern`` across every
    readable text file under ``path`` (a single file is fine too),
    returning ``relative/path:lineno: line`` per match -- close to
    ``grep -rn`` output, which is the format an agent is most likely to
    have seen in training. Files that fail to decode as UTF-8 are skipped
    rather than erroring the whole search.
    """
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
