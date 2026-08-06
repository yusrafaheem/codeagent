"""Path-jailed filesystem root the agent is allowed to touch.

Every file tool goes through Workspace.resolve() before doing any I/O.
This is the single choke point that stops an agent (or a bug in the LLM's
tool-call output) from reading or writing outside the directory it was
given -- no ``../../etc/passwd``, no absolute-path escapes, no following a
symlink out of the jail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PathEscapesWorkspaceError(Exception):
    """Raised when a requested path would resolve outside the workspace root."""


@dataclass
class Workspace:
    root: Path

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"workspace root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"workspace root is not a directory: {self.root}")

    def resolve(self, relative_path: str) -> Path:
        """Resolve ``relative_path`` against the workspace root and verify
        the result is still inside the root.

        Raises PathEscapesWorkspaceError for absolute paths, ``..`` escapes,
        and symlinks that resolve outside the root.
        """
        if relative_path in ("", "."):
            return self.root

        candidate = (self.root / relative_path).resolve()

        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PathEscapesWorkspaceError(
                f"path {relative_path!r} resolves to {candidate}, "
                f"which is outside the workspace root {self.root}"
            ) from None

        return candidate

    def exists(self, relative_path: str) -> bool:
        return self.resolve(relative_path).exists()

    def relative(self, absolute_path: Path) -> str:
        """Inverse of resolve(): render an absolute path (inside the
        workspace) as a path relative to the root, for display purposes."""
        return str(Path(absolute_path).resolve().relative_to(self.root))
