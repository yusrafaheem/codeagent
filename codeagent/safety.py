"""Gate on what shell commands the agent is allowed to run.

This is deliberately conservative and allowlist-first: a command is only
runnable if its first word matches something in ``allowed_commands``, AND
it doesn't match any pattern in ``denied_patterns``. Both checks apply --
being on the allowlist does not exempt a command from the denylist (e.g.
``git`` is allowed, but ``git push --force`` is still blocked by the
denylist below).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

DEFAULT_ALLOWED_COMMANDS = frozenset(
    {
        "python", "python3", "pip",
        "ls", "cat", "echo", "head", "tail", "wc", "grep", "find",
        "mkdir", "touch", "cp", "mv",
        "git",
        "pytest", "unittest",
        "black", "ruff", "flake8", "mypy",
        "node", "npm",
        "go",
    }
)

# Patterns checked against the *whole* command string (not just argv[0]).
# These win even if the base command is allowlisted above.
DEFAULT_DENIED_PATTERNS = (
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f\b"),   # rm -rf, rm -fr, rm -Rf, ...
    re.compile(r"\brm\s+-[a-z]*f[a-z]*r\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bsu\b"),
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if="),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"\bcurl\b.*\|\s*(sh|bash)\b"),
    re.compile(r"\bwget\b.*\|\s*(sh|bash)\b"),
    re.compile(r":\(\)\s*\{.*:\|:.*\}"),        # classic fork bomb
    re.compile(r"\bgit\s+push\b.*--force\b"),
    re.compile(r"\bgit\s+push\b.*-f\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
)


class CommandNotAllowedError(Exception):
    """Raised when a command fails the safety policy check."""


@dataclass
class SafetyPolicy:
    allowed_commands: frozenset[str] = field(default_factory=lambda: DEFAULT_ALLOWED_COMMANDS)
    denied_patterns: tuple[re.Pattern, ...] = DEFAULT_DENIED_PATTERNS
    timeout_seconds: float = 30.0

    def check(self, command: str) -> None:
        """Raise CommandNotAllowedError if ``command`` violates the policy.
        Does nothing (returns None) if the command is permitted."""
        stripped = command.strip()
        if not stripped:
            raise CommandNotAllowedError("empty command")

        for pattern in self.denied_patterns:
            if pattern.search(stripped):
                raise CommandNotAllowedError(
                    f"command matches a denied pattern ({pattern.pattern!r}): {command!r}"
                )

        try:
            tokens = shlex.split(stripped)
        except ValueError as e:
            raise CommandNotAllowedError(f"could not parse command: {e}") from e

        if not tokens:
            raise CommandNotAllowedError("empty command")

        base = tokens[0]
        # Strip a leading path so "./venv/bin/python" matches "python".
        base_name = base.rsplit("/", 1)[-1]
        if base_name not in self.allowed_commands:
            raise CommandNotAllowedError(
                f"command {base_name!r} is not on the allowlist: {command!r}"
            )

        # Shell metacharacters that chain into a second command need each
        # sub-command checked too, since "echo hi && rm -rf /" has an
        # allowlisted first word but a denied second one.
        for connector in ("&&", "||", ";", "|"):
            if connector in tokens:
                parts = command.split(connector)
                for part in parts[1:]:
                    self.check(part)
