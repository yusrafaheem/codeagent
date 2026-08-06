# codeagent

A small, dependency-light autonomous coding agent: an observe -> think ->
act loop that reads a task, decides on one tool call at a time (read a
file, edit it, run a shell command, ...), and keeps going until it
declares the task finished or runs out of step budget.

It's deliberately not trying to be a full IDE-integrated assistant.
The goal here was to build the *loop itself* correctly: a jailed
workspace an agent can't escape, a shell command allowlist it can't
talk its way around, precise (not fuzzy) file editing, and a fully
deterministic mock LLM backend so the entire loop -- including the
retry-on-malformed-output path -- is unit-testable without a network
connection or an API key.

## Quickstart

```bash
pip install -e .
codeagent run "find and fix the bug in calc.py" \
    --workdir ./some/project \
    --provider mock \
    --script examples/demo_actions.json
```

See [`examples/demo_transcript.md`](examples/demo_transcript.md) for a
real, unedited run of this exact command.

To drive it with a real model instead of a scripted mock:

```bash
pip install -e ".[openai]"     # or .[anthropic]
export OPENAI_API_KEY=...
codeagent run "add type hints to utils.py" --workdir ./some/project --provider openai
```

## Architecture

```
task string
     |
     v
+-----------+      messages       +-----------+
|  prompts  |  ----------------->  |    LLM    |
| (system + |                      | (mock /   |
|  history) |  <-----------------  |  openai / |
+-----------+   one JSON action    | anthropic)|
     ^                             +-----------+
     |
+-----------+     dispatch      +-----------+
|   Agent   |  --------------->  |   tools   |
| (the loop)|  <---------------  | (read/    |
+-----------+   ToolResult       |  write/   |
     |                           |  edit/... |
     v                           +-----------+
+-----------+                          |
|  History  |                          v
| (log +    |                   +------------+   +------------+
| checkpoint|                   | Workspace  |   |  Safety    |
| /undo)    |                   | (path jail)|   |  Policy    |
+-----------+                   +------------+   +------------+
```

Each turn: `prompts.build_messages()` renders the task plus every prior
step as a conversation, `llm.complete()` returns one raw string, `agent`
parses it as a single JSON action (`{"thought", "tool", "args"}`), looks
up the matching function in `tools.TOOLS`, and calls it with the
`Workspace` and `SafetyPolicy` as the first two arguments. The tool
result (success + text output) becomes the next turn's observation.
`finish` is the one action that ends the loop instead of producing an
observation.

### Why a jailed workspace and a command allowlist

An autonomous agent with file and shell access is, by construction, a
program that executes instructions produced by another program (the
LLM) that you don't fully control the output of. Two things follow from
that:

1. **Every file path needs to be validated before touching disk.**
   `Workspace.resolve()` is the single choke point every file tool goes
   through -- it rejects `../` escapes, absolute paths outside the root,
   and (via `Path.resolve()`) symlinks that point outside the root. No
   tool function does its own path arithmetic; they all just call
   `workspace.resolve(path)` and let it raise.

2. **Not all syntactically valid shell commands should run.**
   `SafetyPolicy` is allowlist-first (a command's base name must be on
   the list) *and* denylist-checked (specific dangerous patterns --
   `rm -rf`, `sudo`, `curl | sh`, force-pushes, fork bombs -- are
   blocked even for an allowlisted base command), and it recursively
   re-checks each side of a `&&`/`||`/`;`/`|` chain, because
   `echo hi && rm -rf /` has an innocuous first word.

Both are deliberately conservative defaults, not a claim of being
unbreakable -- see **Known limitations** below.

### edit_file's uniqueness requirement

`edit_file(path, old, new)` replaces exactly one occurrence of `old`,
and refuses to guess: zero matches or more than one match is an error,
not a best-effort replace-first-or-replace-all. An agent (or the human
prompting it) that only gives enough context to match two different
places in a file gets told that plainly, rather than silently editing
the wrong one. This mirrors the same discipline a careful find-and-replace
tool should have.

### MockLLM: testing an LLM-driven loop without an LLM

`llm.MockLLM` takes a list of pre-scripted actions and replays them one
per call, in order, ignoring the actual prompt content (though it still
records every `messages` argument it was called with, for tests that
want to assert on what the agent sent). Script entries can be either a
dict (JSON-encoded as a normal action would be) or a raw string
(returned verbatim) -- the latter is what makes it possible to test the
"model returned unparseable garbage" recovery path deterministically:
script `["not json", {"tool": "finish", ...}]` and assert the agent
retries once and then succeeds.

This is also why `tests/test_agent_loop.py` can assert on the *entire*
loop -- multi-step tasks, max-step cutoffs, unknown-tool handling,
malformed-JSON retries, denied shell commands -- without any network
access, non-determinism, or API cost.

### History and undo

`History` records every step (thought, tool, args, success, observation)
and, before any mutating tool call (`write_file`, `edit_file`,
`run_shell`), snapshots the text content of every file currently in the
workspace. `history.undo(root, n)` restores file contents to how they
were before the last `n` mutating steps and drops those steps from the
log.

**Known limitation, documented rather than hidden:** `undo()` restores
file *contents* it had already seen, but won't delete a file a step
newly *created* -- the snapshot taken before that step simply didn't
know the file would exist yet. For an agent whose most common failure
mode is "edited the wrong thing" rather than "created a stray file",
that's an intentional trade-off: a full restore would mean diffing the
entire directory tree (not just tracking one snapshot dict) on every
single step, for a safety net that's needed far less often than a
straightforward content revert.

## Testing philosophy

All 84 tests run with the standard library's `unittest` (`python3 -m
unittest discover -s tests`) -- no `pytest`, no mocking framework beyond
the project's own `MockLLM`, and everything actually executes real code
against a real temp-directory filesystem (`tempfile.TemporaryDirectory`),
not stubs. A few specific choices worth calling out:

- `test_safety.py` tests the denylist against real attack-shaped strings
  (`rm -rf /`, `sudo ...`, `curl ... | bash`, a literal fork bomb,
  `git push --force`) rather than abstract cases, because the whole
  point of the policy is to catch exactly these.
- `test_tools.py`'s `run_shell` tests actually spawn a subprocess
  (`echo`, a failing `python3 -c`) and check real stdout/exit codes --
  and one test constructs a workspace with a real binary file to check
  `read_file`/`search` don't crash on a `UnicodeDecodeError`.
- `test_agent_loop.py` exercises the *whole* stack end-to-end: a
  scripted multi-step task that reads a file, edits it, and finishes;
  a pathological script that never calls `finish` (asserts the
  `max_steps` cutoff actually stops it); a scripted unknown tool name;
  a scripted tool call missing a required argument; and the
  malformed-JSON retry-then-recover path.
- `examples/demo_transcript.md` is the literal, unedited stdout of a
  real `codeagent run` invocation, not hand-written prose -- easy to
  regenerate and verify with `--provider mock --script
  examples/demo_actions.json`.

## Known limitations

- `run_shell`'s safety policy is a best-effort allow/deny list, not a
  sandbox -- it makes obviously-dangerous commands harder to run by
  accident, it does not make arbitrary code execution safe. Anything
  genuinely untrusted should still run this inside a container or VM,
  not rely on `SafetyPolicy` as the only boundary.
- `Workspace` jails *paths*, not process capabilities -- an allowlisted
  command like `python3` can still do whatever a Python interpreter can
  do once it's running (e.g. `python3 -c "open('/etc/passwd')"`  isn't
  a path the workspace tools ever resolved, so it isn't caught).
- `History.undo()`'s file-creation gap, described above.
- There's no token budget or context-window trimming -- `prompts.py`
  includes every prior step verbatim in every request. Fine for the
  scale this was built and tested at; a long-running task against a
  real model would eventually need to summarize or truncate older
  steps.

## Project layout

```
codeagent/
  workspace.py   -- path-jailed filesystem root (Workspace, PathEscapesWorkspaceError)
  safety.py      -- shell command allow/deny policy (SafetyPolicy, CommandNotAllowedError)
  tools.py       -- read_file, write_file, edit_file, list_dir, run_shell, search, finish
  llm.py         -- LLMClient protocol, MockLLM, OpenAIClient, AnthropicClient
  history.py     -- step log + checkpoint/undo (History, Step)
  prompts.py     -- system prompt + message-building (build_messages)
  agent.py       -- the loop itself (Agent, AgentResult)
  cli.py         -- `codeagent run ...` entrypoint
tests/           -- 84 unittest tests, one file per module above
examples/        -- a real, reproducible demo run
```

## License

MIT -- see [LICENSE](LICENSE).
