"""codeagent: a small, dependency-light autonomous coding agent.

Core pieces:
    workspace.Workspace   -- path-jailed filesystem root the agent operates in
    safety.SafetyPolicy   -- allow/deny rules for shell commands
    tools                 -- read_file, write_file, edit_file, list_dir,
                              run_shell, search, finish
    llm.LLMClient         -- protocol for pluggable model backends
    llm.MockLLM           -- deterministic, scripted backend used for tests
                              and offline demos
    agent.Agent           -- the observe/think/act loop tying it together
    history.History       -- step log + checkpoint/undo support
"""

__version__ = "0.1.0"
