"""Command-line entrypoint: `codeagent run "<task>" --workdir DIR [...]`"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent
from .llm import AnthropicClient, LLMError, MockLLM, OpenAIClient
from .safety import SafetyPolicy
from .workspace import Workspace


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeagent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent on a task")
    run.add_argument("task", help="natural-language description of what to do")
    run.add_argument("--workdir", required=True, help="workspace directory the agent may touch")
    run.add_argument(
        "--provider",
        choices=["mock", "openai", "anthropic"],
        default="mock",
        help="which LLM backend to use (default: mock)",
    )
    run.add_argument(
        "--script",
        help="path to a JSON file of scripted actions, required when --provider=mock",
    )
    run.add_argument("--model", help="model name override for openai/anthropic providers")
    run.add_argument("--max-steps", type=int, default=20, help="step budget before giving up")

    return parser


def _build_llm(args: argparse.Namespace):
    if args.provider == "mock":
        if not args.script:
            raise SystemExit("--provider=mock requires --script <path to JSON action list>")
        with open(args.script, encoding="utf-8") as f:
            script = json.load(f)
        return MockLLM(script)

    if args.provider == "openai":
        kwargs = {"model": args.model} if args.model else {}
        return OpenAIClient(**kwargs)

    if args.provider == "anthropic":
        kwargs = {"model": args.model} if args.model else {}
        return AnthropicClient(**kwargs)

    raise SystemExit(f"unknown provider: {args.provider}")


def run_command(args: argparse.Namespace) -> int:
    try:
        llm = _build_llm(args)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    workspace = Workspace(args.workdir)
    agent = Agent(llm=llm, workspace=workspace, safety=SafetyPolicy())

    result = agent.run(args.task, max_steps=args.max_steps)

    print(result.history.transcript())
    print()
    print(f"{'SUCCESS' if result.success else 'STOPPED'} after {result.steps_taken} step(s)")
    print(result.summary)

    return 0 if result.success else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
