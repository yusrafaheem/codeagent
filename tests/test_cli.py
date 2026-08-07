import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from codeagent.cli import build_arg_parser, main


class TestCli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_script(self, actions) -> str:
        script_path = self.root / "script.json"
        script_path.write_text(json.dumps(actions))
        return str(script_path)

    def test_run_with_mock_provider_executes_the_scripted_task_successfully(self):
        script_path = self._write_script(
            [
                {"thought": "write it", "tool": "write_file", "args": {"path": "out.txt", "content": "hi"}},
                {"thought": "done", "tool": "finish", "args": {"summary": "wrote out.txt"}},
            ]
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "run",
                    "write a file",
                    "--workdir", str(self.root),
                    "--provider", "mock",
                    "--script", script_path,
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("SUCCESS", stdout.getvalue())
        self.assertEqual((self.root / "out.txt").read_text(), "hi")

    def test_run_with_mock_provider_and_no_script_errors_out(self):
        with self.assertRaises(SystemExit):
            main(["run", "do something", "--workdir", str(self.root), "--provider", "mock"])

    def test_run_returns_nonzero_exit_code_when_agent_does_not_finish(self):
        script_path = self._write_script(
            [{"thought": "loop", "tool": "list_dir", "args": {}}]
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(IndexError):
                # max-steps of 5 with only 1 scripted action exhausts the
                # MockLLM script before the step budget -- that's a genuine
                # misconfiguration (script too short), and it should
                # surface as an error rather than silently succeeding.
                main(
                    [
                        "run",
                        "loop forever",
                        "--workdir", str(self.root),
                        "--provider", "mock",
                        "--script", script_path,
                        "--max-steps", "5",
                    ]
                )

    def test_arg_parser_requires_a_subcommand(self):
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_arg_parser_defaults_provider_to_mock(self):
        parser = build_arg_parser()
        args = parser.parse_args(["run", "task", "--workdir", "."])
        self.assertEqual(args.provider, "mock")
        self.assertEqual(args.max_steps, 20)


if __name__ == "__main__":
    unittest.main()
