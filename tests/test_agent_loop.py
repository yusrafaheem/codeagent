import tempfile
import unittest
from pathlib import Path

from codeagent.agent import Agent
from codeagent.llm import MockLLM
from codeagent.workspace import Workspace


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = Workspace(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_end_to_end_task_creates_a_file_and_finishes(self):
        script = [
            {
                "thought": "I'll create hello.py",
                "tool": "write_file",
                "args": {"path": "hello.py", "content": "print('hi')\n"},
            },
            {
                "thought": "task complete",
                "tool": "finish",
                "args": {"summary": "created hello.py"},
            },
        ]
        llm = MockLLM(script)
        agent = Agent(llm=llm, workspace=self.ws)

        result = agent.run("create a hello world script", max_steps=10)

        self.assertTrue(result.success)
        self.assertEqual(result.summary, "created hello.py")
        self.assertEqual(result.steps_taken, 2)
        self.assertEqual((self.root / "hello.py").read_text(), "print('hi')\n")

    def test_multi_step_task_reads_then_edits_then_finishes(self):
        (self.root / "app.py").write_text("DEBUG = True\n")
        script = [
            {"thought": "check current state", "tool": "read_file", "args": {"path": "app.py"}},
            {
                "thought": "flip the flag",
                "tool": "edit_file",
                "args": {"path": "app.py", "old": "DEBUG = True", "new": "DEBUG = False"},
            },
            {"thought": "done", "tool": "finish", "args": {"summary": "disabled debug mode"}},
        ]
        agent = Agent(llm=MockLLM(script), workspace=self.ws)

        result = agent.run("turn off debug mode", max_steps=10)

        self.assertTrue(result.success)
        self.assertEqual((self.root / "app.py").read_text(), "DEBUG = False\n")
        self.assertEqual(len(result.history.steps), 3)

    def test_stops_after_max_steps_if_finish_is_never_called(self):
        # A pathological script that just keeps listing the directory
        # forever -- the agent must not loop past its budget.
        script = [{"thought": "look again", "tool": "list_dir", "args": {}} for _ in range(50)]
        agent = Agent(llm=MockLLM(script), workspace=self.ws)

        result = agent.run("do something", max_steps=5)

        self.assertFalse(result.success)
        self.assertEqual(result.steps_taken, 5)
        self.assertIn("max_steps", result.summary)

    def test_unknown_tool_name_is_recorded_as_a_failed_step_and_loop_continues(self):
        script = [
            {"thought": "oops", "tool": "fly_to_the_moon", "args": {}},
            {"thought": "recover", "tool": "finish", "args": {"summary": "gave up on the moon"}},
        ]
        agent = Agent(llm=MockLLM(script), workspace=self.ws)

        result = agent.run("do something impossible", max_steps=10)

        self.assertTrue(result.success)
        self.assertFalse(result.history.steps[0].success)
        self.assertIn("unknown tool", result.history.steps[0].observation)

    def test_tool_call_with_missing_required_argument_is_recorded_as_failure(self):
        script = [
            {"thought": "forgot the content arg", "tool": "write_file", "args": {"path": "x.txt"}},
            {"thought": "give up", "tool": "finish", "args": {"summary": "couldn't write file"}},
        ]
        agent = Agent(llm=MockLLM(script), workspace=self.ws)

        result = agent.run("write something", max_steps=10)

        self.assertTrue(result.success)  # the agent still reaches finish
        self.assertFalse(result.history.steps[0].success)  # but the write step itself failed
        self.assertFalse((self.root / "x.txt").exists())

    def test_recovers_from_a_single_malformed_json_response(self):
        script = [
            "this is not json at all",
            {"thought": "retry with valid json", "tool": "finish", "args": {"summary": "recovered"}},
        ]
        agent = Agent(llm=MockLLM(script), workspace=self.ws, max_parse_retries=2)

        result = agent.run("do something", max_steps=10)

        self.assertTrue(result.success)
        self.assertEqual(result.summary, "recovered")

    def test_gives_up_after_exhausting_parse_retries(self):
        script = ["garbage 1", "garbage 2", "garbage 3"]
        agent = Agent(llm=MockLLM(script), workspace=self.ws, max_parse_retries=2)

        result = agent.run("do something", max_steps=10)

        self.assertFalse(result.success)
        self.assertIn("could not produce a parseable action", result.summary)

    def test_tolerates_markdown_fenced_json_response(self):
        script = [
            '```json\n{"thought": "fenced", "tool": "finish", "args": {"summary": "ok"}}\n```',
        ]
        agent = Agent(llm=MockLLM(script), workspace=self.ws)

        result = agent.run("do something", max_steps=10)

        self.assertTrue(result.success)
        self.assertEqual(result.summary, "ok")

    def test_denied_shell_command_is_recorded_but_does_not_crash_the_loop(self):
        script = [
            {"thought": "try something dangerous", "tool": "run_shell", "args": {"command": "rm -rf /"}},
            {"thought": "give up on that", "tool": "finish", "args": {"summary": "refused to run it"}},
        ]
        agent = Agent(llm=MockLLM(script), workspace=self.ws)

        result = agent.run("delete everything", max_steps=10)

        self.assertTrue(result.success)
        self.assertFalse(result.history.steps[0].success)
        self.assertIn("safety policy", result.history.steps[0].observation)


if __name__ == "__main__":
    unittest.main()
