import tempfile
import unittest
from pathlib import Path

from codeagent import tools
from codeagent.safety import SafetyPolicy
from codeagent.workspace import Workspace


class TestTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = Workspace(self.root)
        self.safety = SafetyPolicy()

    def tearDown(self):
        self._tmp.cleanup()

    # -- read_file --

    def test_read_file_returns_content(self):
        (self.root / "a.txt").write_text("hello world")
        result = tools.read_file(self.ws, self.safety, path="a.txt")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "hello world")

    def test_read_file_missing_file_fails_cleanly(self):
        result = tools.read_file(self.ws, self.safety, path="missing.txt")
        self.assertFalse(result.success)

    def test_read_file_on_a_directory_fails_cleanly(self):
        (self.root / "adir").mkdir()
        result = tools.read_file(self.ws, self.safety, path="adir")
        self.assertFalse(result.success)

    def test_read_file_rejects_path_escape(self):
        result = tools.read_file(self.ws, self.safety, path="../outside.txt")
        self.assertFalse(result.success)

    def test_read_file_rejects_binary_content(self):
        (self.root / "bin.dat").write_bytes(b"\xff\xfe\x00\x01binary")
        result = tools.read_file(self.ws, self.safety, path="bin.dat")
        self.assertFalse(result.success)

    # -- write_file --

    def test_write_file_creates_new_file(self):
        result = tools.write_file(self.ws, self.safety, path="new.txt", content="content here")
        self.assertTrue(result.success)
        self.assertEqual((self.root / "new.txt").read_text(), "content here")

    def test_write_file_creates_parent_directories(self):
        result = tools.write_file(self.ws, self.safety, path="a/b/c.txt", content="deep")
        self.assertTrue(result.success)
        self.assertEqual((self.root / "a" / "b" / "c.txt").read_text(), "deep")

    def test_write_file_overwrites_existing_content(self):
        (self.root / "f.txt").write_text("old")
        tools.write_file(self.ws, self.safety, path="f.txt", content="new")
        self.assertEqual((self.root / "f.txt").read_text(), "new")

    def test_write_file_rejects_path_escape(self):
        result = tools.write_file(self.ws, self.safety, path="../evil.txt", content="x")
        self.assertFalse(result.success)
        self.assertFalse((self.root.parent / "evil.txt").exists())

    def test_write_file_rejects_oversized_content(self):
        huge = "x" * (tools.MAX_FILE_BYTES + 1)
        result = tools.write_file(self.ws, self.safety, path="huge.txt", content=huge)
        self.assertFalse(result.success)
        self.assertFalse((self.root / "huge.txt").exists())

    # -- edit_file --

    def test_edit_file_replaces_unique_match(self):
        (self.root / "e.py").write_text("def foo():\n    return 1\n")
        result = tools.edit_file(self.ws, self.safety, path="e.py", old="return 1", new="return 2")
        self.assertTrue(result.success)
        self.assertEqual((self.root / "e.py").read_text(), "def foo():\n    return 2\n")

    def test_edit_file_fails_when_old_text_not_found(self):
        (self.root / "e.py").write_text("def foo():\n    return 1\n")
        result = tools.edit_file(self.ws, self.safety, path="e.py", old="return 99", new="return 2")
        self.assertFalse(result.success)
        # file is untouched
        self.assertEqual((self.root / "e.py").read_text(), "def foo():\n    return 1\n")

    def test_edit_file_fails_when_old_text_not_unique(self):
        (self.root / "e.py").write_text("x = 1\nx = 1\n")
        result = tools.edit_file(self.ws, self.safety, path="e.py", old="x = 1", new="x = 2")
        self.assertFalse(result.success)
        self.assertIn("2 occurrences", result.output)
        self.assertEqual((self.root / "e.py").read_text(), "x = 1\nx = 1\n")

    def test_edit_file_missing_file_fails_cleanly(self):
        result = tools.edit_file(self.ws, self.safety, path="missing.py", old="a", new="b")
        self.assertFalse(result.success)

    # -- list_dir --

    def test_list_dir_lists_files_and_dirs(self):
        (self.root / "sub").mkdir()
        (self.root / "b.txt").write_text("x")
        result = tools.list_dir(self.ws, self.safety, path=".")
        self.assertTrue(result.success)
        self.assertIn("d sub", result.output)
        self.assertIn("f b.txt", result.output)

    def test_list_dir_on_empty_directory(self):
        (self.root / "empty").mkdir()
        result = tools.list_dir(self.ws, self.safety, path="empty")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "(empty directory)")

    def test_list_dir_on_a_file_fails_cleanly(self):
        (self.root / "f.txt").write_text("x")
        result = tools.list_dir(self.ws, self.safety, path="f.txt")
        self.assertFalse(result.success)

    # -- run_shell --

    def test_run_shell_executes_allowed_command(self):
        result = tools.run_shell(self.ws, self.safety, command="echo hi_there")
        self.assertTrue(result.success)
        self.assertIn("hi_there", result.output)

    def test_run_shell_reports_nonzero_exit_as_failure(self):
        result = tools.run_shell(self.ws, self.safety, command="python3 -c \"import sys; sys.exit(1)\"")
        self.assertFalse(result.success)
        self.assertIn("[exit code 1]", result.output)

    def test_run_shell_rejects_denied_command_without_running_it(self):
        result = tools.run_shell(self.ws, self.safety, command="rm -rf /")
        self.assertFalse(result.success)
        self.assertIn("safety policy", result.output)

    def test_run_shell_runs_in_the_workspace_root(self):
        (self.root / "marker.txt").write_text("present")
        result = tools.run_shell(self.ws, self.safety, command="cat marker.txt")
        self.assertTrue(result.success)
        self.assertIn("present", result.output)

    # -- search --

    def test_search_finds_matching_lines_with_line_numbers(self):
        (self.root / "a.py").write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
        result = tools.search(self.ws, self.safety, pattern=r"^def ", path=".")
        self.assertTrue(result.success)
        self.assertIn("a.py:1: def foo():", result.output)
        self.assertIn("a.py:4: def bar():", result.output)

    def test_search_with_no_matches(self):
        (self.root / "a.py").write_text("nothing interesting here\n")
        result = tools.search(self.ws, self.safety, pattern=r"needle", path=".")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "no matches")

    def test_search_rejects_invalid_regex(self):
        result = tools.search(self.ws, self.safety, pattern="[unclosed", path=".")
        self.assertFalse(result.success)

    def test_search_skips_binary_files_without_crashing(self):
        (self.root / "bin.dat").write_bytes(b"\xff\xfe\x00\x01")
        (self.root / "text.txt").write_text("needle here\n")
        result = tools.search(self.ws, self.safety, pattern="needle", path=".")
        self.assertTrue(result.success)
        self.assertIn("text.txt:1:", result.output)

    # -- finish --

    def test_finish_returns_the_summary_as_output(self):
        result = tools.finish(self.ws, self.safety, summary="all done")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "all done")


if __name__ == "__main__":
    unittest.main()
