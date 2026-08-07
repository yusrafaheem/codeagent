import tempfile
import unittest
from pathlib import Path

from codeagent.workspace import PathEscapesWorkspaceError, Workspace


class TestWorkspace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = Workspace(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_resolves_a_simple_relative_path(self):
        resolved = self.ws.resolve("foo.txt")
        self.assertEqual(resolved, self.root / "foo.txt")

    def test_resolves_nested_relative_path(self):
        resolved = self.ws.resolve("a/b/c.txt")
        self.assertEqual(resolved, (self.root / "a" / "b" / "c.txt").resolve())

    def test_dot_and_empty_string_resolve_to_root(self):
        self.assertEqual(self.ws.resolve("."), self.root)
        self.assertEqual(self.ws.resolve(""), self.root)

    def test_rejects_dotdot_escape(self):
        with self.assertRaises(PathEscapesWorkspaceError):
            self.ws.resolve("../outside.txt")

    def test_rejects_deeply_nested_dotdot_escape(self):
        with self.assertRaises(PathEscapesWorkspaceError):
            self.ws.resolve("a/b/../../../outside.txt")

    def test_rejects_absolute_path_outside_root(self):
        with self.assertRaises(PathEscapesWorkspaceError):
            self.ws.resolve("/etc/passwd")

    def test_allows_absolute_path_that_is_actually_inside_root(self):
        # An absolute path that happens to point back inside the root
        # (e.g. the agent echoing a path it got from list_dir) should
        # still resolve fine -- only *escaping* the root is rejected.
        inside = str(self.root / "nested" / "file.txt")
        resolved = self.ws.resolve(inside)
        self.assertEqual(resolved, (self.root / "nested" / "file.txt").resolve())

    def test_exists_reflects_real_filesystem_state(self):
        self.assertFalse(self.ws.exists("missing.txt"))
        (self.root / "present.txt").write_text("hi")
        self.assertTrue(self.ws.exists("present.txt"))

    def test_relative_is_the_inverse_of_resolve(self):
        target = self.ws.resolve("a/b.txt")
        self.assertEqual(self.ws.relative(target), "a/b.txt")

    def test_constructor_rejects_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            Workspace(self.root / "does_not_exist")

    def test_constructor_rejects_a_file_as_root(self):
        file_path = self.root / "im_a_file.txt"
        file_path.write_text("x")
        with self.assertRaises(NotADirectoryError):
            Workspace(file_path)


if __name__ == "__main__":
    unittest.main()
