import tempfile
import unittest
from pathlib import Path

from codeagent.history import History


class TestHistory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_appends_a_step_with_incrementing_index(self):
        history = History()
        history.record("t1", "list_dir", {}, True, "obs1")
        history.record("t2", "read_file", {"path": "a.txt"}, True, "obs2")
        self.assertEqual([s.index for s in history.steps], [0, 1])
        self.assertEqual(history.steps[1].tool, "read_file")

    def test_transcript_includes_thought_action_and_observation(self):
        history = History()
        history.record("thinking", "list_dir", {"path": "."}, True, "a\nb")
        text = history.transcript()
        self.assertIn("thinking", text)
        self.assertIn("list_dir", text)
        self.assertIn("a\nb", text)

    def test_undo_restores_a_file_modified_by_write_file(self):
        (self.root / "a.txt").write_text("original")
        history = History()

        history.snapshot_before(0, self.root)
        (self.root / "a.txt").write_text("modified")
        history.record("edit it", "write_file", {"path": "a.txt", "content": "modified"}, True, "wrote")

        restored = history.undo(self.root, n=1)
        self.assertIn("a.txt", restored)
        self.assertEqual((self.root / "a.txt").read_text(), "original")

    def test_undo_removes_the_undone_step_from_history(self):
        (self.root / "a.txt").write_text("v0")
        history = History()
        history.snapshot_before(0, self.root)
        (self.root / "a.txt").write_text("v1")
        history.record("t", "write_file", {}, True, "wrote")

        self.assertEqual(len(history.steps), 1)
        history.undo(self.root, n=1)
        self.assertEqual(len(history.steps), 0)

    def test_undo_multiple_steps_restores_the_earliest_checkpoint(self):
        (self.root / "a.txt").write_text("v0")
        history = History()

        history.snapshot_before(0, self.root)
        (self.root / "a.txt").write_text("v1")
        history.record("t1", "write_file", {}, True, "wrote v1")

        history.snapshot_before(1, self.root)
        (self.root / "a.txt").write_text("v2")
        history.record("t2", "write_file", {}, True, "wrote v2")

        history.undo(self.root, n=2)
        self.assertEqual((self.root / "a.txt").read_text(), "v0")
        self.assertEqual(len(history.steps), 0)

    def test_undo_with_no_mutating_steps_is_a_no_op(self):
        history = History()
        history.record("just looking", "list_dir", {}, True, "empty")
        restored = history.undo(self.root, n=1)
        self.assertEqual(restored, [])
        self.assertEqual(len(history.steps), 1)

    def test_undo_only_touches_files_that_actually_changed(self):
        (self.root / "a.txt").write_text("same")
        (self.root / "b.txt").write_text("same")
        history = History()

        history.snapshot_before(0, self.root)
        (self.root / "b.txt").write_text("changed")
        history.record("t", "write_file", {"path": "b.txt"}, True, "wrote")

        restored = history.undo(self.root, n=1)
        self.assertEqual(restored, ["b.txt"])

    def test_undo_rejects_non_positive_n(self):
        history = History()
        with self.assertRaises(ValueError):
            history.undo(self.root, n=0)


if __name__ == "__main__":
    unittest.main()
