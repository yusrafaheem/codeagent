import json
import unittest

from codeagent.llm import LLMError, MockLLM


class TestMockLLM(unittest.TestCase):
    def test_returns_scripted_dict_actions_as_json_in_order(self):
        script = [
            {"thought": "first", "tool": "list_dir", "args": {}},
            {"thought": "second", "tool": "finish", "args": {"summary": "done"}},
        ]
        llm = MockLLM(script)

        first = json.loads(llm.complete([]))
        self.assertEqual(first["tool"], "list_dir")

        second = json.loads(llm.complete([]))
        self.assertEqual(second["tool"], "finish")

    def test_string_entries_are_returned_verbatim(self):
        llm = MockLLM(["not valid json at all"])
        self.assertEqual(llm.complete([]), "not valid json at all")

    def test_raises_when_script_is_exhausted(self):
        llm = MockLLM([{"thought": "x", "tool": "finish", "args": {}}])
        llm.complete([])
        with self.assertRaises(IndexError):
            llm.complete([])

    def test_records_every_messages_argument_it_received(self):
        llm = MockLLM([{"thought": "x", "tool": "finish", "args": {}}])
        messages = [{"role": "system", "content": "hi"}]
        llm.complete(messages)
        self.assertEqual(llm.received_messages, [messages])

    def test_calls_made_tracks_progress_through_the_script(self):
        llm = MockLLM(
            [
                {"thought": "a", "tool": "list_dir", "args": {}},
                {"thought": "b", "tool": "finish", "args": {}},
            ]
        )
        self.assertEqual(llm.calls_made, 0)
        llm.complete([])
        self.assertEqual(llm.calls_made, 1)
        llm.complete([])
        self.assertEqual(llm.calls_made, 2)


class TestRealBackendsRequireDependencies(unittest.TestCase):
    def test_openai_client_raises_llm_error_if_package_missing(self):
        # This environment has neither the openai nor anthropic packages
        # installed, so constructing these should fail loudly with a
        # clear LLMError rather than an opaque ImportError deep in some
        # other module.
        from codeagent.llm import OpenAIClient

        try:
            import openai  # noqa: F401

            self.skipTest("openai package is installed in this environment")
        except ImportError:
            pass

        with self.assertRaises(LLMError):
            OpenAIClient()

    def test_anthropic_client_raises_llm_error_if_package_missing(self):
        from codeagent.llm import AnthropicClient

        try:
            import anthropic  # noqa: F401

            self.skipTest("anthropic package is installed in this environment")
        except ImportError:
            pass

        with self.assertRaises(LLMError):
            AnthropicClient()


if __name__ == "__main__":
    unittest.main()
