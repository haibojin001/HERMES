"""Check hosted model request shapes without credentials or network access."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes import pipeline as p


class TestHostedModelInterfaces(unittest.TestCase):
    def setUp(self):
        self.saved = (p.BACKEND, p.THINKING_MODEL, p.TRIAGE_MODEL,
                      p.HOSTED_REASONING_EFFORT, p.STRIP_THINK_TAGS,
                      p.FILE_CONTEXT_CHARS, p.MODEL_NAME)

    def tearDown(self):
        (p.BACKEND, p.THINKING_MODEL, p.TRIAGE_MODEL,
         p.HOSTED_REASONING_EFFORT, p.STRIP_THINK_TAGS,
         p.FILE_CONTEXT_CHARS, p.MODEL_NAME) = self.saved

    def test_openai_uses_responses_reasoning_without_temperature(self):
        create = Mock(return_value=types.SimpleNamespace(
            output_text='{"ready": true}', status="completed",
            incomplete_details=None,
            usage=types.SimpleNamespace(input_tokens=7, output_tokens=4)))
        client = Mock()
        client.responses.create = create
        fake = types.SimpleNamespace(OpenAI=Mock(return_value=client))
        with patch.dict(sys.modules, {"openai": fake}):
            p.use_hosted("openai/gpt-test", reasoning_effort="medium")
            self.assertEqual(
                p.llm_json('Return {"ready": true}', "openai/gpt-test"),
                {"ready": True})
        sent = create.call_args.kwargs
        self.assertEqual(sent["model"], "gpt-test")
        self.assertEqual(sent["reasoning"], {"effort": "medium"})
        self.assertNotIn("temperature", sent)
        self.assertFalse(sent["store"])

    def test_claude_uses_messages_output_config_without_temperature(self):
        create = Mock(return_value=types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text='{"ready": true}')],
            stop_reason="end_turn",
            usage=types.SimpleNamespace(input_tokens=7, output_tokens=4)))
        client = Mock()
        client.messages.create = create
        fake = types.SimpleNamespace(Anthropic=Mock(return_value=client))
        with patch.dict(sys.modules, {"anthropic": fake}):
            p.use_hosted("anthropic/claude-test", reasoning_effort="high")
            self.assertEqual(
                p.llm_json('Return {"ready": true}',
                           "anthropic/claude-test"),
                {"ready": True})
        sent = create.call_args.kwargs
        self.assertEqual(sent["model"], "claude-test")
        self.assertEqual(sent["output_config"], {"effort": "high"})
        self.assertNotIn("temperature", sent)

    def test_non_object_json_retries_as_invalid_response(self):
        with patch.object(p, "llm", return_value='["ready"]') as call:
            answer = p.llm_json("Return an object", "openai/gpt-test")
        self.assertEqual(answer["_error"], "parse failed")
        self.assertEqual(call.call_count, 2)

    def test_other_hosted_provider_receives_effort_explicitly(self):
        completion = Mock(return_value=types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content='{"ready": true}'))],
            usage=None))
        with patch.dict(sys.modules, {
                "litellm": types.SimpleNamespace(completion=completion)}):
            p.use_hosted("bedrock/example", reasoning_effort="medium")
            self.assertEqual(
                p.llm_json('Return {"ready": true}', "bedrock/example"),
                {"ready": True})
        self.assertEqual(
            completion.call_args.kwargs["reasoning_effort"], "medium")
        self.assertNotIn("temperature", completion.call_args.kwargs)

    def test_local_primitive_routes_with_hosted_planner(self):
        with patch.object(
                p, "_ollama_chat",
                return_value=('<think>reasoning</think>{"ready":true}',
                              p._Usage(5, 3), "reasoning")) as local:
            p.use_hosted("openai/gpt-test", reasoning_effort="medium")
            self.assertEqual(
                p.llm_json("Return JSON", "ollama/qwen3:8b"),
                {"ready": True})
        self.assertEqual(local.call_args.args[0], "ollama/qwen3:8b")

    def test_vllm_role_prefix_uses_served_model_id(self):
        with patch.object(
                p, "_vllm_chat",
                return_value=('<think>reasoning</think>{"ready":true}',
                              p._Usage(5, 3))) as local:
            p.use_hosted("openai/gpt-test")
            self.assertEqual(
                p.llm_json("Return JSON", "vllm/Qwen/Qwen3-8B"),
                {"ready": True})
        self.assertEqual(local.call_args.args[0], "Qwen/Qwen3-8B")


if __name__ == "__main__":
    unittest.main()
