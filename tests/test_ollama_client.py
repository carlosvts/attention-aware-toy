"""Tests for the Ollama HTTP client lifecycle."""

import unittest
from unittest.mock import patch

from src.llm.ollama_client import OllamaClient, OllamaGPUError


class OllamaClientLifecycleTests(unittest.TestCase):
    @patch("src.llm.ollama_client.nvidia_gpu_available", return_value=False)
    @patch("src.llm.ollama_client.record_ollama_lifecycle")
    @patch("src.llm.ollama_client.record_model_metrics")
    @patch("src.llm.ollama_client.record_elapsed")
    @patch("src.llm.ollama_client.requests.get")
    @patch("src.llm.ollama_client.requests.post")
    def test_chat_streams_and_records_first_token_and_total(
        self,
        post,
        get,
        record_elapsed,
        record_model_metrics,
        record_lifecycle,
        gpu_available,
    ) -> None:
        response = post.return_value
        response.iter_lines.return_value = [
            '{"message":{"content":"Olá"},"done":false}',
            '{"message":{"content":"!"},"done":true,'
            '"total_duration":3000000000,"load_duration":1000000000,'
            '"prompt_eval_duration":1200000000,"eval_duration":800000000}',
        ]
        get.return_value.json.side_effect = [
            {"models": []},
            {"models": [{"name": "qwen-test", "size_vram": 1024}]},
        ]
        client = OllamaClient("http://localhost:11434", "qwen-test", 30.0)

        result = client.chat("system", "user", profiling_name="qwen-test")

        self.assertEqual(result, "Olá!")
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertTrue(post.call_args.kwargs["json"]["stream"])
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_gpu"], -1)
        self.assertEqual(
            [item.args[0] for item in record_elapsed.call_args_list],
            ["qwen-test_first_token", "qwen-test_total"],
        )
        record_model_metrics.assert_called_once()
        self.assertEqual(
            [item.args[0] for item in record_lifecycle.call_args_list],
            ["[OLLAMA] Loading model...", "[OLLAMA] Model loaded"],
        )
        metrics = record_model_metrics.call_args.args[2]
        self.assertEqual(metrics["ollama_load_duration_seconds"], 1.0)
        self.assertEqual(metrics["prompt_eval_duration_seconds"], 1.2)
        self.assertEqual(metrics["eval_duration_seconds"], 0.8)
        self.assertEqual(metrics["ollama_total_duration_seconds"], 3.0)
        response.close.assert_called_once_with()

    @patch("src.llm.ollama_client.nvidia_gpu_available", return_value=False)
    @patch("src.llm.ollama_client.record_ollama_lifecycle")
    @patch("src.llm.ollama_client.record_model_metrics")
    @patch("src.llm.ollama_client.requests.get")
    @patch("src.llm.ollama_client.requests.post")
    def test_reports_reuse_when_model_is_already_resident(
        self, post, get, record_model_metrics, record_lifecycle, gpu_available
    ) -> None:
        post.return_value.iter_lines.return_value = [
            '{"message":{"content":"Olá"},"done":true}'
        ]
        get.return_value.json.return_value = {
            "models": [{"name": "qwen-test", "size_vram": 1024}]
        }
        client = OllamaClient("http://localhost:11434", "qwen-test", 30.0)

        client.chat("system", "user", profiling_name="qwen_llm")

        self.assertEqual(record_lifecycle.call_count, 1)
        self.assertEqual(
            record_lifecycle.call_args.args[0],
            "[OLLAMA] Reusing loaded model",
        )

    @patch("src.llm.ollama_client.nvidia_gpu_available", return_value=True)
    @patch("src.llm.ollama_client.record_ollama_lifecycle")
    @patch("src.llm.ollama_client.record_model_metrics")
    @patch("src.llm.ollama_client.requests.get")
    @patch("src.llm.ollama_client.requests.post")
    def test_rejects_cpu_fallback_when_nvidia_gpu_is_available(
        self, post, get, record_model_metrics, record_lifecycle, gpu_available
    ) -> None:
        post.return_value.iter_lines.return_value = [
            '{"message":{"content":"Olá"},"done":true}'
        ]
        get.return_value.json.return_value = {
            "models": [{"name": "qwen-test", "size_vram": 0}]
        }
        client = OllamaClient("http://localhost:11434", "qwen-test", 30.0)

        with self.assertRaises(OllamaGPUError):
            client.chat("system", "user", profiling_name="qwen_llm")

    @patch("src.llm.ollama_client.record_ollama_lifecycle")
    @patch("src.llm.ollama_client.requests.get")
    @patch("src.llm.ollama_client.requests.post")
    def test_unload_requests_immediate_model_eviction(
        self, post, get, record_lifecycle
    ) -> None:
        client = OllamaClient("http://localhost:11434", "qwen-test", 30.0)

        client.unload()

        post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "qwen-test", "keep_alive": 0},
            timeout=(2.0, 10.0),
        )
        post.return_value.raise_for_status.assert_called_once_with()
        self.assertEqual(
            record_lifecycle.call_args.args[0], "[OLLAMA] Unloading model"
        )


if __name__ == "__main__":
    unittest.main()
