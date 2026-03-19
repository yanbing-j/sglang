"""
For intel_amx attention backend FP8 tests
Usage:
python3 -m unittest test_intel_amx_attention_backend_1.TestIntelAMXAttnBackendQuant.test_latency_fp8_qwen
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    intel_amx_benchmark,
    is_in_ci,
    popen_launch_server,
)


class TestIntelAMXAttnBackendQuant(CustomTestCase):

    @intel_amx_benchmark(
        extra_args=["--batch-size", "4", "--mem-fraction-static", "0.1"],
        min_throughput=150,
    )
    def test_latency_fp8_qwen(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8

    @intel_amx_benchmark(
        extra_args=["--batch-size", "4", "--mem-fraction-static", "0.1"],
        min_throughput=50,
    )
    def test_latency_fp8_moe_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE

    def test_mmlu(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.1",
                "--disable-radix",
                "--trust-remote-code",
                "--disable-overlap-schedule",
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )
            metrics = run_eval(args)
            if is_in_ci():
                self.assertGreater(metrics["score"], 0.45)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
