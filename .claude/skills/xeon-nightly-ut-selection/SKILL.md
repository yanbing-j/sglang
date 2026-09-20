---
name: xeon-nightly-ut-selection
description: "Select SGLang tests that should run in Intel Xeon CPU nightly CI. Use when updating nightly-intel-cpu-gnr, auditing the current checkout or branch for Xeon nightly UT coverage, checking newly added UTs, or reviewing register_cpu_ci changes. Defaults to a full inventory of test/registered and distinguishes CPU-runnable tests from tests with Xeon/CPU-backend-specific value."
argument-hint: "[current branch | full inventory | optional base ref]"
---

# Xeon Nightly UT Selection

Use this skill to find tests that should be registered for the Intel Xeon nightly
suite, currently `nightly-intel-cpu-gnr`. Default to a full inventory of the
current checkout under `test/registered`, then identify which tests have Xeon or
CPU-backend-specific value and should get Xeon nightly registration. The goal is
cost-effective Xeon coverage, not a broad sweep of every test that can run on CPU.

## Selection Policy

Put a test in Xeon nightly when it has Xeon or CPU-backend-specific value:

- CPU backend inference, serving, or config paths such as `--device cpu`.
- Intel/Xeon/AMX behavior, CPU graph, thread binding, TP/comm, NUMA, or CPU process behavior.
- `sgl_kernel` CPU ops, CPU attention, CPU MLA/Mamba/MoE, CPU sampling, CPU KV/cache/store kernels.
- CPU runtime observability such as CPU monitor, crash dump, watchdog, socket/network utilities.
- CPU serving smoke tests that exercise cache/scheduler/API behavior on the CPU backend.

Do not add a test just because it is CPU-runnable:

- Pure Python parser/config/tokenizer/function-call/mock tests usually belong in hosted `base-a-test-cpu`, not Xeon nightly.
- Tests that use `device="cpu"` only as a reference for CUDA/AMD/NPU/XPU kernels do not need Xeon.
- Vendor-specific paths under `amd/`, `xpu/`, `npu/`, `mlx/`, `musa/`, CUDA graph, GPU kernel, benchmark, or perf directories should not be registered for Xeon unless a CPU-only part is split out and justified.
- If a file mixes CPU-only tests with GPU-only tests, do not register the whole file. Split the CPU-only class/function into a separate file, then register that file.

## Procedure

1. Confirm the target suite name. For the current Intel Xeon GNR nightly, use
   `nightly-intel-cpu-gnr`.
2. By default, run the **Inventory Script**. This is the preferred workflow when
    the caller only says "current branch" or has simply checked out/pulled a
    branch to the latest revision.
3. Run the **Current Branch Delta Script** only when the caller explicitly asks
    to inspect newly added/modified UT files. It does not require the caller to
    know the delta: it compares the checked-out branch against the best available
    main/default-branch ref.
4. Review `LIKELY_KEEP` / `KEEP` first. These are high-confidence files that
   should be in Xeon nightly.
5. Review `MAYBE` separately. Add only after reading/running the file or splitting
   CPU-only tests out of mixed files.
6. The default inventory includes a reviewed triage layer for CPU-looking tests:
    reviewed CPU-kernel/NUMA/host-cache/spec-CPU files are promoted into `KEEP`,
    cache/scheduler/attention-adjacent files are promoted into `MAYBE`, and
    reviewed parser/vendor/GPU/model/API/pure-logic files are reported as `NO`.
7. Files outside `LIKELY_KEEP` / `KEEP` and `MAYBE` should normally stay out of
    Xeon nightly unless there is a specific Xeon/CPU-backend reason.
8. The scripts write the full Markdown table to a report file and print only a
    short summary to stdout. Read the report file when you need the complete
    table; stdout may be truncated by the terminal or chat tool.

## Current Branch Delta Script

Use this only when the caller explicitly asks for current-branch delta or
newly-added/modified UT files. It is read-only. By default it auto-detects a base
ref from `origin/HEAD`, `origin/main`, `sgl-public/main`, `upstream/main`, or
local `main`. Override `BASE_REF` only if the branch should compare against a
different base. The script also includes untracked `test/registered/**/*.py`
files, so local newly-created tests are not missed.

```bash
python - <<'PY'
from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

SUITE = "nightly-intel-cpu-gnr"

EXCLUDE_TOP_LEVEL = {
    "amd",
    "xpu",
    "npu",
    "mlx",
    "musa",
    "gb300",
    "cuda_graph",
    "kernels",
    "kernel",
    "perf",
}

LIKELY_PATH_RE = re.compile(
    r"(^test/registered/cpu/)|"
    r"(cpu_monitor|cpu_graph|server_args_backend|rank_consensus|request_headers|request_decompression)|"
    r"(intel_amx|amx|numa|binding|comm)|"
    r"(test_(decode|extend|flash_attn|mla|mamba|moe|shared_expert|topk|gemm|bmm|activation|norm|rope|qkv|sampling|store_cache|spec_kernels))",
    re.IGNORECASE,
)
LIKELY_CONTENT_RE = re.compile(
    r"--device\s+cpu|device=[\"']cpu|cpu backend|backend.*cpu|"
    r"torch\.ops\.sgl_kernel|sgl_kernel|AMX|amx|numa|cpu_graph",
    re.IGNORECASE,
)
MAYBE_CONTENT_RE = re.compile(
    r"host|pool_host|radix|swa|hicache|kv_cache|offload|fallback|cpu",
    re.IGNORECASE,
)
GPU_ONLY_RE = re.compile(
    r"cuda|rocm|hip|xpu|npu|mlx|musa|flashattention|flashinfer|triton|cutlass|fa3|deepgemm",
    re.IGNORECASE,
)


def git_output(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def choose_base_ref() -> str:
    override = os.environ.get("BASE_REF")
    if override:
        return override

    candidates: list[str] = []
    remote_head = git_output(["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"])
    if remote_head:
        candidates.append(remote_head)
    candidates.extend(["origin/main", "sgl-public/main", "upstream/main", "main"])
    for candidate in candidates:
        if git_output(["rev-parse", "--verify", candidate]) is not None:
            return candidate
    raise SystemExit(
        "Could not find a base ref. Fetch origin/main or run with BASE_REF=<ref>."
    )


def changed_test_files(base_ref: str) -> list[str]:
    merge_base = subprocess.check_output(
        ["git", "merge-base", base_ref, "HEAD"], text=True
    ).strip()
    lines = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-status",
            "--diff-filter=ACMR",
            merge_base,
            "HEAD",
            "--",
            "test/registered",
        ],
        text=True,
    ).splitlines()
    files: list[str] = []
    for line in lines:
        parts = line.split("\t")
        path = parts[-1]
        name = os.path.basename(path)
        if path.endswith(".py") and name not in {"__init__.py", "conftest.py"}:
            files.append(path)

    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "test/registered"],
        text=True,
    ).splitlines()
    for path in untracked:
        name = os.path.basename(path)
        if path.endswith(".py") and name not in {"__init__.py", "conftest.py"}:
            files.append(path)
    return sorted(set(files))


def registry_status(text: str) -> str:
    if SUITE in text:
        return "registered-nightly"
    if "register_cpu_ci" in text:
        return "cpu-registered-other-suite"
    return "not-cpu-registered"


def symbols(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.append(node.name)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            names.append(node.name)
    return ", ".join(names[:4])


def classify(path: str) -> tuple[str, str, str, str]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    status = registry_status(text)
    top = path.split("/")[2] if len(path.split("/")) > 2 else ""
    haystack = f"{path}\n{text}"
    tests = symbols(text)

    if top in EXCLUDE_TOP_LEVEL and not path.startswith("test/registered/cpu/"):
        return "NO", status, "vendor/GPU/perf/kernel path; do not register whole file for Xeon", tests
    if path.startswith("test/registered/cpu/") and "/arm64/" not in path:
        return "LIKELY_KEEP", status, "CPU backend test directory", tests
    if LIKELY_PATH_RE.search(path) or LIKELY_CONTENT_RE.search(haystack):
        if GPU_ONLY_RE.search(path) and not re.search(r"cpu|host|fallback|backend", haystack, re.IGNORECASE):
            return "MAYBE", status, "GPU-looking file with CPU hints; read before registering", tests
        return "LIKELY_KEEP", status, "Xeon/CPU-backend-specific signal", tests
    if MAYBE_CONTENT_RE.search(haystack):
        return "MAYBE", status, "CPU/cache/host-looking signal; confirm or split", tests
    return "NO", status, "CPU-runnable or pure logic is not enough for Xeon nightly", tests


rows = []
base_ref = choose_base_ref()
paths = changed_test_files(base_ref)

for path in paths:
    if not Path(path).exists():
        continue
    bucket, status, reason, tests = classify(path)
    rows.append((bucket, path, status, reason, tests))

report_path = Path(os.environ.get("REPORT_PATH", "xeon_nightly_ut_delta.md"))
report_lines = [
    "# Xeon Nightly UT Delta Report",
    "",
    f"Base ref: `{base_ref}`",
    f"Changed or untracked registered test files: {len(paths)}",
]

for bucket in ("LIKELY_KEEP", "MAYBE", "NO"):
    bucket_rows = [row for row in rows if row[0] == bucket]
    report_lines.extend(
        [
            "",
            f"## {bucket} ({len(bucket_rows)})",
            "| File | Status | Reason | Tests |",
            "|---|---|---|---|",
        ]
    )
    for _, path, status, reason, tests in bucket_rows:
        report_lines.append(f"| {path} | {status} | {reason} | {tests} |")

report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print(f"Base ref: {base_ref}")
print(f"Changed or untracked registered test files: {len(paths)}")
for bucket in ("LIKELY_KEEP", "MAYBE", "NO"):
    print(f"{bucket}: {sum(1 for row in rows if row[0] == bucket)}")
print(f"Full report: {report_path}")
PY
```

Interpretation:

- `LIKELY_KEEP`: add or keep `register_cpu_ci(..., suite="nightly-intel-cpu-gnr", nightly=True)` unless there is a concrete blocker.
- `MAYBE`: read or run the file. If it is mixed CPU/GPU, split the CPU-only test first.
- `NO`: do not register for Xeon nightly by default.

## Inventory Script

Run this from the repo root by default. Use it whenever the caller says only
"current branch", "current checkout", "latest branch", or asks for the Xeon
nightly UT selection without explicitly limiting the scope to changed files:

```bash
python - <<'PY'
from __future__ import annotations

import ast
import glob
import os
import re
from pathlib import Path

SUITE = "nightly-intel-cpu-gnr"
ROOT = Path("test/registered")

# Non-cpu/ files with known Xeon-nightly value. Keep this list intentionally
# small; most pure Python unit tests should stay in hosted CPU CI.
KEEP_EXACT = {
    "test/registered/e2e/models/test_transformers_backend_eval.py": "CPU transformers backend e2e smoke",
    "test/registered/prefill_only/test_openai_embedding.py": "CPU embedding serving API",
    "test/registered/openai_server/function_call/test_anthropic_tool_use.py": "CPU serving tool-use API",
    "test/registered/openai_server/validation/test_large_max_new_tokens.py": "CPU serving request validation",
    "test/registered/openai_server/validation/test_matched_stop.py": "CPU serving stop matching",
    "test/registered/radix_cache/test_radix_attention.py": "CPU radix/cache serving behavior",
    "test/registered/radix_cache/test_radix_cache_hit.py": "CPU radix cache hit behavior",
    "test/registered/radix_cache/test_swa_radix_cache_kl.py": "CPU SWA/radix behavior",
    "test/registered/scheduler/test_abort_with_metrics.py": "CPU scheduler/runtime metrics",
    "test/registered/scheduler/test_routing_key_scheduling.py": "CPU serving scheduler routing",
    "test/registered/sessions/test_streaming_session_swa_extra.py": "CPU session/SWA cache behavior",
    "test/registered/unit/managers/test_mm_embed_scatter.py": "CPU host tensor multimodal manager path",
    "test/registered/unit/managers/test_mm_embedding_length.py": "CPU host tensor multimodal manager path",
    "test/registered/unit/observability/test_cpu_monitor.py": "CPU-specific observability",
    "test/registered/unit/cpu/test_diffusion_norm.py": "sgl_kernel CPU diffusion norm ops",
    "test/registered/unit/mem_cache/test_mem_pool_host.py": "host KV/cache pool allocation and CPU host-pool bookkeeping",
    "test/registered/unit/spec/test_spec_cpu_overlap_constraint.py": "explicit CPU speculative overlap scheduling constraint",
    "test/registered/unit/utils/test_diffusion_torch_fallback.py": "CPU torch fallback",
    "test/registered/debug_utils/test_crash_dump.py": "CPU runtime crash dump",
    "test/registered/debug_utils/test_soft_watchdog.py": "CPU process watchdog",
    "test/registered/utils/test_log_utils.py": "CPU/runtime utility",
    "test/registered/utils/test_network_address.py": "CPU serving network utility",
    "test/registered/utils/test_numa_utils.py": "NUMA binding/detection utility",
    "test/registered/utils/test_socket_utils.py": "CPU socket/runtime utility",
}

# Review these manually. Some are mixed CPU/GPU files and should be split before
# registration; others may be valid CPU backend capability tests.
MAYBE_EXACT = {
    "test/registered/spec/dspark/test_ragged_verify_backend_capability.py": "confirm this is CPU capability-only, not speculative GPU backend coverage",
    "test/registered/unit/mem_cache/test_dsa_pool_host_unit.py": "mixed: split CPU-only TestDSAOffloadSignatures from CUDA/ROCm transfer tests",
    "test/registered/unit/mem_cache/test_minimax_sparse_pool_host_unit.py": "mixed: split CPU host integration from CUDA/ROCm transfer tests",
    "test/registered/unit/mem_cache/test_radix_cache_unit.py": "likely CPU-side radix/cache logic; run once on CPU before adding",
    "test/registered/unit/mem_cache/test_swa_eviction_boundary.py": "confirm no CUDA-only dependency",
    "test/registered/unit/mem_cache/test_swa_lock_release_lifecycle.py": "confirm pure lifecycle/state-machine behavior",
    "test/registered/unit/mem_cache/test_swa_unittest.py": "mixed: split CPU-safe cache classes from CUDA sync/peer-mapped tests",
    "test/registered/unit/mem_cache/test_unified_mla_block_table.py": "mixed: split CPU TestBlockTable from FA3/CUDA metadata tests",
    "test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py": "likely CPU-side unified radix/cache logic; run once on CPU before adding",
    "test/registered/unit/spec/test_dflash_extra_buffer_lazy.py": "dflash name suggests GPU attention; confirm before adding",
    "test/registered/unit/spec/test_resolve_swa_kv_pool.py": "likely CPU/cache relevant; confirm no GPU-only path",
    "test/registered/attention/unittests/swa/test_swa_out_cache_loc.py": "cache/host-looking unit test; add only if CPU-safe and CPU-backend-specific",
    "test/registered/kv_canary/test_e2e_base.py": "cache/host-looking unit test; add only if CPU-safe and CPU-backend-specific",
    "test/registered/kv_canary/test_self_unit_e2e_base.py": "cache/host-looking unit test; add only if CPU-safe and CPU-backend-specific",
    "test/registered/unit/disaggregation/test_minimax_sparse_disagg_state_kv_args.py": "cache/host-looking unit test; add only if CPU-safe and CPU-backend-specific",
    "test/registered/unit/disaggregation/test_specv2_kvcache_offloading.py": "cache/host-looking unit test; add only if CPU-safe and CPU-backend-specific",
    "test/registered/unit/layers/attention/test_gdn_prefill_backend_policy.py": "linear-attention backend policy; confirm CPU backend branch exists",
    "test/registered/unit/layers/attention/test_linear_attn_config.py": "linear-attention config; confirm CPU backend value over pure config",
    "test/registered/unit/layers/test_mamba2_track_ssm_indices.py": "Mamba state indices; confirm CPU-side path",
    "test/registered/unit/layers/test_minicpm_sparse_cache.py": "sparse cache behavior; confirm CPU-safe subset",
    "test/registered/unit/layers/test_minicpm_sparse_metadata.py": "sparse cache metadata; likely pure logic, review before adding",
    "test/registered/unit/managers/test_schedule_batch_convert_decode_to_extend.py": "decode/extend conversion path; only add if CPU backend-specific behavior exists",
    "test/registered/unit/managers/test_schedule_batch_prepare_for_decode.py": "decode preparation path; only add if CPU backend-specific behavior exists",
    "test/registered/unit/managers/test_scheduler_flush_cache.py": "scheduler cache flush behavior; CPU serving cache relevance",
    "test/registered/unit/managers/test_scheduler_hicache_events.py": "scheduler HiCache event handling; CPU serving cache relevance",
    "test/registered/unit/managers/test_scheduler_pause_generation.py": "pause generation scheduler state; may matter for CPU serving, confirm",
    "test/registered/unit/mem_cache/test_dsv4_compressed_pools.py": "compressed KV/pool layouts; confirm CPU-side pool logic",
    "test/registered/unit/mem_cache/test_dsv4_hicache_l2.py": "HiCache L2/deepseek host-cache behavior; confirm CPU-safe subset",
    "test/registered/unit/mem_cache/test_dsv4_unified_fp8_pool.py": "unified FP8 pool layout; confirm no GPU-only assumption",
    "test/registered/unit/mem_cache/test_hicache_auto_size.py": "HiCache CPU/host sizing policy; confirm it is not only config math",
    "test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py": "HiCache DCP host pool behavior; confirm no GPU transfer dependency",
    "test/registered/unit/mem_cache/test_hicache_host_register.py": "HiCache host registration path; confirm no platform-specific GPU dependency",
    "test/registered/unit/mem_cache/test_hicache_staged_write_back_dispatch.py": "HiCache staged host write-back dispatch; confirm no GPU transfer dependency",
    "test/registered/unit/mem_cache/test_kv_index_translator.py": "KV index translator; likely cache routing value, but pure logic",
    "test/registered/unit/mem_cache/test_minimax_sparse_pool_pd_unit.py": "sparse pool PD path; confirm no GPU transfer dependency",
    "test/registered/unit/mem_cache/test_paged_allocator_lazy_release.py": "paged allocator release lifecycle; likely CPU-side cache allocator",
    "test/registered/unit/mem_cache/test_paged_free_segment.py": "paged free segment lifecycle; likely CPU-side cache allocator",
    "test/registered/unit/mem_cache/test_prefill_memory_budget.py": "prefill memory budget/admission; scheduler-cache relevant but may be pure policy",
    "test/registered/unit/mem_cache/test_quantized_kv_pool.py": "quantized KV pool; confirm CPU-safe construction",
    "test/registered/unit/mem_cache/test_session_unified_radix_cache.py": "session unified radix cache behavior; cache-relevant",
    "test/registered/unit/mem_cache/test_streaming_session_unit.py": "streaming session cache state; cache/session relevant",
    "test/registered/unit/mem_cache/test_umbp_host_allocator.py": "host allocator behavior; likely CPU memory-pool value",
    "test/registered/unit/mem_cache/test_unified_free_no_host_sync.py": "host-sync avoidance/free path; confirm CPU-safe subset",
    "test/registered/unit/mem_cache/test_unified_mamba_views.py": "Mamba cache views; confirm CPU-side view logic not GPU backend parity",
    "test/registered/unit/mem_cache/test_unified_mha_views.py": "MHA cache views; confirm CPU-side view logic not GPU backend parity",
    "test/registered/unit/model_executor/runner/test_hidden_state_graph_recapture.py": "graph recapture behavior; likely GPU/CUDA graph unless CPU graph branch exists",
    "test/registered/unit/model_executor/runner_utils/test_graph_pool_borrow.py": "graph pool utility; likely not Xeon unless CPU graph path is explicit",
    "test/registered/unit/platforms/test_platform_interface.py": "platform CPU enum/device capability; maybe CPU platform contract",
    "test/registered/unit/utils/test_tensor_bridge.py": "explicit TensorBridgeCpu but mixed with Metal sharing; split CPU test if adding",
}

EXCLUDE_TOP_LEVEL = {
    "amd",
    "xpu",
    "npu",
    "mlx",
    "musa",
    "gb300",
    "cuda_graph",
    "kernels",
    "kernel",
    "perf",
}

STRONG_REVIEW_HINTS = re.compile(
    r"xeon|intel|amx|numa|ipex|oneapi|onednn|mkldnn|"
    r"cpu_graph|cpu backend|backend.*cpu|--device\s+cpu|"
    r"device=[\"']cpu|torch\.ops\.sgl_kernel|sgl_kernel",
    re.IGNORECASE,
)
REVIEWED_MAYBE_HINTS = re.compile(
    r"unit/mem_cache/|unit/managers/test_scheduler_|unit/layers/attention/|"
    r"unit/layers/test_mamba|cache|kv|host",
    re.IGNORECASE,
)
REVIEWED_NO_HINTS = re.compile(
    r"xpu|npu|mlx|musa|cuda|gpu|triton|flashinfer|fa3|deepgemm|rocm|amd|"
    r"marlin|vllm|lora|parser|function_call|grammar|tokenizer|conversation|"
    r"jinja|sampling_params|penalty|json_response|http_server_auth|"
    r"patch_tokenizer|benchmark|gsm8k|external_models|quant_config|fp8_utils|"
    r"humming|mxfp4|fp4|checkpoint|config|audio|transcription|anthropic|"
    r"openai|vlm|multimodal|rust|vision|qwen|kimi|zaya|longcat|glm|runai|"
    r"model_overrides|label_transform|metrics_utils|func_timer|request_metrics|"
    r"startup_func|priority_metrics|gauge_histogram",
    re.IGNORECASE,
)


def registry_status(path: str) -> tuple[str, str]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    has_suite = SUITE in text
    has_cpu = "register_cpu_ci" in text
    if has_suite:
        status = "registered-nightly"
    elif has_cpu:
        status = "cpu-registered-other-suite"
    else:
        status = "not-cpu-registered"
    return status, text


def test_symbols(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.append(node.name)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            names.append(node.name)
    return ", ".join(names[:4])


def cpu_path_reason(path: str, text: str) -> str:
    name = os.path.basename(path)
    lower = (path + "\n" + text).lower()
    if "amx" in lower:
        return "Intel AMX / Xeon backend"
    if "binding" in name or "comm" in name or "cpu_graph" in name:
        return "Xeon CPU runtime/TP/graph path"
    if any(key in name for key in ("decode", "extend", "flash_attn", "mla", "mamba")):
        return "CPU attention/linear-attention backend"
    if any(key in name for key in ("moe", "shared_expert", "topk")):
        return "CPU MoE/routing backend"
    if any(key in name for key in ("gemm", "bmm", "activation", "norm", "rope", "qkv")):
        return "CPU kernel/op correctness"
    if any(key in name for key in ("sampling", "store_cache", "spec_kernels")):
        return "CPU sampling/cache/spec kernel"
    if any(key in name for key in ("server_args", "request", "rank_consensus")):
        return "CPU serving/runtime path"
    return "CPU backend test directory"


def append_table(
    report_lines: list[str], title: str, rows: list[tuple[str, str, str, str]]
) -> None:
    report_lines.append("")
    report_lines.append(f"## {title} ({len(rows)})")
    report_lines.append("| File | Status | Reason | Tests |")
    report_lines.append("|---|---|---|---|")
    for path, status, reason, tests in rows:
        report_lines.append(f"| {path} | {status} | {reason} | {tests} |")


keep: list[tuple[str, str, str, str]] = []
maybe: list[tuple[str, str, str, str]] = []
no: list[tuple[str, str, str, str]] = []
review_new: list[tuple[str, str, str, str]] = []

for path in sorted(glob.glob("test/registered/**/*.py", recursive=True)):
    base = os.path.basename(path)
    if base in {"__init__.py", "conftest.py"} or base.startswith("utils"):
        continue
    status, text = registry_status(path)
    tests = test_symbols(text)

    if path.startswith("test/registered/cpu/") and "/arm64/" not in path:
        keep.append((path, status, cpu_path_reason(path, text), tests))
    elif path in KEEP_EXACT:
        keep.append((path, status, KEEP_EXACT[path], tests))
    elif path in MAYBE_EXACT:
        maybe.append((path, status, MAYBE_EXACT[path], tests))
    else:
        parts = path.split("/")
        top = parts[2] if len(parts) > 2 else ""
        is_review_candidate = top not in EXCLUDE_TOP_LEVEL and STRONG_REVIEW_HINTS.search(
            path + "\n" + text
        )
        if is_review_candidate and REVIEWED_MAYBE_HINTS.search(path):
            maybe.append((path, status, "reviewed maybe: cache/scheduler/attention-looking candidate", tests))
        elif is_review_candidate and REVIEWED_NO_HINTS.search(path):
            no.append((path, status, "reviewed out: vendor/GPU/model/parser/API/pure-logic signal", tests))
        elif is_review_candidate:
            review_new.append((path, status, "new CPU-looking candidate; classify before registering", tests))

report_path = Path(os.environ.get("REPORT_PATH", "xeon_nightly_ut_inventory.md"))
report_lines = [
    "# Xeon Nightly UT Inventory Report",
    "",
    f"Suite: `{SUITE}`",
    f"Scanned root: `{ROOT}`",
]
append_table(report_lines, "KEEP: register or keep in Xeon nightly", keep)
append_table(report_lines, "MAYBE: confirm or split before Xeon nightly", maybe)
append_table(report_lines, "NO: reviewed out of Xeon nightly", no)
append_table(report_lines, "REVIEW: new CPU-looking files not in curated policy", review_new)
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print(f"KEEP: {len(keep)}")
print(f"MAYBE: {len(maybe)}")
print(f"NO: {len(no)}")
print(f"REVIEW: {len(review_new)}")
print(f"Full report: {report_path}")
PY
```

## Broad Registration Diff Shortcut

For any diff that adds `nightly-intel-cpu-gnr` broadly, first count what it changes:

```bash
BASE=<merge-base-or-base-sha> HEAD=<pr-head-sha-or-ref> python - <<'PY'
import collections
import subprocess
import os

base = os.environ["BASE"]
head = os.environ["HEAD"]
files = subprocess.check_output(
    ["git", "diff", "--name-only", base, head, "--", "test/registered"],
    text=True,
).splitlines()
nightly = []
for path in files:
    if not path.endswith(".py"):
        continue
    diff = subprocess.check_output(
        ["git", "diff", base, head, "--", path],
        text=True,
        errors="ignore",
    )
    if "nightly-intel-cpu-gnr" in diff:
        nightly.append(path)

print("nightly registrations:", len(nightly))
print("by top-level directory:")
for key, count in sorted(collections.Counter(p.split("/")[2] for p in nightly).items()):
    print(f"  {key}: {count}")
PY
```

Then compare the changed files with the `KEEP` and `MAYBE` output from the inventory
script. Files outside both sets should normally be removed from the Xeon nightly
registration unless the branch provides a specific CPU-backend/Xeon reason.

## Reporting Template

When reporting results, use three sections:

1. `Keep in Xeon nightly`: files, current registration status, and one-line reason.
2. `Maybe / split first`: files and the decision needed.
3. `Do not register`: summarize counts by directory and cite representative examples,
   especially vendor-specific or pure-logic `unit/` files.

Use this wording for the core distinction:

> The Xeon nightly suite should use Xeon/CPU-backend-specific value as the
> criterion, not CPU-runnable as the criterion.
