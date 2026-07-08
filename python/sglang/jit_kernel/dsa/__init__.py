from sglang.srt.utils import is_hip

from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)

if not is_hip():
    # Preserve the original eager import behavior on non-ROCm platforms.
    # CPU-only environments (pyproject_cpu.toml) do not ship nvidia-cutlass-dsl;
    # fall back to None there, matching the ROCm behavior. Callers assert at use.
    try:
        from .cutedsl_paged_mqa_logits import (
            CuteDSLPagedMQALogitsRunner,
            pick_dsl_expand,
        )
    except ImportError:
        CuteDSLPagedMQALogitsRunner = None
        pick_dsl_expand = None

__all__ = [
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "cutedsl_paged_mqa_logits",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]
