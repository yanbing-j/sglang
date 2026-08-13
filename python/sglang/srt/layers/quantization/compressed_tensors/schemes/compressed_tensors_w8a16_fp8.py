# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, List, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from sglang.srt.layers.quantization.utils import convert_to_channelwise
from sglang.srt.utils import cpu_has_amx_support, is_cpu

__all__ = ["CompressedTensorsW8A16Fp8"]

SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR]
CPU_FP8_W8A16_BLOCK_SIZE = [64, 128]

_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()


def _requantize_weight_to_cpu_block_fp8(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size_n, block_size_k = CPU_FP8_W8A16_BLOCK_SIZE
    n, k = weight.shape
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    eps = torch.finfo(torch.float32).eps

    scale = weight_scale.reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(n)
    elif scale.numel() != n:
        assert n % scale.numel() == 0
        scale = scale.repeat_interleave(n // scale.numel())

    dequant_weight = weight.float() * scale.reshape(n, 1)
    block_scales = torch.empty(
        (
            (n + block_size_n - 1) // block_size_n,
            (k + block_size_k - 1) // block_size_k,
        ),
        dtype=torch.float32,
        device=weight.device,
    )
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)

    for n_start in range(0, n, block_size_n):
        n_end = min(n_start + block_size_n, n)
        n_idx = n_start // block_size_n
        for k_start in range(0, k, block_size_k):
            k_end = min(k_start + block_size_k, k)
            k_idx = k_start // block_size_k
            block = dequant_weight[n_start:n_end, k_start:k_end]
            block_scale = (block.abs().max() / fp8_max).clamp(min=eps)
            block_scales[n_idx, k_idx] = block_scale
            qweight[n_start:n_end, k_start:k_end] = (
                (block / block_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            )

    return qweight, block_scales


class CompressedTensorsW8A16Fp8(CompressedTensorsLinearScheme):
    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    # W8A8-Fp8 kernels support only per-tensor and per-channel cases.
    # So if we have a fused module (QKV, MLP) with per tensor scales,
    # we expand each scale to its shard's channels.
    def process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            ws_channelwise = convert_to_channelwise(
                layer.weight_scale, layer.logical_widths
            )
            layer.weight_scale = torch.nn.Parameter(ws_channelwise, requires_grad=False)
        else:
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )

        if _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "CompressedTensorsW8A16Fp8 on CPU requires AMX support"
            weight, weight_scale = _requantize_weight_to_cpu_block_fp8(
                layer.weight, layer.weight_scale
            )
            layer.weight = torch.nn.Parameter(
                torch.ops.sgl_kernel.convert_weight_packed(weight.contiguous()),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
            layer.weight_block_size = CPU_FP8_W8A16_BLOCK_SIZE
            layer.use_intel_amx_backend = True
            return

        # Weights must be transposed for marlin
        layer.weight = torch.nn.Parameter(layer.weight.t(), requires_grad=False)

        if self.is_static_input_scheme:
            # required by torch.compile to be torch.nn.Parameter
            layer.input_scale = torch.nn.Parameter(
                layer.input_scale.data, requires_grad=False
            )
        prepare_fp8_layer_for_marlin(layer, size_k_first=True)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_partition_sizes: List[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        elif self.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        else:
            raise ValueError(
                f"Unsupported weight strategy={self.strategy}, "
                f"supported strategies are {SUPPORTED_STRATEGIES}"
            )

        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (to deal with converted checkpoints)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if _is_cpu:
            input_2d = x.reshape(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                input_2d,
                layer.weight,
                layer.weight_scale,
                layer.weight_block_size,
                bias,
                x.dtype,
                True,
            )
            return output.reshape(*x.shape[:-1], layer.output_size_per_partition)

        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
