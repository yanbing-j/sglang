# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Callable
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.modelopt_quant import (
    enable_flashinfer_fp4_gemm,
    fp4_gemm,
    fp4_quantize,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.srt.utils import cpu_has_amx_support, is_cpu

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A4Fp4"]

_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()

_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _convert_swizzled_to_linear(
    scale: torch.Tensor, rows: int, cols: int, block_size: int = 16
) -> torch.Tensor:
    row_tiles = (rows + 127) // 128
    col_tiles = (cols + block_size * 4 - 1) // (block_size * 4)
    scale = scale.reshape(1, row_tiles, col_tiles, 32, 4, 4)
    scale = scale.permute(0, 1, 4, 3, 2, 5)
    return scale.reshape(row_tiles * 128, col_tiles * 4)[:rows, : cols // block_size]


def _swizzle_blockscale_cpu(scale: torch.Tensor) -> torch.Tensor:
    assert scale.dtype == torch.float8_e4m3fn
    scale_ndim = scale.ndim
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
    assert scale.ndim == 3
    batches, rows, cols = scale.shape
    rows_padded = (rows + 127) // 128 * 128
    cols_padded = (cols + 3) // 4 * 4
    padded_scale = torch.zeros(
        (batches, rows_padded, cols_padded), dtype=scale.dtype, device=scale.device
    )
    padded_scale[:batches, :rows, :cols] = scale
    swizzled_scale = padded_scale.reshape(
        batches, rows_padded // 128, 4, 32, cols_padded // 4, 4
    ).permute(0, 1, 4, 3, 2, 5)
    return (
        swizzled_scale.reshape(rows_padded, cols_padded)
        if scale_ndim == 2
        else swizzled_scale.reshape(batches, rows_padded, cols_padded)
    )


def _dequantize_nvfp4_linear(
    value: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    scale_is_swizzled: bool,
) -> torch.Tensor:
    rows, half_cols = value.shape
    cols = half_cols * 2
    low = (value & 0xF).to(torch.int64)
    high = (value >> 4).to(torch.int64)
    lut = _FP4_E2M1_LUT.to(device=value.device)
    dequant = torch.empty((rows, cols), dtype=torch.float32, device=value.device)
    dequant[:, 0::2] = lut[low]
    dequant[:, 1::2] = lut[high]

    scale = scale.view(torch.float8_e4m3fn)
    if scale_is_swizzled:
        scale = _convert_swizzled_to_linear(scale, rows, cols)
    scale = scale.to(torch.float32) / global_scale.to(torch.float32)
    return (dequant * scale.repeat_interleave(16, dim=-1)).to(out_dtype)


class CompressedTensorsW4A4Fp4(CompressedTensorsLinearScheme):
    def __init__(self):
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        global_input_scale = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)

        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        if _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "CompressedTensorsW4A4Fp4 on CPU requires AMX support"
            swizzled_weight_scale = _swizzle_blockscale_cpu(layer.weight_scale)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(
                layer.weight_packed.data, requires_grad=False
            )
            layer.alpha = Parameter(
                1 / (layer.input_global_scale * layer.weight_global_scale),
                requires_grad=False,
            )
            return

        if get_fp4_gemm_runner_backend().is_flashinfer_trtllm():
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight_packed.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(weight, requires_grad=False)
        else:
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(
                layer.weight_packed.data, requires_grad=False
            )

        layer.alpha = Parameter(
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        w_n = layer.output_size_per_partition
        output_shape = [x.shape[0], w_n]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = fp4_quantize(x, layer.input_global_scale)

        assert x_fp4.dtype == torch.uint8
        assert layer.weight_packed.dtype == torch.uint8
        assert layer.weight_scale.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        w = layer.weight_packed
        w_blockscale = layer.weight_scale
        if x.device.type != "cpu" and enable_flashinfer_fp4_gemm:
            w = layer.weight_packed.T
            w_blockscale = layer.weight_scale.T

        out = fp4_gemm(
            x_fp4,
            w,
            x_blockscale,
            w_blockscale,
            layer.alpha,
            output_dtype,
            w_n,
        )
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
