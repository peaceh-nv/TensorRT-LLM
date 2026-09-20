# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and dispatch coverage for the SM107 dense NVFP4 SwiGLU kernels."""

import pytest
import torch
from torch._subclasses import fake_tensor
from torch.nn import functional as F

from tensorrt_llm import _utils, math_utils
from tensorrt_llm._torch import cute_dsl_utils, utils
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
from tensorrt_llm._torch.moe.fused_moe import quantization

pytestmark = pytest.mark.skipif(
    not cute_dsl_utils.IS_CUTLASS_DSL_RUBIN_AVAILABLE or _utils.get_sm_version() != 107,
    reason="requires SM107 and CuTe DSL Rubin support",
)


def _dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    m, packed_k = packed.shape
    k = packed_k * 2
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).long().reshape(m, k)
    sf = (
        utils.unswizzle_sf(scales, math_utils.pad_up(m, 128), k)[:m]
        .view(torch.float8_e4m3fn)
        .float()
    )
    return lut[indices] * sf.repeat_interleave(16, dim=-1)


def _inputs(m: int, k: int, inter: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    torch.manual_seed(19362)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn(2 * inter, k, device="cuda", dtype=torch.bfloat16) * 0.25
    ag = (a.abs().max().float() / (448 * 6)).reshape(1)
    bg = (b.abs().max().float() / (448 * 6)).reshape(1)
    aq, asf = torch.ops.trtllm.fp4_quantize(a, ag.reciprocal(), 16, False, True)
    bq, bsf = torch.ops.trtllm.fp4_quantize(b, bg.reciprocal(), 16, False, True)
    alpha = ag * bg
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        gemm = (_dequantize(aq, asf) @ _dequantize(bq, bsf).T) * alpha
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    up, gate = gemm.chunk(2, dim=-1)
    reference = up * F.silu(gate)
    bq = quantization.interleave_linear_and_gate(bq, group_size=64, dim=0)
    sf = utils.unswizzle_sf(bsf, 2 * inter, k)
    bsf = utils.swizzle_sf(
        quantization.interleave_linear_and_gate(sf, group_size=64, dim=0), 2 * inter, k
    )
    norm_const = (448 * 6 / reference.abs().max()).reshape(1)
    return [aq, bq, asf, bsf, alpha, norm_const], reference


def _run(inputs: list[torch.Tensor], fp4_out: bool, use_tvm_ffi: bool = True):
    if fp4_out:
        return torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_rubin(*inputs, use_tvm_ffi)
    return torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_rubin(
        *inputs[:5], torch.bfloat16, use_tvm_ffi
    )


def _check(output, reference: torch.Tensor, inputs: list[torch.Tensor], fp4_out: bool):
    m, inter = reference.shape
    if not fp4_out:
        assert output.shape == (m, inter)
        assert output.dtype == torch.bfloat16
        torch.testing.assert_close(output.float(), reference, rtol=0.008, atol=0.001)
        return
    packed, sf = output
    assert packed.shape == (m, inter // 2)
    assert packed.dtype == torch.uint8
    assert sf.numel() == math_utils.pad_up(m, 128) * math_utils.pad_up(inter // 16, 4)
    # Quantize the FP32 fused result directly: rounding GEMM or SwiGLU to BF16
    # first would change the E4M3 scales at rounding boundaries.
    blocks = reference.reshape(m, inter // 16, 16)
    norm_const = inputs[-1]
    ref_sf = (blocks.abs().amax(dim=-1) * (norm_const / 6)).to(torch.float8_e4m3fn)
    actual_sf = utils.unswizzle_sf(sf, math_utils.pad_up(m, 128), inter)[:m]
    torch.testing.assert_close(
        actual_sf.view(torch.float8_e4m3fn).float(), ref_sf.float(), rtol=0.0, atol=0.0
    )
    scaled = blocks * (norm_const / ref_sf.float()).unsqueeze(-1)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=reference.device)
    # First argmin chooses the lower code; midpoint ties choose the even code.
    distances = (scaled.abs().unsqueeze(-1) - lut).abs()
    codes = distances.argmin(dim=-1)
    ties = distances == distances.amin(dim=-1, keepdim=True)
    even_ties = ties[..., ::2].any(dim=-1)
    even_codes = distances[..., ::2].argmin(dim=-1) * 2
    codes = torch.where(even_ties, even_codes, codes)
    codes = (codes | ((scaled < 0).long() << 3)).reshape(m, inter).to(torch.uint8)
    expected = codes[:, ::2] | (codes[:, 1::2] << 4)
    # Approximate reciprocal/exp can move values at FP4 rounding boundaries.
    match = ((packed & 15) == (expected & 15)).sum() + ((packed >> 4) == (expected >> 4)).sum()
    assert match.item() / reference.numel() > 0.99


@pytest.mark.parametrize(
    "m,k,inter",
    [
        (1, 128, 64),
        (17, 256, 192),
        (64, 512, 128),
        (128, 256, 512),
        (129, 256, 192),
        (257, 512, 512),
    ],
)
@pytest.mark.parametrize("fp4_out", [False, True])
def test_swiglu_rubin_shapes(m, k, inter, fp4_out):
    inputs, reference = _inputs(m, k, inter)
    if fp4_out:
        assert ops.CuteDSLNVFP4SwigluFP4OutRubinRunner().get_valid_tactics(inputs, None)
    _check(_run(inputs, fp4_out), reference, inputs, fp4_out)


@pytest.mark.parametrize(
    "tactic",
    [
        ((128, 128), (1, 1), False),
        ((128, 256), (1, 2), True),
        ((256, 128), (2, 1), True),
        ((256, 256), (2, 2), False),
    ],
)
@pytest.mark.parametrize("fp4_out", [False, True])
@pytest.mark.parametrize("use_tvm_ffi", [False, True])
@pytest.mark.parametrize("m", [1, 129])
def test_swiglu_rubin_tactics(tactic, fp4_out, use_tvm_ffi, m):
    inputs, reference = _inputs(m, 512, 256)
    if fp4_out:
        runner = ops.CuteDSLNVFP4SwigluFP4OutRubinRunner(use_tvm_ffi)
        args = inputs
    else:
        runner = ops.CuteDSLNVFP4SwigluRubinRunner(torch.bfloat16, use_tvm_ffi)
        args = inputs[:5]
    # Exercise prefetch even when the tuning heuristic prunes it for this shape.
    assert (*tactic[:2], False) in runner.get_valid_tactics(args, None)
    _check(runner(args, tactic=tactic), reference, inputs, fp4_out)


@pytest.mark.parametrize("fp4_out", [False, True])
def test_swiglu_rubin_cuda_graph(fp4_out):
    inputs, reference = _inputs(129, 256, 192)
    _run(inputs, fp4_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _run(inputs, fp4_out)
    graph.replay()
    _check(output, reference, inputs, fp4_out)


@pytest.mark.parametrize("fp4_out", [False, True])
def test_swiglu_rubin_fake(fp4_out):
    with fake_tensor.FakeTensorMode():
        x = torch.empty(17, 128, device="cuda", dtype=torch.uint8)
        w = torch.empty(384, 128, device="cuda", dtype=torch.uint8)
        sf = torch.empty(2048, device="cuda", dtype=torch.uint8)
        bsf = torch.empty(6144, device="cuda", dtype=torch.uint8)
        alpha = torch.empty(1, device="cuda")
        output = _run([x, w, sf, bsf, alpha, alpha], fp4_out)
        if fp4_out:
            assert output[0].shape == (17, 96)
            assert output[1].shape == (1536,)
        else:
            assert output.shape == (17, 192)
            assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("excluded", [None, "bias", "dtype", "shape", "flag", "dsl"])
def test_swiglu_rubin_capability(monkeypatch, excluded):
    import types

    from tensorrt_llm._torch.modules import gated_mlp, linear

    layer = linear.Linear.__new__(linear.Linear)
    torch.nn.Module.__init__(layer)
    layer.use_cute_dsl_nvfp4_swiglu_blackwell = True
    layer.use_cute_dsl_blockscaling_mm = excluded != "flag"
    layer._weights_created = True
    layer.quant_method = types.SimpleNamespace(quantizes_nvfp4_activations=True)
    layer.has_bias = excluded == "bias"
    layer.dtype = torch.float16 if excluded == "dtype" else torch.bfloat16
    layer.out_features = 192 if excluded == "shape" else 384
    monkeypatch.setattr(linear, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", excluded != "dsl")

    mlp = gated_mlp.GatedMLP.__new__(gated_mlp.GatedMLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = layer
    mlp.activation = F.silu
    mlp.swiglu_alpha = None
    mlp.swiglu_beta = None

    assert not layer.can_use_cute_dsl_nvfp4_swiglu_blackwell()
    assert layer.can_use_cute_dsl_nvfp4_swiglu() == (excluded is None)
    assert mlp._can_fuse_gate_up_swiglu() == (excluded is None)


@pytest.mark.parametrize("fp4_out", [False, True])
def test_swiglu_rubin_rejects_wrong_architecture(monkeypatch, fp4_out):
    inputs, _ = _inputs(128, 256, 128)
    monkeypatch.setattr(ops, "get_sm_version", lambda: 100)
    with pytest.raises(ValueError, match="requires SM107"):
        _run(inputs, fp4_out)
