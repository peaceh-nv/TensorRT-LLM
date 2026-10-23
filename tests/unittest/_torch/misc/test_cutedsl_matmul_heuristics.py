# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the nvMatmulHeuristics guidance of CuTe DSL GEMM/BMM runners."""

import pytest

from tensorrt_llm._torch.autotuner import AutoTuner, NvMMHConfig
from tensorrt_llm._torch.custom_ops import cutedsl_matmul_heuristics as nvmmh


def _sm100_nvmmh_problem(case):
    """(runner, inputs, run, reference) of one SM100 nvMMH-aware runner;
    run(tactic) returns that tactic's output."""
    import torch
    from _torch.helpers import per_block_cast_to_fp8

    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    torch.manual_seed(0)
    bf16 = torch.bfloat16

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=bf16)

    def into(runner, inputs):
        def run(tactic):
            inputs[-1].fill_(float("nan"))
            runner(inputs, tactic=tactic)
            return inputs[-1]

        return run

    def returned(runner, inputs):
        return lambda tactic: runner(inputs, tactic=tactic)

    if case in ("bf16_gemm", "bf16_gemm_fp32_out"):
        # nvMMH splits K of the FP32-output router GEMM and tiles the
        # BF16-output decode GEMM narrowly.
        fp32_out = case == "bf16_gemm_fp32_out"
        m, n, k = (64, 256, 7168) if fp32_out else (64, 2112, 7168)
        c_dtype = torch.float32 if fp32_out else bf16
        a, b = randn(m, k), randn(n, k)
        runner = ops.CuteDSLBf16BlackwellGemmRunner(output_dtype=c_dtype)
        inputs = [a, b, torch.empty(m, n, device="cuda", dtype=c_dtype)]
        return runner, inputs, into(runner, inputs), a.float() @ b.float().t()
    if case in ("bf16_bmm", "bf16_bmm_fp8_out"):
        fp8_out = case == "bf16_bmm_fp8_out"
        batch, m, n, k = 8, 300, 256, 512
        a, b = randn(batch, m, k), randn(batch, n, k)
        runner = (
            ops.CuteDSLBf16BlackwellBmmFp8OutRunner()
            if fp8_out
            else ops.CuteDSLBf16BlackwellBmmRunner()
        )
        c_dtype = torch.float8_e4m3fn if fp8_out else bf16
        inputs = [a, b, torch.empty(batch, m, n, device="cuda", dtype=c_dtype)]
        reference = torch.bmm(a.float(), b.float().transpose(1, 2))
        if fp8_out:
            reference = reference.to(c_dtype).float()
        return runner, inputs, into(runner, inputs), reference
    if case == "fp8_gemm":
        m, n, k = 64, 2048, 7168
        a, b = randn(m, k) / k, randn(n, k) / k
        a_fp8, a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
        b_fp8, b_sf = per_block_cast_to_fp8(b)
        runner = ops.CuteDSLFp8BlackwellRunner()
        inputs = [a_fp8, b_fp8, a_sf, b_sf]
        return runner, inputs, returned(runner, inputs), a.float() @ b.float().t()
    if case == "fp8_bmm":
        # nvMMH splits the 64 K tiles in 3 (unequal slices).
        batch, m, n, k = 4, 64, 512, 8192
        a, b = randn(m, batch, k) / k, randn(batch, n, k) / k
        a_fp8, a_sf = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(a)
        b_fp8 = torch.empty_like(b, dtype=torch.float8_e4m3fn)
        b_sf = torch.empty(batch, n // 128, k // 128, device="cuda", dtype=torch.float32)
        for group in range(batch):
            b_fp8[group], b_sf[group] = per_block_cast_to_fp8(b[group])
        runner = ops.CuteDSLFp8BlackwellBmmRunner()
        inputs = [a_fp8, b_fp8, a_sf, b_sf, torch.empty(batch, m, n, device="cuda", dtype=bf16)]
        reference = torch.einsum("mgk,gnk->gmn", a.float(), b.float())
        return runner, inputs, into(runner, inputs), reference
    assert case == "nvfp4_gemm"
    # nvMMH splits the 64 K blocks in 2 and 3.
    m, n, k = 64, 2048, 16384
    x, w = randn(m, k), randn(n, k)
    x_scale = 2688 / x.abs().max().float()
    w_scale = 2688 / w.abs().max().float()
    x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(x, x_scale, 16, False)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, w_scale, 16, False)
    alpha = (1 / (x_scale * w_scale)).reshape(1)
    inputs = [x_fp4, w_fp4, x_sf, w_sf, alpha]
    runner = ops.CuteDSLNVFP4BlackwellRunner(bf16)
    reference = torch.ops.trtllm.nvfp4_gemm_cutlass(*inputs, bf16).float()
    return runner, inputs, returned(runner, inputs), reference


@pytest.mark.parametrize(
    "case",
    [
        "bf16_gemm",
        "bf16_gemm_fp32_out",
        "bf16_bmm",
        "bf16_bmm_fp8_out",
        "fp8_gemm",
        "fp8_bmm",
        "nvfp4_gemm",
    ],
)
def test_sm100_nvmmh_tactics_compute_the_product(case, monkeypatch):
    """Every nvMMH-guided tactic of the SM100 GEMM/BMM runners is correct,
    including tiles, rasters and split-K factors their full sweeps never
    enumerate."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    from tensorrt_llm._utils import get_sm_version

    if not IS_CUTLASS_DSL_AVAILABLE or get_sm_version() not in (100, 103):
        pytest.skip("The test requires SM100/SM103 and CuTe DSL.")
    if not nvmmh.IS_NVMMH_AVAILABLE:
        pytest.skip("nvMatmulHeuristics is not installed")
    import torch
    from _torch.helpers import calc_diff

    runner, inputs, run, reference = _sm100_nvmmh_problem(case)
    selected = []
    select_tactics = nvmmh.select_tactics

    def recording_select_tactics(ranked, max_tactics):
        selected.append(select_tactics(ranked, max_tactics))
        return selected[-1]

    monkeypatch.setattr(nvmmh, "select_tactics", recording_select_tactics)
    monkeypatch.setattr(AutoTuner.get(), "nvmmh_config", NvMMHConfig(max_tactics=3))
    tactics = runner.get_valid_tactics(inputs, None)
    # The tactics come from nvMMH, not from the full-sweep fallback.
    assert tactics and tactics == selected[-1]
    # FP8 blockwise inputs are quantized from the BF16 reference operands.
    max_diff = 1e-3 if case.startswith("fp8") else 1e-4
    for tactic in tactics:
        output = run(tactic).float()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all(), tactic
        assert calc_diff(output, reference) < max_diff, tactic
