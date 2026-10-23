# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the nvMatmulHeuristics guidance of CuTe DSL GEMM/BMM runners."""

import types

import pytest

from tensorrt_llm._torch.autotuner import AutoTuner, NvMMHConfig
from tensorrt_llm._torch.custom_ops import cutedsl_matmul_heuristics as nvmmh
from tensorrt_llm.llmapi import AutoTunerNvMMHConfig


def _config(score, cta=(128, 128, 64), cluster=(1, 1), split_k=1):
    return nvmmh.KernelConfig(
        cta=cta,
        cluster=cluster,
        split_k=split_k,
        swizzle=1,
        cta_order=0,
        stages=4,
        warp=(64, 64, cta[2]),
        instr=(64, 8, 16),
        score=score,
    )


def test_select_tactics_keeps_order_and_counts_distinct_configs():
    ranked = [
        (_config(1.0), [("a", 0), ("a", 1)]),
        # Differs only in a field the kernel does not expose: same tactics.
        (_config(1.1), [("a", 0), ("a", 1)]),
        (_config(1.2), []),
        (_config(1.3), [("b", 0), ("a", 1)]),
        (_config(1.4), [("c", 0)]),
    ]
    assert nvmmh.select_tactics(ranked, 2) == [("a", 0), ("a", 1), ("b", 0)]
    assert nvmmh.select_tactics(ranked, 5) == [("a", 0), ("a", 1), ("b", 0), ("c", 0)]


def test_query_count_oversamples():
    assert nvmmh.query_count(1) == nvmmh.MIN_QUERY_COUNT
    assert nvmmh.query_count(16) == 16 * nvmmh.QUERY_OVERSAMPLING


def test_cta_order_to_raster():
    assert nvmmh.cta_order_to_raster(0) == "m"
    assert nvmmh.cta_order_to_raster(1) == "n"


@pytest.mark.parametrize(
    "version, cluster_m, paired",
    [
        ("0.1.0.27", 2, True),
        ("0.1.0.27", 4, False),
        ("0.1.0.20261021+git.2d9094fc", 4, True),
        ("0.2.0", 4, True),
        ("0.2.0", 1, False),
    ],
)
def test_pairs_ctas_follows_the_library_version(monkeypatch, version, cluster_m, paired):
    nvmmh._pairs_only_cluster_m_2.cache_clear()
    monkeypatch.setattr(nvmmh, "library_version", lambda: version)
    try:
        assert nvmmh.pairs_ctas(_config(1.0, cluster=(cluster_m, 1))) == paired
    finally:
        nvmmh._pairs_only_cluster_m_2.cache_clear()


def test_merge_rankings_follows_the_ranking_score():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import _merge_nvmmh_rankings

    plain = [(_config(2.0), ["p0"]), (_config(5.0), ["p1"])]
    swapped = [(_config(1.0), ["s0"]), (_config(3.0), ["s1"]), (_config(4.0), ["s2"])]
    merged = _merge_nvmmh_rankings([plain, swapped])
    assert [tactics for _, tactics in merged] == [["s0"], ["p0"], ["s1"], ["s2"], ["p1"]]
    # Each ranking keeps nvMMH's own order even when its scores are not
    # sorted (Rubin ranks with an L2 simulator on top of the score).
    unsorted = [(_config(3.0), ["u0"]), (_config(1.0), ["u1"])]
    merged = _merge_nvmmh_rankings([unsorted])
    assert [tactics for _, tactics in merged] == [["u0"], ["u1"]]


def test_sm107_fp8_blockwise_tactics_map_256_row_tiles(monkeypatch):
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("The test requires SM107 CuTe DSL support.")
    import cutlass

    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    configs = [
        _config(1.0, cta=(256, 128, 128), cluster=(2, 1)),
        _config(2.0, cta=(256, 128, 128), cluster=(1, 1)),
        _config(3.0, cta=(128, 128, 128), cluster=(2, 1)),
    ]

    def fake_query(problem, count, is_valid, backend=None, cache_key=None):
        return [config for config in configs if is_valid(config)]

    nvmmh._pairs_only_cluster_m_2.cache_clear()
    monkeypatch.setattr(nvmmh, "library_version", lambda: "0.2.0")
    monkeypatch.setattr(nvmmh, "query", fake_query)
    try:
        args = (ops.SM107BlockwiseGemmKernel, 4096, 1024, 4096, 8)
        # A 256-row CTA tile is two 128-row MMAs sharing B (2-CTA when the
        # cluster M is even).
        assert ops._sm107_fp8_blockwise_nvmmh_tactics(*args, cutlass.BFloat16, 5) == [
            (True, (512, 128, 128), (256, 128, 64), (2, 1)),
            (False, (256, 128, 128), (128, 128, 64), (1, 1)),
            (True, (256, 128, 128), (256, 128, 64), (2, 1)),
        ]
        # The fused FP8 1x128 output needs 128x128 CTA tiles.
        assert ops._sm107_fp8_blockwise_nvmmh_tactics(
            *args, cutlass.Float8E4M3FN, 5, fp8_quantize_1x128=True
        ) == [(True, (256, 128, 128), (256, 128, 64), (2, 1))]
    finally:
        nvmmh._pairs_only_cluster_m_2.cache_clear()


def test_query_without_library(monkeypatch):
    monkeypatch.setattr(nvmmh, "IS_NVMMH_AVAILABLE", False)
    problem = nvmmh.Problem(128, 128, 128, 1, nvmmh.BF16_PRECISION)
    assert nvmmh.query(problem, 8, lambda config: True) == []


def test_query_memoizes_and_survives_failures(monkeypatch):
    calls = []

    def fake_query(problem, count, backend, is_valid):
        calls.append((problem, count, backend))
        return [_config(1.0)]

    monkeypatch.setattr(nvmmh, "IS_NVMMH_AVAILABLE", True)
    monkeypatch.setattr(nvmmh, "_query_uncached", fake_query)
    monkeypatch.setattr(nvmmh, "_query_cache", {})
    problem = nvmmh.Problem(4096, 4096, 4096, 1, nvmmh.BF16_PRECISION)
    backend = nvmmh.Backend(split_k_kind=nvmmh.SPLIT_K_IN_PLACE)
    first = nvmmh.query(problem, 8, lambda c: True, backend, cache_key="k")
    second = nvmmh.query(problem, 8, lambda c: True, backend, cache_key="k")
    assert first == second == [_config(1.0)]
    assert len(calls) == 1
    # A different validity check (cache key) queries again.
    nvmmh.query(problem, 8, lambda c: True, backend, cache_key="other")
    assert len(calls) == 2

    def failing_query(*args):
        raise RuntimeError("no hardware descriptor")

    monkeypatch.setattr(nvmmh, "_query_uncached", failing_query)
    assert nvmmh.query(problem, 16, lambda c: True, backend) == []


def test_ranking_score_falls_back_by_library_capability(monkeypatch):
    metrics = types.SimpleNamespace(RUNTIME_RELATIVE_FAST_S=14, RANKING_SCORE=15)
    monkeypatch.setattr(
        nvmmh, "_nvmmh", types.SimpleNamespace(NvMatmulHeuristicsSiliconMetric=metrics)
    )
    requested = []

    class NewInterface:
        def setBackendPropertyValue(self, *args):
            pass

        def estimateSiliconMetricEx(self, precision, backend, problem, raw, metric):
            requested.append(metric)
            return 2.5e-6

    assert nvmmh._ranking_score(NewInterface(), None, None, None, "TST", 9.0) == 2.5e-6
    assert requested == [15]

    # libheuristics builds without RANKING_SCORE: the fast relative metric.
    monkeypatch.setattr(
        nvmmh,
        "_nvmmh",
        types.SimpleNamespace(
            NvMatmulHeuristicsSiliconMetric=types.SimpleNamespace(RUNTIME_RELATIVE_FAST_S=14)
        ),
    )
    nvmmh._ranking_score(NewInterface(), None, None, None, "TST", 9.0)
    assert requested[-1] == 14

    class OldInterface:
        """0.1.0.x: no per-backend estimate."""

    assert nvmmh._ranking_score(OldInterface(), None, None, None, "TST", 9.0) == 9.0


def test_nvmmh_query_honors_the_validity_check():
    if not nvmmh.IS_NVMMH_AVAILABLE:
        pytest.skip("nvMatmulHeuristics is not installed")
    problem = nvmmh.Problem(4096, 4096, 4096, 1, nvmmh.BF16_PRECISION)
    try:
        configs = nvmmh._query_uncached(
            problem, 16, nvmmh.Backend(), lambda config: config.cta[1] == 128
        )
    except Exception as e:  # noqa: BLE001 - no hardware descriptor here
        pytest.skip(f"nvMatmulHeuristics cannot model this device: {e}")
    if not configs:
        pytest.skip("nvMatmulHeuristics returned no configuration")
    assert all(config.cta[1] == 128 for config in configs)
    assert all(config.score > 0 for config in configs)


def test_autotuner_nvmmh_config_reaches_the_autotuner(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.model_engine import _configure_autotuner_nvmmh

    tuner = AutoTuner.get()
    monkeypatch.setattr(tuner, "nvmmh_config", None)
    _configure_autotuner_nvmmh(
        types.SimpleNamespace(autotuner_nvmmh_config=AutoTunerNvMMHConfig(max_tactics=3))
    )
    assert tuner.nvmmh_config == NvMMHConfig(max_tactics=3)
    _configure_autotuner_nvmmh(types.SimpleNamespace(autotuner_nvmmh_config=None))
    assert tuner.nvmmh_config is None
    # Partial test doubles of the LLM args may omit the field.
    _configure_autotuner_nvmmh(types.SimpleNamespace())
    assert tuner.nvmmh_config is None


def test_autotuner_nvmmh_config_validation():
    assert AutoTunerNvMMHConfig().max_tactics == 5
    with pytest.raises(ValueError):
        AutoTunerNvMMHConfig(max_tactics=0)
    with pytest.raises(ValueError):
        AutoTunerNvMMHConfig(env_controls=True)
    assert NvMMHConfig().max_tactics == 5
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            NvMMHConfig(max_tactics=bad)


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
