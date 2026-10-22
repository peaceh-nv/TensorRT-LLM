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
