# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for split-K in the SM100/SM103 CuTe DSL BF16 dense GEMM."""

import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

skip_not_sm100f = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE or getSMVersion() not in (100, 103),
    reason="requires SM100/SM103 and cutlass-dsl",
)


def _gemm_operands(m, n, k, c_dtype):
    act = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(m, n, device="cuda", dtype=c_dtype)
    return act, weight, output


def _split_k(tactic):
    return tactic[5] if len(tactic) == 6 else 1


@skip_not_sm100f
@pytest.mark.parametrize("split_k_slices", [2, 4, 8])
@pytest.mark.parametrize("c_dtype", [torch.bfloat16, torch.float32])
def test_cute_dsl_bf16_split_k_gemm_blackwell(split_k_slices, c_dtype):
    """Split-K matches the dense reference for large-K, small-N shapes."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLBf16BlackwellGemmRunner

    torch.manual_seed(2026)
    runner = CuteDSLBf16BlackwellGemmRunner()
    act, weight, output = _gemm_operands(64, 256, 7168, c_dtype)
    base = (False, (128, 128), (1, 1))
    tactic = base + ("m", 1, split_k_slices)

    # Direct split-K rounds each partial to the output dtype before the
    # atomic TMA ADD, so BF16 output needs tolerance for both rounding and
    # arrival-order changes.
    rtol, atol = (2e-2, 2.5) if c_dtype == torch.bfloat16 else (1e-2, 1.0)

    expected = act.float() @ weight.t().float()
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=rtol, atol=atol)

    # A second launch must not accumulate on the previous output; poisoning C
    # also catches a missing zero inside CUDA-graph replay.
    output.fill_(float("nan"))
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=rtol, atol=atol)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner([act, weight, output], tactic=tactic)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=rtol, atol=atol)

    if c_dtype == torch.float32:
        split1_output = torch.empty_like(output)
        runner([act, weight, split1_output], tactic=base)
        torch.cuda.synchronize()
        # Both write FP32; direct split-K only changes the reduction order.
        torch.testing.assert_close(output, split1_output, rtol=1e-3, atol=1e-2)


@skip_not_sm100f
def test_cute_dsl_bf16_split_k_gemm_blackwell_full_sweep(monkeypatch):
    """The full sweep splits K only for FP32 output (no BF16 rounding of the
    reduced partials) and only when the output holds every row of A."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLBf16BlackwellGemmRunner

    monkeypatch.setattr(AutoTuner.get(), "nvmmh_config", None)
    runner = CuteDSLBf16BlackwellGemmRunner()
    act, weight, output = _gemm_operands(64, 256, 7168, torch.float32)
    splits = {_split_k(t) for t in runner.get_valid_tactics([act, weight, output], None)}
    assert splits == {1, 2, 4, 8}
    bf16_output = output.to(torch.bfloat16)
    assert all(_split_k(t) == 1 for t in runner.get_valid_tactics([act, weight, bf16_output], None))
    # Dynamic tuning buckets input M while the output keeps the full M.
    bucketed = act[:32]
    assert all(_split_k(t) == 1 for t in runner.get_valid_tactics([bucketed, weight, output], None))


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available")
@pytest.mark.parametrize(
    "runner_name", ["CuteDSLBf16BlackwellGemmRunner", "CuteDSLBf16BlackwellBmmRunner"]
)
def test_cute_dsl_bf16_blackwell_autotune_profiles(runner_name):
    """The output's M follows the bucketed input M, so runtime M values that
    share a bucket share one profiling-cache entry."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    tuning_config = getattr(cute_dsl_custom_ops, runner_name).tuning_config
    batch, n, k = 2, 128, 256
    is_bmm = runner_name.endswith("BmmRunner")

    def make_inputs(m):
        if is_bmm:
            return [torch.empty(batch, m, k), torch.empty(batch, n, k), torch.empty(batch, m, n)]
        return [torch.empty(m, k), torch.empty(n, k), torch.empty(m, n)]

    m_dim = 1 if is_bmm else 0
    profiles = AutoTuner()._optimization_profiles(tuning_config, make_inputs(7))
    for profile in profiles:
        shapes = profile.get_opt_shapes()
        assert shapes[2][m_dim] == shapes[0][m_dim]

    def cache_profile(m):
        return AutoTuner._find_nearest_profile(
            tuple(t.shape for t in make_inputs(m)),
            tuning_config.dynamic_tensor_specs,
            tuning_config.constraint_specs,
            tuning_config.tune_max_num_tokens,
        )

    assert cache_profile(5) == cache_profile(7)
