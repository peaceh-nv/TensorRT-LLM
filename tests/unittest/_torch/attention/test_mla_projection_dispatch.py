# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Rubin MLA projection dispatch and strided KV expansion."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention import mla as mla_module
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import module as dsv4
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._utils import get_sm_version


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm,dsl,rubin,expected",
    [
        (100, True, False, "blackwell"),
        (103, True, True, "blackwell"),
        (107, True, True, "rubin"),
        (107, True, False, None),
        (107, False, False, None),
        (90, True, True, None),
        (120, True, True, None),
    ],
)
def test_dsv4_q_b_dispatch(monkeypatch, sm, dsl, rubin, expected) -> None:
    monkeypatch.setattr(dsv4, "get_sm_version", lambda: sm)
    monkeypatch.setattr(dsv4, "IS_CUTLASS_DSL_AVAILABLE", dsl)
    monkeypatch.setattr(dsv4, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", rubin)
    # Exercise the contiguous conversion as well as dispatch.
    q = torch.randn(32, 7, dtype=torch.bfloat16).t()
    weight = torch.randn(32, 16, dtype=torch.bfloat16).t()
    calls = []

    def gemm(name, a, b, out):
        assert a.is_contiguous() and b.is_contiguous()
        calls.append(name)
        out.copy_(torch.nn.functional.linear(a, b))

    for arch in ("blackwell", "rubin"):
        monkeypatch.setattr(
            torch.ops.trtllm,
            f"cute_dsl_bf16_gemm_{arch}",
            lambda a, b, out, name=arch: gemm(name, a, b, out),
            raising=False,
        )
    output = dsv4._q_b_proj_cute_dsl_bf16(q, weight)
    torch.testing.assert_close(output, torch.nn.functional.linear(q, weight))
    assert calls == ([] if expected is None else [expected])


def _output_projection(device: str, enabled: bool, rank: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        num_heads_tp=4,
        n_local_groups=2,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        o_lora_rank=rank,
        o_a_proj=torch.randn(2, rank, 32, dtype=torch.bfloat16, device=device),
        o_b_proj=torch.nn.Identity(),
        use_cute_dsl_bf16_bmm=enabled,
        inverse_rotary_emb=SimpleNamespace(rotary_cos_sin=None, is_neox=False),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm,enabled,rubin,rank,expected",
    [
        (107, True, True, 16, "rubin"),
        (107, False, True, 16, "bmm"),
        (107, True, False, 16, "bmm"),
        (107, True, True, 15, "bmm"),
        (100, True, True, 16, "bmm"),
        (90, True, True, 16, "bmm"),
    ],
)
def test_dsv4_o_a_dispatch(monkeypatch, sm, enabled, rubin, rank, expected) -> None:
    monkeypatch.setattr(dsv4, "get_sm_version", lambda: sm)
    monkeypatch.setattr(dsv4, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", rubin)
    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", lambda *args: None)
    calls = []

    def bmm(name, a, b, out):
        calls.append(name)
        assert not out.is_contiguous()
        out.copy_(torch.bmm(a, b))

    monkeypatch.setattr(torch.ops.trtllm, "bmm_out", lambda a, b, out: bmm("bmm", a, b, out))
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_bmm_rubin",
        lambda a, b, out: bmm("rubin", a, b.transpose(1, 2), out),
        raising=False,
    )
    model = _output_projection("cpu", enabled, rank)
    attn = torch.randn(7, 64, dtype=torch.bfloat16)
    output = dsv4.project_sparse_attn_output(model, [attn], torch.arange(7))
    reference = (
        torch.bmm(attn.view(7, 2, 32).transpose(0, 1), model.o_a_proj.transpose(1, 2))
        .transpose(0, 1)
        .flatten(1)
    )
    torch.testing.assert_close(output, reference)
    assert calls == [expected]


def _context_projection(device: str, dtype: torch.dtype, bias: bool = False) -> SimpleNamespace:
    model = SimpleNamespace(
        num_heads_tp=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        qk_head_dim=24,
        v_head_dim=16,
        kv_lora_rank=32,
        kv_b_proj=torch.nn.Linear(32, 128, bias=bias, dtype=dtype, device=device),
        apply_rotary_emb=True,
        out_scale=None,
    )
    model._use_kvbproj_strided_out = MethodType(MLA._use_kvbproj_strided_out, model)
    model._get_kvbproj_head_weights = MethodType(MLA._get_kvbproj_head_weights, model)
    return model


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm100f,dtype,bias,reshaped,expected",
    [
        (True, torch.bfloat16, False, False, True),
        (False, torch.bfloat16, False, False, False),
        (True, torch.float16, False, False, False),
        (True, torch.bfloat16, True, False, False),
        (True, torch.bfloat16, False, True, False),
    ],
)
def test_kvb_strided_eligibility(monkeypatch, sm100f, dtype, bias, reshaped, expected) -> None:
    monkeypatch.setattr(mla_module, "is_sm_100f", lambda: sm100f)
    model = _context_projection("cpu", dtype, bias)
    if reshaped:
        model.kv_b_proj.weight = torch.nn.Parameter(model.kv_b_proj.weight.flatten())
    assert model._use_kvbproj_strided_out() == expected


@pytest.mark.cpu_only
def test_kvb_weight_views_follow_weight_replacement() -> None:
    model = _context_projection("cpu", torch.bfloat16)
    first_k, first_v = model._get_kvbproj_head_weights()
    cached_k, cached_v = model._get_kvbproj_head_weights()
    assert cached_k is first_k and cached_v is first_v
    model.kv_b_proj.weight = torch.nn.Parameter(torch.randn_like(model.kv_b_proj.weight))
    new_k, new_v = model._get_kvbproj_head_weights()
    assert new_k is not first_k and new_v is not first_v
    torch.testing.assert_close(new_k.flatten(0, 1), model.kv_b_proj.weight[:64])
    torch.testing.assert_close(new_v.flatten(0, 1), model.kv_b_proj.weight[64:])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_tokens", [1, 17, 256])
@pytest.mark.parametrize("use_strided", [False, True])
@pytest.mark.parametrize("apply_rope", [False, True])
def test_kvb_context_layout(monkeypatch, device, num_tokens, use_strided, apply_rope) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setattr(mla_module, "is_sm_100f", lambda: use_strided)
    if device == "cpu":
        monkeypatch.setattr(
            torch.ops.trtllm, "bmm_out", lambda a, b, out: out.copy_(torch.bmm(a, b))
        )
    monkeypatch.setattr(mla_module, "maybe_compiled_copy_", lambda dst, src: dst.copy_(src))
    model = _context_projection(device, torch.bfloat16)
    model.apply_rotary_emb = apply_rope
    captured = {}

    def attention(q, k, v, metadata, forward_args):
        captured.update(k=k, v=v, latent=forward_args.latent_cache)
        return forward_args.output

    model.mha = SimpleNamespace(forward=attention)
    ckv = torch.randn(num_tokens, 32, dtype=torch.bfloat16, device=device)
    q = torch.empty(num_tokens, 96, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(num_tokens, 8, dtype=torch.bfloat16, device=device)
    output = torch.empty(num_tokens, 64, dtype=torch.bfloat16, device=device)
    latent_cache = torch.empty(0, device=device)
    reference = model.kv_b_proj(ckv)
    projection = Mock(wraps=model.kv_b_proj.forward)
    monkeypatch.setattr(model.kv_b_proj, "forward", projection)
    result = MLA.forward_context_default(model, q, ckv, k_pe, None, None, output, latent_cache)
    k = captured["k"].view(num_tokens, 4, 24)
    torch.testing.assert_close(k[..., :16].flatten(1), reference[:, :64], atol=0.0625, rtol=0.01)
    torch.testing.assert_close(captured["v"], reference[:, 64:], atol=0.0625, rtol=0.01)
    if apply_rope:
        torch.testing.assert_close(k[..., 16:], k_pe[:, None, :].expand(-1, 4, -1))
    # The FP8 consumer assumes the original KV token stride, including the unused K half.
    assert captured["v"].stride() == reference[:, 64:].stride()
    assert captured["latent"] is latent_cache
    assert result is output
    assert projection.call_count == (0 if use_strided else 1)


@pytest.mark.parametrize("num_tokens", [1, 17, 256])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_dsv4_rubin_bf16_projections(monkeypatch, num_tokens, use_cuda_graph) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 107:
        pytest.skip("requires SM107")
    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("requires Rubin CuTe DSL")
    # Isolate the projections; RoPE numerics have separate DSV4 output-projection coverage.
    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", lambda *args: None)
    model = _output_projection("cuda", True)
    q = torch.randn(num_tokens, 32, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    attn = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(num_tokens, device="cuda")

    def run():
        return (
            dsv4._q_b_proj_cute_dsl_bf16(q, weight),
            dsv4.project_sparse_attn_output(model, [attn], positions),
        )

    if use_cuda_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q_out, o_out = run()
        graph.replay()
    else:
        q_out, o_out = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, torch.nn.functional.linear(q, weight), atol=0.0625, rtol=0.01)
    reference_o = (
        torch.bmm(attn.view(num_tokens, 2, 32).transpose(0, 1), model.o_a_proj.transpose(1, 2))
        .transpose(0, 1)
        .flatten(1)
    )
    torch.testing.assert_close(o_out, reference_o, atol=0.0625, rtol=0.01)
