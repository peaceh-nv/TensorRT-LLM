# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feasibility and launch regression tests for Rubin BF16 preferred clusters."""

import pytest
import torch

try:
    import cutlass
    import cutlass.cute as cute
except ImportError:
    pytest.skip("requires CuTe DSL", allow_module_level=True)

from cuda.bindings import driver
from cutlass.cute import testing

from tensorrt_llm._torch.cute_dsl_kernels.rubin import dense_bf16_gemm_persistent as kernels


def _can_implement(
    use_2cta: bool,
    tile_m: int,
    m: int,
    preferred: tuple[int, int] = (4, 2),
    fallback: tuple[int, int] = (2, 1),
    n: int = 128,
    k: int = 128,
    a_major: str = "k",
    c_major: str = "n",
) -> bool:
    return kernels.PersistentDenseGemmKernelPreferredCluster.can_implement(
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.BFloat16,
        use_2cta,
        (tile_m, 128),
        preferred,
        fallback,
        m,
        n,
        k,
        1,
        a_major,
        "k",
        c_major,
    )


@pytest.mark.parametrize("use_2cta,tile_m", [(False, 64), (False, 128), (True, 128), (True, 256)])
def test_preferred_cluster_m_feasibility(use_2cta: bool, tile_m: int) -> None:
    cta_m = tile_m // (2 if use_2cta else 1)
    # Two real M CTAs fit the fallback but cannot fill the preferred cluster.
    m = 2 * cta_m
    assert kernels.PersistentDenseGemmKernel.can_implement(
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.BFloat16,
        use_2cta,
        (tile_m, 128),
        (2, 1),
        m,
        128,
        128,
        1,
        "k",
        "k",
        "n",
    )
    assert not _can_implement(use_2cta, tile_m, m)
    assert not _can_implement(use_2cta, tile_m, 3 * cta_m)
    # A partial fourth CTA is valid with TMA stores, as is an exact fit.
    assert _can_implement(use_2cta, tile_m, 3 * cta_m + 1)
    assert _can_implement(use_2cta, tile_m, 4 * cta_m)
    # N padding must not be mistaken for the unsafe M padding.
    assert _can_implement(use_2cta, tile_m, 4 * cta_m, n=8)


@pytest.mark.parametrize(
    "preferred,fallback",
    [
        ((0, 2), (2, 1)),
        ((4, 3), (2, 1)),
        ((4, 8), (2, 1)),
        ((4, 2), (0, 1)),
        ((4, 2), (2, 3)),
        ((4, 2), (1, 1)),
        ((2, 1), (4, 2)),
    ],
)
def test_preferred_cluster_rejects_invalid_shapes(
    preferred: tuple[int, int], fallback: tuple[int, int]
) -> None:
    assert not _can_implement(True, 128, 512, preferred, fallback)


@pytest.mark.parametrize("dimension", ["k", "n", "a_major", "c_major"])
def test_preferred_cluster_rejects_misalignment(dimension: str) -> None:
    kwargs = {dimension: 127} if dimension in ("k", "n") else {dimension: "m"}
    assert not _can_implement(True, 128, 257, **kwargs)


@pytest.fixture
def sm107() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    pytest.importorskip("cutlass.utils.rubin_helpers")


def _operands(m: int, batch: int = 1) -> tuple:
    a = torch.randn(batch, m, 128, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(batch, 128, 128, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(batch, m, 128, dtype=torch.bfloat16, device="cuda")
    tensors = tuple(
        cute.runtime.from_dlpack(t.permute(1, 2, 0), assumed_align=16) for t in (a, b, c)
    )
    return a, b, c, tensors


@pytest.mark.parametrize("use_2cta,m", [(False, 256), (True, 128)])
def test_preferred_cluster_rejects_before_launch(sm107: None, use_2cta: bool, m: int) -> None:
    _, _, _, tensors = _operands(m)
    gemm = kernels.PersistentDenseGemmKernelPreferredCluster(
        cutlass.Float32, use_2cta, (128, 128), (4, 2), (2, 1)
    )
    # Compile only: a regression fails this assertion without running unsafe CTAs.
    with pytest.raises(testing.CantImplementError, match="preferred and fallback"):
        cute.compile(gemm, *tensors, 1, 1, driver.CUstream(torch.cuda.current_stream().cuda_stream))


@pytest.mark.parametrize("use_2cta", [False, True])
@pytest.mark.parametrize("entrypoint", ["direct", "wrapper", "wrapper_strided"])
def test_preferred_cluster_valid_gemm(sm107: None, use_2cta: bool, entrypoint: str) -> None:
    torch.manual_seed(18333)
    m = 512 if not use_2cta else 256
    a, b, c, tensors = _operands(m, batch=2)
    gemm = kernels.PersistentDenseGemmKernelPreferredCluster(
        cutlass.Float32, use_2cta, (128, 128), (4, 2), (2, 1)
    )
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    if entrypoint != "direct":
        a_ptr = cute.runtime.make_ptr(cutlass.BFloat16, a.data_ptr(), assumed_align=16)
        b_ptr = cute.runtime.make_ptr(cutlass.BFloat16, b.data_ptr(), assumed_align=16)
        c_tensor = tensors[2].mark_layout_dynamic(leading_dim=1)
        args = (m, 128, 128, 2, a_ptr, b_ptr, c_tensor)
        if entrypoint == "wrapper_strided":
            args += (a.stride(1), a.stride(0), b.stride(1), b.stride(0))
            compiled = cute.compile(gemm.wrapper_strided, *args, 1, 4, stream)
        else:
            compiled = cute.compile(gemm.wrapper, *args, 1, 4, stream)
        compiled(*args, stream)
    else:
        compiled = cute.compile(gemm, *tensors, 1, 4, stream)
        compiled(*tensors, stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        c, torch.bmm(a.float(), b.transpose(1, 2).float()).bfloat16(), rtol=1e-2, atol=1e-2
    )
