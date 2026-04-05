# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Standalone test for BF16 GEMM + RoPE epilogue kernel.

Exercises the custom op torch.ops.trtllm.cute_dsl_bf16_gemm_rope_blackwell
end-to-end:
  1. Constructs BF16 input tensors A [M, K] and B [N, K].
  2. Creates cos_sin_cache and position_ids for RoPE.
  3. Runs the fused GEMM+RoPE kernel.
  4. Validates against PyTorch reference: matmul + manual RoPE.

Usage:
    python run_dense_gemm_persistent_rope_epilogue.py
    python run_dense_gemm_persistent_rope_epilogue.py --m 512 --k 1536 --num_heads 16
    python run_dense_gemm_persistent_rope_epilogue.py --skip_ref_check
"""

import argparse
import sys
from pathlib import Path

import torch

# Ensure tensorrt_llm is importable
sys.path.insert(0, str(Path(__file__).parents[3]))

# Import custom ops (registers torch.ops.trtllm.*)
import tensorrt_llm._torch.custom_ops  # noqa: F401


def apply_rope_reference(
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    num_heads: int,
) -> torch.Tensor:
    """Apply RoPE to q's rope portion using half-rotation (neox style).

    Args:
        q: [M, num_heads * qk_head_dim] BF16
        cos_sin_cache: [max_seq_len, qk_rope_head_dim] float32
                       interleaved [cos0, sin0, cos1, sin1, ...]
        position_ids: [M] int32
        qk_nope_head_dim: nope dimension per head
        qk_rope_head_dim: rope dimension per head
        num_heads: number of heads

    Returns:
        q with RoPE applied on rope portions, same shape as input.
    """
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    rope_half = qk_rope_head_dim // 2
    m = q.shape[0]

    # Work in float32 for precision
    q_out = q.clone().float()

    for token_idx in range(m):
        pos = position_ids[token_idx].item()
        for head_idx in range(num_heads):
            head_offset = head_idx * qk_head_dim + qk_nope_head_dim
            for i in range(rope_half):
                v1 = q_out[token_idx, head_offset + i].item()
                v2 = q_out[token_idx, head_offset + rope_half + i].item()
                cos_val = cos_sin_cache[pos, 2 * i].item()
                sin_val = cos_sin_cache[pos, 2 * i + 1].item()
                q_out[token_idx, head_offset + i] = v1 * cos_val - v2 * sin_val
                q_out[token_idx,
                      head_offset + rope_half + i] = v1 * sin_val + v2 * cos_val

    return q_out.to(q.dtype)


def run_test(
    m: int,
    k: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    max_seq_len: int,
    skip_ref_check: bool,
    use_tvm_ffi: bool,
) -> bool:
    """Run fused GEMM+RoPE and optionally compare against reference."""
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    n = num_heads * qk_head_dim

    print(f"\n{'='*60}")
    print(f"Test: M={m}, K={k}, N={n} (heads={num_heads}, "
          f"nope={qk_nope_head_dim}, rope={qk_rope_head_dim})")
    print(f"{'='*60}")

    device = torch.device("cuda")

    # Create inputs
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1

    # Create cos_sin_cache: [max_seq_len, qk_rope_head_dim] float32
    # interleaved [cos0, sin0, cos1, sin1, ...]
    rope_half = qk_rope_head_dim // 2
    freqs = torch.randn(max_seq_len, rope_half, device=device)
    cos_vals = torch.cos(freqs)
    sin_vals = torch.sin(freqs)
    cos_sin_cache = torch.stack([cos_vals, sin_vals],
                                dim=-1).reshape(max_seq_len,
                                                -1).contiguous().float()
    assert cos_sin_cache.shape == (max_seq_len, qk_rope_head_dim)

    # Create position_ids: [M] int32, random positions in [0, max_seq_len)
    position_ids = torch.randint(0,
                                 max_seq_len, (m, ),
                                 dtype=torch.int32,
                                 device=device)

    # Allocate output
    output = torch.empty(m, n, dtype=torch.bfloat16, device=device)

    # Run fused kernel
    print("Running fused GEMM+RoPE kernel...")
    torch.ops.trtllm.cute_dsl_bf16_gemm_rope_blackwell(
        a,
        b,
        output,
        cos_sin_cache,
        position_ids,
        qk_nope_head_dim,
        qk_rope_head_dim,
        use_tvm_ffi,
    )
    torch.cuda.synchronize()
    print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"  Output range: [{output.float().min():.4f}, "
          f"{output.float().max():.4f}]")

    if skip_ref_check:
        print("  Skipping reference check.")
        return True

    # Compute reference: matmul + RoPE
    print("Computing reference (matmul + RoPE)...")
    ref_gemm = torch.mm(a.float(), b.float().t()).to(torch.bfloat16)
    ref_output = apply_rope_reference(ref_gemm, cos_sin_cache, position_ids,
                                      qk_nope_head_dim, qk_rope_head_dim,
                                      num_heads)

    # Compare
    diff = (output.float() - ref_output.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # Relative error
    ref_abs = ref_output.float().abs()
    rel_err = (diff / (ref_abs + 1e-8)).mean().item()

    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Mean relative error: {rel_err:.6f}")

    # Check nope portion (should be just GEMM output, no RoPE)
    nope_diffs = []
    rope_diffs = []
    for head_idx in range(num_heads):
        nope_start = head_idx * qk_head_dim
        nope_end = nope_start + qk_nope_head_dim
        rope_start = nope_end
        rope_end = (head_idx + 1) * qk_head_dim

        nope_diff = (output[:, nope_start:nope_end].float() -
                     ref_output[:, nope_start:nope_end].float()).abs().max()
        rope_diff = (output[:, rope_start:rope_end].float() -
                     ref_output[:, rope_start:rope_end].float()).abs().max()
        nope_diffs.append(nope_diff.item())
        rope_diffs.append(rope_diff.item())

    print(f"  Max nope diff (per head): {max(nope_diffs):.6f}")
    print(f"  Max rope diff (per head): {max(rope_diffs):.6f}")

    # BF16 has ~7 bits of mantissa, so rtol ~1e-2 is reasonable for GEMM+RoPE
    passed = max_diff < 0.5 and rel_err < 0.05
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Test fused GEMM+RoPE epilogue kernel")
    parser.add_argument("--m",
                        type=int,
                        default=64,
                        help="Number of tokens (M dimension)")
    parser.add_argument("--k",
                        type=int,
                        default=1536,
                        help="q_lora_rank (K dimension)")
    parser.add_argument("--num_heads",
                        type=int,
                        default=16,
                        help="Number of attention heads (after TP)")
    parser.add_argument("--qk_nope_head_dim",
                        type=int,
                        default=128,
                        help="Nope dimension per head")
    parser.add_argument("--qk_rope_head_dim",
                        type=int,
                        default=64,
                        help="Rope dimension per head")
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--skip_ref_check",
                        action="store_true",
                        help="Skip reference comparison")
    parser.add_argument("--use_tvm_ffi",
                        default=True,
                        action="store_true",
                        help="Use TVM FFI")
    args = parser.parse_args()

    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA capability: {torch.cuda.get_device_capability()}")

    all_passed = True

    # Default test: DeepSeek-R1 dimensions
    all_passed &= run_test(
        m=args.m,
        k=args.k,
        num_heads=args.num_heads,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        max_seq_len=args.max_seq_len,
        skip_ref_check=args.skip_ref_check,
        use_tvm_ffi=args.use_tvm_ffi,
    )

    if not args.skip_ref_check:
        # Additional tests with different M sizes
        for m_val in [1, 16, 128, 256, 512]:
            if m_val == args.m:
                continue
            all_passed &= run_test(
                m=m_val,
                k=args.k,
                num_heads=args.num_heads,
                qk_nope_head_dim=args.qk_nope_head_dim,
                qk_rope_head_dim=args.qk_rope_head_dim,
                max_seq_len=args.max_seq_len,
                skip_ref_check=False,
                use_tvm_ffi=args.use_tvm_ffi,
            )

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
