# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 NVFP4 dense GEMM with fused SwiGLU and optional FP4 output."""

import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import blockscaled_layout, rubin_helpers

from ..blackwell import dense_blockscaled_gemm_act_fusion
from .dense_blockscaled_gemm_persistent import Sm107BlockScaledPersistentDenseGemmKernel


class Sm107BlockScaledPersistentDenseGemmActFusionKernel(
    dense_blockscaled_gemm_act_fusion.Sm100BlockScaledPersistentDenseGemmActFusionKernel
):
    """Reuse the fused epilogue with Rubin's K=128 block-scaled MMA.

    The K tile remains 256 elements (two Rubin instructions), and weights
    retain the interleaved 64-row up/gate layout consumed by the epilogue.
    """

    arch = "sm_107"
    mma_inst_bits_k = 512

    def _make_tiled_mma(
        self, cta_group: tcgen05.CtaGroup, mma_inst_shape: tuple[int, int, int]
    ) -> cute.TiledMma:
        return rubin_helpers.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            mma_inst_shape,
        )

    def _sf_tmem_columns(self, tiled_mma: cute.TiledMma) -> tuple[int, int]:
        sfa = blockscaled_layout.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        sfb = blockscaled_layout.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )
        # TMEM addresses encode the row in the upper 16 bits.
        return tuple(
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, layout)) & 0xFFFF
            for layout in (sfa, sfb)
        )

    def mainloop_s2t_copy_and_partition(
        self, sSF: cute.Tensor, tSF: cute.Tensor
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        # The Rubin K16 scale layout needs an MN broadcast mode on the
        # shared-memory source of the 4x32dp128bit copy.
        return Sm107BlockScaledPersistentDenseGemmKernel._mainloop_s2t_copy_and_partition(
            self, sSF, tSF
        )

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: type[cutlass.Numeric],
        sf_dtype: type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: type[cutlass.Numeric],
    ) -> bool:
        return (
            ab_dtype == cutlass.Float4E2M1FN
            and sf_dtype == cutlass.Float8E4M3FN
            and sf_vec_size == 16
            and c_dtype in (cutlass.BFloat16, cutlass.Float4E2M1FN)
        )
