/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaChunkedPrefill.cuh"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/thop/attentionOp.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <cstdint>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
using tk::KVBlockArray;
using tensorrt_llm::torch_ext::buildPagedKvCacheBuffers;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

template <typename T, typename TCache>
void loadPagedKVCacheForMLAHelper(torch::Tensor& compressed_kv, torch::Tensor& k_pe, KVBlockArray& kv_cache,
    int const num_contexts, torch::Tensor const& cu_ctx_cached_kv_lens, int const max_input_seq_len,
    int const lora_size, int const rope_size, float const* kv_scale_quant_orig_ptr)
{
    auto stream = at::cuda::getCurrentCUDAStream(compressed_kv.get_device());

    auto* compressed_kv_ptr = static_cast<T*>(compressed_kv.data_ptr());
    auto* k_pe_ptr = static_cast<T*>(k_pe.data_ptr());
    auto const* cu_ctx_cached_kv_lens_ptr = cu_ctx_cached_kv_lens.data_ptr<int64_t>();
    tensorrt_llm::kernels::invokeMLALoadPagedKV<T, TCache>(compressed_kv_ptr, k_pe_ptr, kv_cache, num_contexts,
        cu_ctx_cached_kv_lens_ptr, max_input_seq_len, lora_size, rope_size, kv_scale_quant_orig_ptr, stream);
}

template <typename T, typename TCache>
void loadChunkedKVCacheForMLAHelper(torch::Tensor& output_kv, torch::Tensor& output_k_pe, KVBlockArray& kv_cache,
    int const num_contexts, torch::Tensor const& cu_ctx_chunked_len, torch::Tensor const& chunked_ld_global_offset,
    int lora_size, int rope_size, int const max_seq_len, float const* kv_scale_quant_orig_ptr)
{
    auto stream = at::cuda::getCurrentCUDAStream(output_kv.get_device());

    T* output_kv_ptr = static_cast<T*>(output_kv.data_ptr());
    T* output_k_pe_ptr = static_cast<T*>(output_k_pe.data_ptr());
    tensorrt_llm::kernels::invokeMLALoadChunkedKV<T, TCache>(output_kv_ptr, output_k_pe_ptr, kv_cache, num_contexts,
        cu_ctx_chunked_len.data_ptr<int64_t>(), chunked_ld_global_offset.data_ptr<int64_t>(), lora_size, rope_size,
        max_seq_len, kv_scale_quant_orig_ptr, stream);
}

template <typename T, typename TCache>
void invokeMLARopeAppendPagedKVAssignQHelper(KVBlockArray& kv_cache, torch::Tensor& q, torch::Tensor& latent_cache,
    int const num_requests, torch::Tensor const& cu_ctx_cached_kv_lens, torch::Tensor const& cu_seq_lens,
    int const max_input_uncached_seq_len, torch::Tensor const& cos_sin_cache, int const head_num, int const nope_size,
    int const rope_size, int const lora_size, float const* kv_scale_orig_quant_ptr, bool const apply_q_rope)
{
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    auto* q_ptr = static_cast<T*>(q.data_ptr());
    auto* latent_cache_ptr = static_cast<T*>(latent_cache.data_ptr());
    auto const* cu_ctx_cached_kv_lens_ptr = cu_ctx_cached_kv_lens.data_ptr<int64_t>();
    auto const* cu_seq_lens_ptr = cu_seq_lens.data_ptr<int64_t>();
    auto const* cos_sin_cache_ptr = static_cast<float2 const*>(cos_sin_cache.data_ptr());
    tensorrt_llm::kernels::invokeMLARopeAppendPagedKVAssignQ<T, TCache>(kv_cache, q_ptr, latent_cache_ptr, num_requests,
        cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr, max_input_uncached_seq_len, cos_sin_cache_ptr, head_num, nope_size,
        rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope, stream);
}

template <typename T, typename TCache>
void invokeMLARopeAppendPagedKVSplitInputHelper(KVBlockArray& kv_cache, torch::Tensor& q,
    torch::Tensor& compressed_kv, torch::Tensor& k_pe, int const num_requests,
    torch::Tensor const& cu_ctx_cached_kv_lens, torch::Tensor const& cu_seq_lens,
    int const max_input_uncached_seq_len, torch::Tensor const& cos_sin_cache, int const head_num, int const nope_size,
    int const rope_size, int const lora_size, float const* kv_scale_orig_quant_ptr, bool const apply_q_rope)
{
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    auto* q_ptr = static_cast<T*>(q.data_ptr());
    auto* compressed_kv_ptr = static_cast<T*>(compressed_kv.data_ptr());
    auto* k_pe_ptr = static_cast<T*>(k_pe.data_ptr());
    auto const* cu_ctx_cached_kv_lens_ptr = cu_ctx_cached_kv_lens.data_ptr<int64_t>();
    auto const* cu_seq_lens_ptr = cu_seq_lens.data_ptr<int64_t>();
    auto const* cos_sin_cache_ptr = static_cast<float2 const*>(cos_sin_cache.data_ptr());
    tensorrt_llm::kernels::invokeMLARopeAppendPagedKVSplitInput<T, TCache>(kv_cache, q_ptr, compressed_kv_ptr, k_pe_ptr,
        num_requests, cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr, max_input_uncached_seq_len, cos_sin_cache_ptr,
        head_num, nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope, stream);
}

template <typename T>
void mlaPackKFromKVRopeHelper(torch::Tensor const& kv, torch::Tensor const& k_pe_roped, torch::Tensor& k,
    int64_t const head_num, int64_t const qk_nope_head_dim, int64_t const qk_rope_head_dim, int64_t const v_head_dim)
{
    auto stream = at::cuda::getCurrentCUDAStream(kv.get_device());
    auto const* kv_ptr = static_cast<T const*>(kv.data_ptr());
    auto const* k_pe_roped_ptr = static_cast<T const*>(k_pe_roped.data_ptr());
    auto* k_ptr = static_cast<T*>(k.data_ptr());
    tensorrt_llm::kernels::invokeMLAPackKFromKVRope<T>(kv_ptr, k_pe_roped_ptr, k_ptr, static_cast<int>(kv.size(0)),
        static_cast<int>(head_num), static_cast<int>(qk_nope_head_dim), static_cast<int>(qk_rope_head_dim),
        static_cast<int>(v_head_dim), stream);
}

template <typename T>
void mlaContextFp8QuantizeKVSplitHelper(torch::Tensor const& q, torch::Tensor const& kv,
    torch::Tensor const& k_pe_roped, torch::Tensor& quant_q, torch::Tensor& quant_k, torch::Tensor& quant_v,
    torch::optional<torch::Tensor> const& quant_scale_qkv, torch::Tensor& bmm1_scale, torch::Tensor& bmm2_scale,
    torch::optional<torch::Tensor> const& quant_scale_o, torch::optional<torch::Tensor> const& dequant_scale_q,
    torch::optional<torch::Tensor> const& dequant_scale_kv, double const host_bmm1_scale, int64_t const head_num,
    int64_t const qk_nope_head_dim, int64_t const qk_rope_head_dim, int64_t const v_head_dim)
{
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    auto const* q_ptr = static_cast<T const*>(q.data_ptr());
    auto const* kv_ptr = static_cast<T const*>(kv.data_ptr());
    auto const* k_pe_roped_ptr = static_cast<T const*>(k_pe_roped.data_ptr());
    auto* quant_q_ptr = quant_q.data_ptr();
    auto* quant_k_ptr = quant_k.data_ptr();
    auto* quant_v_ptr = quant_v.data_ptr();
    auto const* quant_scale_qkv_ptr = quant_scale_qkv.has_value() ? quant_scale_qkv.value().data_ptr<float>() : nullptr;
    auto* bmm1_scale_ptr = bmm1_scale.data_ptr<float>();
    auto* bmm2_scale_ptr = bmm2_scale.data_ptr<float>();
    auto const* quant_scale_o_ptr = quant_scale_o.has_value() ? quant_scale_o.value().data_ptr<float>() : nullptr;
    auto const* dequant_scale_q_ptr = dequant_scale_q.has_value() ? dequant_scale_q.value().data_ptr<float>() : nullptr;
    auto const* dequant_scale_kv_ptr
        = dequant_scale_kv.has_value() ? dequant_scale_kv.value().data_ptr<float>() : nullptr;
    tensorrt_llm::kernels::invokeMLAContextFp8QuantizeKVSplit<T>(q_ptr, kv_ptr, k_pe_roped_ptr, quant_q_ptr,
        quant_k_ptr, quant_v_ptr, static_cast<int>(q.size(0)), static_cast<int>(kv.size(0)), static_cast<int>(head_num),
        static_cast<int>(qk_nope_head_dim), static_cast<int>(qk_rope_head_dim), static_cast<int>(v_head_dim),
        quant_scale_qkv_ptr, bmm1_scale_ptr, bmm2_scale_ptr, quant_scale_o_ptr, dequant_scale_q_ptr,
        dequant_scale_kv_ptr, static_cast<float>(host_bmm1_scale), stream);
}

template <typename T>
void mergeChunkedAttentionForMLAHelper(torch::Tensor& merged_attn, torch::Tensor const& temp_attn,
    torch::Tensor& merged_softmax_stats, torch::Tensor const& temp_softmax_stats, int64_t const num_requests,
    torch::Tensor const& cu_q_seq_lens, int64_t const max_q_seq_len, torch::Tensor const& merge_op,
    int64_t const num_heads, int64_t const head_size)
{
    auto stream = at::cuda::getCurrentCUDAStream(merged_attn.get_device());
    T* merged_attn_ptr = static_cast<T*>(merged_attn.data_ptr());
    T* temp_attn_ptr = static_cast<T*>(temp_attn.data_ptr());
    float* merged_softmax_stats_ptr = static_cast<float*>(merged_softmax_stats.data_ptr());
    float* temp_softmax_stats_ptr = static_cast<float*>(temp_softmax_stats.data_ptr());
    int64_t* const cu_q_seq_lens_ptr = cu_q_seq_lens.data_ptr<int64_t>();
    int64_t* const merge_op_ptr = merge_op.data_ptr<int64_t>();

    tensorrt_llm::kernels::invokeMergeAttnWithSoftmax(merged_attn_ptr, merged_softmax_stats_ptr, merged_attn_ptr,
        merged_softmax_stats_ptr, temp_attn_ptr, temp_softmax_stats_ptr, num_requests, cu_q_seq_lens_ptr, max_q_seq_len,
        merge_op_ptr, num_heads, head_size, stream);
}

} // namespace

std::vector<torch::Tensor> loadPagedKVCacheForMLA(torch::ScalarType out_dtype, int64_t const num_contexts,
    int64_t const num_ctx_cached_tokens, int64_t const max_ctx_cached_kv_len, torch::Tensor& cu_ctx_cached_kv_lens,
    torch::Tensor const& kv_cache_block_offsets, torch::Tensor const& host_kv_cache_pool_pointers,
    torch::Tensor const& host_kv_cache_pool_mapping, torch::optional<torch::Tensor> kv_scale_orig_quant,
    torch::optional<torch::Tensor> kv_scale_quant_orig, int64_t const layer_idx, int64_t const lora_size,
    int64_t const rope_size, int64_t const tokens_per_block, int64_t const attention_window_size,
    int64_t const sink_token_length, int64_t const beam_width, int64_t const quant_mode)
{
    TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kFloat32 || out_dtype == torch::kBFloat16,
        "out_dtype only support float16, float32, bfloat16");
    TLLM_CHECK(num_contexts > 0);
    TORCH_CHECK(num_ctx_cached_tokens > 0);
    TLLM_CHECK(max_ctx_cached_kv_len > 0);
    CHECK_INPUT(cu_ctx_cached_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_cached_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_cached_kv_lens.size(0) >= num_contexts + 1);

    auto kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    int head_size = lora_size + rope_size;
    KVBlockArray kv_cache_buffer
        = buildPagedKvCacheBuffers(std::optional(kv_cache_block_offsets), std::optional(host_kv_cache_pool_pointers),
            std::optional(host_kv_cache_pool_mapping), kv_cache_quant_mode, layer_idx, num_contexts, tokens_per_block,
            1 /*kv_head_num*/, head_size, attention_window_size, attention_window_size, sink_token_length, beam_width,
            0 /*seq_offset*/, true /*is_mla_enable*/, torch::elementSize(out_dtype))
              .kvCacheBuffer;

    float const* kv_scale_orig_quant_ptr = nullptr;
    float const* kv_scale_quant_orig_ptr = nullptr;
    if (kv_cache_quant_mode.hasKvCacheQuant())
    {
        TLLM_CHECK_WITH_INFO(kv_cache_quant_mode.hasFp8KvCache(), "Only FP8 KV cache is supported for now");
        TORCH_CHECK(kv_scale_orig_quant.has_value());
        TORCH_CHECK(kv_scale_quant_orig.has_value());
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
        TLLM_CHECK(kv_scale_orig_quant_ptr != nullptr);
        TLLM_CHECK(kv_scale_quant_orig_ptr != nullptr);
    }

    std::vector<torch::Tensor> outputs;
    // compressed_kv {num_ctx_cached_tokens, lora_size}
    outputs.push_back(torch::empty(
        {num_ctx_cached_tokens, lora_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false)));
    // k_pe {num_ctx_cached_tokens, rope_size}
    outputs.push_back(torch::empty(
        {num_ctx_cached_tokens, rope_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false)));

    if (out_dtype == torch::kFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadPagedKVCacheForMLAHelper<half, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size, kv_scale_quant_orig_ptr);
        }
        else
        {
            loadPagedKVCacheForMLAHelper<half, half>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size, kv_scale_quant_orig_ptr);
        }
    }
    else if (out_dtype == torch::kFloat32)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadPagedKVCacheForMLAHelper<float, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size, kv_scale_quant_orig_ptr);
        }
        else
        {
            loadPagedKVCacheForMLAHelper<float, float>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size, kv_scale_quant_orig_ptr);
        }
    }
    else if (out_dtype == torch::kBFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadPagedKVCacheForMLAHelper<__nv_bfloat16, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer,
                num_contexts, cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size,
                kv_scale_quant_orig_ptr);
        }
        else
        {
            loadPagedKVCacheForMLAHelper<__nv_bfloat16, __nv_bfloat16>(outputs[0], outputs[1], kv_cache_buffer,
                num_contexts, cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size,
                kv_scale_quant_orig_ptr);
        }
    }

    return outputs;
}

std::vector<torch::Tensor> loadChunkedKVCacheForMLA(torch::ScalarType out_dtype, int64_t const num_contexts,
    int64_t const num_ctx_cached_tokens, torch::Tensor const& cu_ctx_chunked_kv_lens,
    torch::Tensor const& chunked_ld_global_offset, torch::Tensor const& kv_cache_block_offsets,
    torch::Tensor const& host_kv_cache_pool_pointers, torch::Tensor const& host_kv_cache_pool_mapping,
    torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
    int64_t const layer_idx, int64_t const lora_size, int64_t const rope_size, int64_t const tokens_per_block,
    int64_t const max_seq_len, int64_t const attention_window_size, int64_t const sink_token_length,
    int64_t const beam_width, int64_t const quant_mode)
{
    TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kFloat32 || out_dtype == torch::kBFloat16,
        "out_dtype only support float16, float32, bfloat16");
    TLLM_CHECK(num_contexts > 0);
    CHECK_INPUT(cu_ctx_chunked_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_chunked_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_chunked_kv_lens.size(0) >= num_contexts + 1);
    int head_size = lora_size + rope_size;
    auto kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    KVBlockArray kv_cache_buffer
        = buildPagedKvCacheBuffers(std::optional(kv_cache_block_offsets), std::optional(host_kv_cache_pool_pointers),
            std::optional(host_kv_cache_pool_mapping), kv_cache_quant_mode, layer_idx, num_contexts, tokens_per_block,
            1 /*kv_head_num*/, head_size, attention_window_size, attention_window_size, sink_token_length, beam_width,
            0 /*seq_offset*/, true /*is_mla_enable*/, torch::elementSize(out_dtype))
              .kvCacheBuffer;

    float const* kv_scale_orig_quant_ptr = nullptr;
    float const* kv_scale_quant_orig_ptr = nullptr;
    if (kv_cache_quant_mode.hasKvCacheQuant())
    {
        TORCH_CHECK(kv_scale_orig_quant.has_value());
        TORCH_CHECK(kv_scale_quant_orig.has_value());
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
        TLLM_CHECK(kv_scale_orig_quant_ptr != nullptr);
        TLLM_CHECK(kv_scale_quant_orig_ptr != nullptr);
    }

    std::vector<torch::Tensor> outputs;

    // compressed_kv {num_ctx_cached_tokens, lora_size}
    outputs.push_back(torch::empty(
        {num_ctx_cached_tokens, lora_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false)));
    // k_pe {num_ctx_cached_tokens, rope_size}
    outputs.push_back(torch::empty(
        {num_ctx_cached_tokens, rope_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false)));

    if (out_dtype == torch::kFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadChunkedKVCacheForMLAHelper<half, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
        else
        {
            loadChunkedKVCacheForMLAHelper<half, half>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
    }
    else if (out_dtype == torch::kFloat32)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadChunkedKVCacheForMLAHelper<float, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
        else
        {
            loadChunkedKVCacheForMLAHelper<float, float>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
    }
    else if (out_dtype == torch::kBFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadChunkedKVCacheForMLAHelper<__nv_bfloat16, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer,
                num_contexts, cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
        else
        {
            loadChunkedKVCacheForMLAHelper<__nv_bfloat16, __nv_bfloat16>(outputs[0], outputs[1], kv_cache_buffer,
                num_contexts, cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
    }

    return outputs;
}

void MLARopeAppendPagedKVAssignQ(torch::Tensor& q, torch::Tensor& latent_cache, int64_t const num_contexts,
    torch::Tensor const& cu_ctx_cached_kv_lens, torch::Tensor const& cu_seq_lens,
    int64_t const max_input_uncached_seq_len, torch::Tensor const& cos_sin_cache, int64_t const head_num,
    int64_t const nope_size, int64_t const rope_size, int64_t const lora_size,
    torch::Tensor const& kv_cache_block_offsets, torch::Tensor const& host_kv_cache_pool_pointers,
    torch::Tensor const& host_kv_cache_pool_mapping, torch::optional<torch::Tensor> kv_scale_orig_quant,
    torch::optional<torch::Tensor> kv_scale_quant_orig, int64_t const layer_idx, int64_t const tokens_per_block,
    int64_t const attention_window_size, int64_t const sink_token_length, int64_t const beam_width,
    int64_t const quant_mode, bool const apply_q_rope)
{
    auto input_dtype = q.scalar_type();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kFloat32 || input_dtype == torch::kBFloat16);
    TORCH_CHECK(q.numel() > 0);
    TORCH_CHECK(q.dim() == 2);
    CHECK_TH_CUDA(q);
    CHECK_CONTIGUOUS(q);
    CHECK_INPUT(latent_cache, input_dtype);
    TORCH_CHECK(latent_cache.dim() == 2);
    CHECK_INPUT(cu_seq_lens, torch::kInt64);
    TORCH_CHECK(cu_seq_lens.dim() == 1);
    TORCH_CHECK(cu_seq_lens.size(0) >= num_contexts + 1);
    CHECK_INPUT(cu_ctx_cached_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_cached_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_cached_kv_lens.size(0) >= num_contexts + 1);
    TORCH_CHECK(max_input_uncached_seq_len > 0);

    auto kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    int head_size = lora_size + rope_size;
    KVBlockArray kv_cache_buffer
        = buildPagedKvCacheBuffers(std::optional(kv_cache_block_offsets), std::optional(host_kv_cache_pool_pointers),
            std::optional(host_kv_cache_pool_mapping), kv_cache_quant_mode, layer_idx, num_contexts, tokens_per_block,
            1 /*kv_head_num*/, head_size, attention_window_size, attention_window_size, sink_token_length, beam_width,
            0 /*seq_offset*/, true /*is_mla_enable*/, torch::elementSize(input_dtype))
              .kvCacheBuffer;

    float const* kv_scale_orig_quant_ptr = nullptr;
    float const* kv_scale_quant_orig_ptr = nullptr;
    if (kv_cache_quant_mode.hasKvCacheQuant())
    {
        TLLM_CHECK_WITH_INFO(kv_cache_quant_mode.hasFp8KvCache(), "Only FP8 KV cache is supported for now");
        TORCH_CHECK(kv_scale_orig_quant.has_value());
        TORCH_CHECK(kv_scale_quant_orig.has_value());
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
        TLLM_CHECK(kv_scale_orig_quant_ptr != nullptr);
        TLLM_CHECK(kv_scale_quant_orig_ptr != nullptr);
    }

    if (input_dtype == torch::kFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVAssignQHelper<half, __nv_fp8_e4m3>(kv_cache_buffer, q, latent_cache, num_contexts,
                cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num, nope_size,
                rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
        else
        {
            invokeMLARopeAppendPagedKVAssignQHelper<half, half>(kv_cache_buffer, q, latent_cache, num_contexts,
                cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num, nope_size,
                rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
    }
    else if (input_dtype == torch::kFloat32)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVAssignQHelper<float, __nv_fp8_e4m3>(kv_cache_buffer, q, latent_cache,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
        else
        {
            invokeMLARopeAppendPagedKVAssignQHelper<float, float>(kv_cache_buffer, q, latent_cache, num_contexts,
                cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num, nope_size,
                rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
    }
    else if (input_dtype == torch::kBFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVAssignQHelper<__nv_bfloat16, __nv_fp8_e4m3>(kv_cache_buffer, q, latent_cache,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
        else
        {
            invokeMLARopeAppendPagedKVAssignQHelper<__nv_bfloat16, __nv_bfloat16>(kv_cache_buffer, q, latent_cache,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
    }
}

void MLARopeAppendPagedKVSplitInput(torch::Tensor& q, torch::Tensor& compressed_kv, torch::Tensor& k_pe,
    int64_t const num_contexts, torch::Tensor const& cu_ctx_cached_kv_lens, torch::Tensor const& cu_seq_lens,
    int64_t const max_input_uncached_seq_len, torch::Tensor const& cos_sin_cache, int64_t const head_num,
    int64_t const nope_size, int64_t const rope_size, int64_t const lora_size,
    torch::Tensor const& kv_cache_block_offsets, torch::Tensor const& host_kv_cache_pool_pointers,
    torch::Tensor const& host_kv_cache_pool_mapping, torch::optional<torch::Tensor> kv_scale_orig_quant,
    torch::optional<torch::Tensor> kv_scale_quant_orig, int64_t const layer_idx, int64_t const tokens_per_block,
    int64_t const attention_window_size, int64_t const sink_token_length, int64_t const beam_width,
    int64_t const quant_mode, bool const apply_q_rope)
{
    auto input_dtype = q.scalar_type();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kFloat32 || input_dtype == torch::kBFloat16);
    TORCH_CHECK(q.numel() > 0);
    TORCH_CHECK(q.dim() == 2);
    CHECK_TH_CUDA(q);
    CHECK_CONTIGUOUS(q);
    CHECK_INPUT(compressed_kv, input_dtype);
    TORCH_CHECK(compressed_kv.dim() == 2);
    CHECK_INPUT(k_pe, input_dtype);
    TORCH_CHECK(k_pe.dim() == 2);
    CHECK_INPUT(cu_seq_lens, torch::kInt64);
    TORCH_CHECK(cu_seq_lens.dim() == 1);
    TORCH_CHECK(cu_seq_lens.size(0) >= num_contexts + 1);
    CHECK_INPUT(cu_ctx_cached_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_cached_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_cached_kv_lens.size(0) >= num_contexts + 1);
    TORCH_CHECK(max_input_uncached_seq_len > 0);

    auto kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    int head_size = lora_size + rope_size;
    KVBlockArray kv_cache_buffer
        = buildPagedKvCacheBuffers(std::optional(kv_cache_block_offsets), std::optional(host_kv_cache_pool_pointers),
            std::optional(host_kv_cache_pool_mapping), kv_cache_quant_mode, layer_idx, num_contexts, tokens_per_block,
            1 /*kv_head_num*/, head_size, attention_window_size, attention_window_size, sink_token_length, beam_width,
            0 /*seq_offset*/, true /*is_mla_enable*/, torch::elementSize(input_dtype))
              .kvCacheBuffer;

    float const* kv_scale_orig_quant_ptr = nullptr;
    float const* kv_scale_quant_orig_ptr = nullptr;
    if (kv_cache_quant_mode.hasKvCacheQuant())
    {
        TLLM_CHECK_WITH_INFO(kv_cache_quant_mode.hasFp8KvCache(), "Only FP8 KV cache is supported for now");
        TORCH_CHECK(kv_scale_orig_quant.has_value());
        TORCH_CHECK(kv_scale_quant_orig.has_value());
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
        TLLM_CHECK(kv_scale_orig_quant_ptr != nullptr);
        TLLM_CHECK(kv_scale_quant_orig_ptr != nullptr);
    }

    if (input_dtype == torch::kFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVSplitInputHelper<half, __nv_fp8_e4m3>(kv_cache_buffer, q, compressed_kv, k_pe,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
        else
        {
            invokeMLARopeAppendPagedKVSplitInputHelper<half, half>(kv_cache_buffer, q, compressed_kv, k_pe,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
    }
    else if (input_dtype == torch::kFloat32)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVSplitInputHelper<float, __nv_fp8_e4m3>(kv_cache_buffer, q, compressed_kv, k_pe,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
        else
        {
            invokeMLARopeAppendPagedKVSplitInputHelper<float, float>(kv_cache_buffer, q, compressed_kv, k_pe,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
    }
    else if (input_dtype == torch::kBFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVSplitInputHelper<__nv_bfloat16, __nv_fp8_e4m3>(kv_cache_buffer, q, compressed_kv,
                k_pe, num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache,
                head_num, nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
        else
        {
            invokeMLARopeAppendPagedKVSplitInputHelper<__nv_bfloat16, __nv_bfloat16>(kv_cache_buffer, q, compressed_kv,
                k_pe, num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache,
                head_num, nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr, apply_q_rope);
        }
    }
}

void mergeChunkedAttentionForMLA(torch::Tensor& merged_attn, torch::Tensor const& temp_attn,
    torch::Tensor& merged_softmax_stats, torch::Tensor const& temp_softmax_stats, int64_t const num_requests,
    torch::Tensor const& cu_q_seq_lens, int64_t const max_q_seq_len, torch::Tensor const& merge_op,
    int64_t const num_heads, int64_t const head_size)
{
    TORCH_CHECK(merged_attn.numel() > 0);
    TORCH_CHECK(temp_attn.numel() > 0);
    TORCH_CHECK(merged_attn.scalar_type() == temp_attn.scalar_type());
    TORCH_CHECK(merged_attn.scalar_type() == torch::kFloat16 || merged_attn.scalar_type() == torch::kFloat32
        || merged_attn.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(temp_softmax_stats.scalar_type() == merged_softmax_stats.scalar_type());
    TORCH_CHECK(merged_softmax_stats.scalar_type() == torch::kFloat32);

    if (merged_attn.scalar_type() == torch::kFloat16)
    {
        mergeChunkedAttentionForMLAHelper<half>(merged_attn, temp_attn, merged_softmax_stats, temp_softmax_stats,
            num_requests, cu_q_seq_lens, max_q_seq_len, merge_op, num_heads, head_size);
    }
    else if (merged_attn.scalar_type() == torch::kFloat32)
    {
        mergeChunkedAttentionForMLAHelper<float>(merged_attn, temp_attn, merged_softmax_stats, temp_softmax_stats,
            num_requests, cu_q_seq_lens, max_q_seq_len, merge_op, num_heads, head_size);
    }
    else if (merged_attn.scalar_type() == torch::kBFloat16)
    {
        mergeChunkedAttentionForMLAHelper<__nv_bfloat16>(merged_attn, temp_attn, merged_softmax_stats,
            temp_softmax_stats, num_requests, cu_q_seq_lens, max_q_seq_len, merge_op, num_heads, head_size);
    }
}

void MLAPackKFromKVRope(torch::Tensor const& kv, torch::Tensor const& k_pe_roped, torch::Tensor& k,
    int64_t const head_num, int64_t const qk_nope_head_dim, int64_t const qk_rope_head_dim,
    int64_t const v_head_dim)
{
    auto input_dtype = kv.scalar_type();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kFloat32 || input_dtype == torch::kBFloat16);
    CHECK_INPUT(kv, input_dtype);
    CHECK_INPUT(k_pe_roped, input_dtype);
    CHECK_INPUT(k, input_dtype);
    TORCH_CHECK(kv.dim() == 2);
    TORCH_CHECK(k_pe_roped.dim() == 2);
    TORCH_CHECK(k.dim() == 2);
    TORCH_CHECK(kv.get_device() == k_pe_roped.get_device());
    TORCH_CHECK(kv.get_device() == k.get_device());
    TORCH_CHECK(kv.size(0) > 0);
    TORCH_CHECK(kv.size(0) == k_pe_roped.size(0));
    TORCH_CHECK(kv.size(0) == k.size(0));
    TORCH_CHECK(head_num > 0);
    TORCH_CHECK(qk_nope_head_dim > 0);
    TORCH_CHECK(qk_rope_head_dim > 0);
    TORCH_CHECK(v_head_dim > 0);
    TORCH_CHECK(kv.size(1) == head_num * (qk_nope_head_dim + v_head_dim));
    TORCH_CHECK(k_pe_roped.size(1) == qk_rope_head_dim);
    TORCH_CHECK(k.size(1) == head_num * (qk_nope_head_dim + qk_rope_head_dim));

    if (input_dtype == torch::kFloat16)
    {
        mlaPackKFromKVRopeHelper<half>(kv, k_pe_roped, k, head_num, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    }
    else if (input_dtype == torch::kFloat32)
    {
        mlaPackKFromKVRopeHelper<float>(kv, k_pe_roped, k, head_num, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    }
    else if (input_dtype == torch::kBFloat16)
    {
        mlaPackKFromKVRopeHelper<__nv_bfloat16>(
            kv, k_pe_roped, k, head_num, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    }
}

void MLAContextFp8QuantizeKVSplit(torch::Tensor const& q, torch::Tensor const& kv, torch::Tensor const& k_pe_roped,
    torch::Tensor& quant_q, torch::Tensor& quant_k, torch::Tensor& quant_v,
    torch::optional<torch::Tensor> quant_scale_qkv, torch::Tensor& bmm1_scale, torch::Tensor& bmm2_scale,
    torch::optional<torch::Tensor> quant_scale_o, torch::optional<torch::Tensor> dequant_scale_q,
    torch::optional<torch::Tensor> dequant_scale_kv, double const host_bmm1_scale, int64_t const head_num,
    int64_t const qk_nope_head_dim, int64_t const qk_rope_head_dim, int64_t const v_head_dim)
{
    auto input_dtype = q.scalar_type();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kFloat32 || input_dtype == torch::kBFloat16);
    CHECK_INPUT(q, input_dtype);
    CHECK_INPUT(kv, input_dtype);
    CHECK_INPUT(k_pe_roped, input_dtype);
    CHECK_INPUT(quant_q, c10::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(quant_k, c10::ScalarType::Float8_e4m3fn);
    CHECK_INPUT(quant_v, c10::ScalarType::Float8_e4m3fn);
    CHECK_OPTIONAL_INPUT(quant_scale_qkv, torch::kFloat32);
    CHECK_INPUT(bmm1_scale, torch::kFloat32);
    CHECK_INPUT(bmm2_scale, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(quant_scale_o, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(dequant_scale_q, torch::kFloat32);
    CHECK_OPTIONAL_INPUT(dequant_scale_kv, torch::kFloat32);
    TORCH_CHECK(q.dim() == 2);
    TORCH_CHECK(kv.dim() == 2);
    TORCH_CHECK(k_pe_roped.dim() == 2);
    TORCH_CHECK(quant_q.dim() == 2);
    TORCH_CHECK(quant_k.dim() == 2);
    TORCH_CHECK(quant_v.dim() == 2);
    TORCH_CHECK(q.get_device() == kv.get_device());
    TORCH_CHECK(q.get_device() == k_pe_roped.get_device());
    TORCH_CHECK(q.get_device() == quant_q.get_device());
    TORCH_CHECK(q.get_device() == quant_k.get_device());
    TORCH_CHECK(q.get_device() == quant_v.get_device());
    TORCH_CHECK(q.size(0) > 0);
    TORCH_CHECK(kv.size(0) > 0);
    TORCH_CHECK(q.size(0) == kv.size(0));
    TORCH_CHECK(kv.size(0) == k_pe_roped.size(0));
    TORCH_CHECK(head_num > 0);
    TORCH_CHECK(qk_nope_head_dim > 0);
    TORCH_CHECK(qk_rope_head_dim > 0);
    TORCH_CHECK(v_head_dim > 0);
    TORCH_CHECK(q.size(1) == head_num * (qk_nope_head_dim + qk_rope_head_dim));
    TORCH_CHECK(kv.size(1) == head_num * (qk_nope_head_dim + v_head_dim));
    TORCH_CHECK(k_pe_roped.size(1) == qk_rope_head_dim);
    TORCH_CHECK(quant_q.size(0) == q.size(0));
    TORCH_CHECK(quant_q.size(1) == q.size(1));
    TORCH_CHECK(quant_k.size(0) == kv.size(0));
    TORCH_CHECK(quant_k.size(1) == head_num * (qk_nope_head_dim + qk_rope_head_dim));
    TORCH_CHECK(quant_v.size(0) == kv.size(0));
    TORCH_CHECK(quant_v.size(1) == head_num * v_head_dim);
    TORCH_CHECK(bmm1_scale.numel() >= 2);
    TORCH_CHECK(bmm2_scale.numel() >= 1);

    if (input_dtype == torch::kFloat16)
    {
        mlaContextFp8QuantizeKVSplitHelper<half>(q, kv, k_pe_roped, quant_q, quant_k, quant_v, quant_scale_qkv,
            bmm1_scale, bmm2_scale, quant_scale_o, dequant_scale_q, dequant_scale_kv, host_bmm1_scale, head_num,
            qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    }
    else if (input_dtype == torch::kFloat32)
    {
        mlaContextFp8QuantizeKVSplitHelper<float>(q, kv, k_pe_roped, quant_q, quant_k, quant_v, quant_scale_qkv,
            bmm1_scale, bmm2_scale, quant_scale_o, dequant_scale_q, dequant_scale_kv, host_bmm1_scale, head_num,
            qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    }
    else if (input_dtype == torch::kBFloat16)
    {
        mlaContextFp8QuantizeKVSplitHelper<__nv_bfloat16>(q, kv, k_pe_roped, quant_q, quant_k, quant_v,
            quant_scale_qkv, bmm1_scale, bmm2_scale, quant_scale_o, dequant_scale_q, dequant_scale_kv,
            host_bmm1_scale, head_num, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    }
}
} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "load_paged_kv_cache_for_mla("
        "ScalarType out_dtype"
        ", int num_contexts"
        ", int num_ctx_cached_tokens"
        ", int max_ctx_cached_kv_len"
        ", Tensor cu_ctx_cached_kv_lens"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int lora_size"
        ", int rope_size"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("load_paged_kv_cache_for_mla", &tensorrt_llm::torch_ext::loadPagedKVCacheForMLA);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "load_chunked_kv_cache_for_mla("
        "ScalarType out_dtype"
        ", int num_contexts"
        ", int num_ctx_cached_tokens"
        ", Tensor cu_ctx_chunked_kv_lens"
        ", Tensor chunked_ld_global_offset"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int lora_size"
        ", int rope_size"
        ", int tokens_per_block"
        ", int max_seq_len"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("load_chunked_kv_cache_for_mla", &tensorrt_llm::torch_ext::loadChunkedKVCacheForMLA);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_rope_append_paged_kv_assign_q("
        "Tensor q"
        ", Tensor latent_cache"
        ", int num_contexts"
        ", Tensor cu_ctx_cached_kv_lens"
        ", Tensor cu_seq_lens"
        ", int max_input_uncached_seq_len"
        ", Tensor cos_sin_cache"
        ", int head_num"
        ", int nope_size"
        ", int rope_size"
        ", int lora_size"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ", bool apply_q_rope"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_rope_append_paged_kv_assign_q", &tensorrt_llm::torch_ext::MLARopeAppendPagedKVAssignQ);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_rope_append_paged_kv_split_input("
        "Tensor q"
        ", Tensor compressed_kv"
        ", Tensor k_pe"
        ", int num_contexts"
        ", Tensor cu_ctx_cached_kv_lens"
        ", Tensor cu_seq_lens"
        ", int max_input_uncached_seq_len"
        ", Tensor cos_sin_cache"
        ", int head_num"
        ", int nope_size"
        ", int rope_size"
        ", int lora_size"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ", bool apply_q_rope"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_rope_append_paged_kv_split_input", &tensorrt_llm::torch_ext::MLARopeAppendPagedKVSplitInput);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_pack_k_from_kv_rope("
        "Tensor kv"
        ", Tensor k_pe_roped"
        ", Tensor(a!) k"
        ", int head_num"
        ", int qk_nope_head_dim"
        ", int qk_rope_head_dim"
        ", int v_head_dim"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_pack_k_from_kv_rope", &tensorrt_llm::torch_ext::MLAPackKFromKVRope);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_context_fp8_quantize_kv_split("
        "Tensor q"
        ", Tensor kv"
        ", Tensor k_pe_roped"
        ", Tensor(a!) quant_q"
        ", Tensor(b!) quant_k"
        ", Tensor(c!) quant_v"
        ", Tensor? quant_scale_qkv"
        ", Tensor(d!) bmm1_scale"
        ", Tensor(e!) bmm2_scale"
        ", Tensor? quant_scale_o"
        ", Tensor? dequant_scale_q"
        ", Tensor? dequant_scale_kv"
        ", float host_bmm1_scale"
        ", int head_num"
        ", int qk_nope_head_dim"
        ", int qk_rope_head_dim"
        ", int v_head_dim"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_context_fp8_quantize_kv_split", &tensorrt_llm::torch_ext::MLAContextFp8QuantizeKVSplit);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "merge_chunked_attention_for_mla("
        "Tensor(a!) merged_attn"
        ", Tensor temp_attn"
        ", Tensor merged_softmax_stats"
        ", Tensor temp_softmax_stats"
        ", int num_requests"
        ", Tensor cu_q_seq_lens"
        ", int max_q_seq_len"
        ", Tensor merge_op"
        ", int num_heads"
        ", int head_size"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("merge_chunked_attention_for_mla", &tensorrt_llm::torch_ext::mergeChunkedAttentionForMLA);
}
