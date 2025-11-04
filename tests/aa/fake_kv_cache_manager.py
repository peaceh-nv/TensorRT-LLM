import itertools
from random import randbytes
from typing import Dict, List, Optional, Tuple, Union

import torch

import tensorrt_llm.bindings
from tensorrt_llm._utils import (TensorWrapper, convert_to_torch_tensor,
                                 get_size_in_bytes)
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (AttentionLayerConfig,
                                                      BufferConfig,
                                                      DiskCacheTierConfig,
                                                      GpuCacheTierConfig,
                                                      HostCacheTierConfig)
from tensorrt_llm.runtime.kv_cache_manager_v2 import \
    KVCacheManager as KVCacheManagerPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import \
    KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId, _KVCache
from tensorrt_llm.runtime.kv_cache_manager_v2._common import TokenId, TokenIdExt
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import typed_range

BufferManagerCpp = tensorrt_llm.bindings.internal.runtime.BufferManager
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
ModelConfigCpp = tensorrt_llm.bindings.ModelConfig
DataType = tensorrt_llm.bindings.DataType
KVCacheEventManagerCpp = (
    tensorrt_llm.bindings.internal.batch_manager.KVCacheEventManager)
PeftCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.PeftCacheManager
WorldConfig = tensorrt_llm.bindings.WorldConfig

_token_id_gen = itertools.count()


def next_token() -> TokenIdExt:
    token_id = next(_token_id_gen)
    if token_id % 100 == 99:
        return randbytes(32)
    else:
        return TokenId(token_id)


BlocksPerWindow = Dict[int, Tuple[
    int,
    int]]  # window_size -> (blocks_in_primary_pool, blocks_in_secondary_pool)


class Role:
    """Constants for data roles in KV cache management."""

    KEY = DataRole("key")
    VALUE = DataRole("value")
    KEY_BLOCK_QUANT = DataRole("key_block_quant")
    VALUE_BLOCK_QUANT = DataRole("value_block_quant")


if ENABLE_MULTI_DEVICE:
    pass

    from tensorrt_llm._utils import mpi_comm


def get_pp_layers(
    num_layers: int,
    mapping: Mapping,
    spec_config: Optional["DecodingBaseConfig"] = None,
    layer_mask: Optional[List[bool]] = None,
) -> Tuple[List[int], int]:
    from tensorrt_llm._torch.speculative.utils import get_num_spec_layers

    total_num_layers = num_layers
    if layer_mask is not None:
        assert sum(layer_mask) == num_layers, (
            f"The number of enabled layers in layer_mask ({sum(layer_mask)}) "
            f"must match the number of layers ({num_layers}) "
            f"in KV cache manager, but get layer_mask: {layer_mask}")
        total_num_layers = len(layer_mask)
    pp_layers = mapping.pp_layers(total_num_layers)
    if layer_mask is not None:
        pp_layers = [i for i in pp_layers if layer_mask[i]]
    if spec_config is not None:
        num_spec_layers = get_num_spec_layers(spec_config)
        total_num_layers += num_spec_layers
        if mapping.is_last_pp_rank():
            pp_layers.extend(
                range(total_num_layers - num_spec_layers, total_num_layers))
    if len(pp_layers) == 0:
        # Don't support empty KV cache for now, provide at least 1 layer
        pp_layers.append(0)
    return pp_layers, total_num_layers


def mpi_disabled() -> bool:
    """True if TLLM_DISABLE_MPI is set to "1", False otherwise."""
    return os.environ.get("TLLM_DISABLE_MPI") == "1"


def mpi_rank():
    if mpi_disabled():
        try:
            return torch.distributed.get_rank()
        except ValueError:
            # Fallback: return 0 when MPI is absent (Ray / Slurm PMIx)
            return 0
    return mpi_comm().Get_rank() if ENABLE_MULTI_DEVICE else 0


class KVCacheManager:

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config=None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfigCpp] = None,
        max_beam_width: int = 1,
        is_draft: bool = False,
        kv_connector_manager=None,
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.kv_cache_type = kv_cache_type
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.is_draft = is_draft
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {
            idx: offset
            for offset, idx in enumerate(self.pp_layers)
        }

        self.kv_connector_manager = kv_connector_manager

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_local_layers)
            ]
            self.total_num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_layers)
            ]
        else:
            assert len(num_kv_heads) == self.num_layers

            def append_to_kv_heads_per_layer(num_kv_heads_per_layer: List[int],
                                             kv_head: Optional[int]):
                if kv_head is not None:
                    num_kv_heads_per_layer.append(
                        (kv_head + tp_size - 1) // tp_size)
                else:
                    num_kv_heads_per_layer.append(0)

            self.num_kv_heads_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    kv_head = num_kv_heads[i]
                    append_to_kv_heads_per_layer(self.num_kv_heads_per_layer,
                                                 kv_head)

            self.total_num_kv_heads_per_layer = []
            for i in range(self.num_layers):
                kv_head = num_kv_heads[i]
                append_to_kv_heads_per_layer(self.total_num_kv_heads_per_layer,
                                             kv_head)

        # Determine max_attention_window_vec
        if kv_cache_config.max_attention_window is None:
            # Use max_seq_len as default max_attention_window
            self.max_attention_window_vec = [max_seq_len]
        else:
            self.max_attention_window_vec = (
                kv_cache_config.max_attention_window.copy()
            )  # Make a copy to avoid modifying original
            # Clamp all window sizes to max_seq_len before calculating the
            # number of KV cache blocks. This prevents the KV cache pool from
            # being skewed by the largest window values.
            self.max_attention_window_vec = [
                min(max_seq_len, w) for w in self.max_attention_window_vec
            ]

        # Determine if this is VSWA (Variable Sliding Window Attention)
        self.is_vswa = len(set(self.max_attention_window_vec)) > 1

        if kv_cache_type != CacheTypeCpp.SELF:
            assert (len(blocks_per_window) == 1
                    ), "Only one window size is supported for non-self KV cache"
            # rewrite the attention window size in blocks_per_window
            memory_pools = blocks_per_window[self.max_attention_window_vec[0]]
            blocks_per_window = {self.max_seq_len: memory_pools}
            logger.info(
                f"Adjusted attention window size to {self.max_seq_len} in blocks_per_window"
            )

        print(
            f"cache_bytes_per_token: {self.get_cache_bytes_per_token() * tokens_per_block}"
        )

        config = KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=None,
            cache_tiers=[
                # Magic Number for now
                GpuCacheTierConfig(quota=8000 << 20),
                HostCacheTierConfig(quota=8000 << 20),
                DiskCacheTierConfig(quota=1 << 30, path="/workspace/"),
            ],
            layers=[
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(
                            role=Role.KEY,
                            size=self.get_cache_bytes_per_token(
                                layer_idx=layer_id, data_role=Role.KEY) *
                            tokens_per_block,
                        ),
                        BufferConfig(
                            role=Role.VALUE,
                            size=self.get_cache_bytes_per_token(
                                layer_idx=layer_id, data_role=Role.VALUE) *
                            tokens_per_block,
                        ),
                    ],
                    sliding_window_size=None,
                    num_sink_tokens=None,
                ) for layer_id in typed_range(LayerId(self.num_local_layers))
            ],
        )

        print(f"Number of layers: {self.num_local_layers}")

        self.blocks_in_primary_pool = 300

        self.impl = KVCacheManagerPy(config)

        self.enable_block_reuse = kv_cache_config.enable_block_reuse

        self.kv_cache_map: dict[int, _KVCache] = {}

    def get_buffers(self, layer_idx: int) -> Optional[torch.Tensor]:
        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
        addr_value = self.impl.get_mem_pool_base_address(
            layer_offset, Role.VALUE)
        print(f"addr_key: {addr_key}, addr_value: {addr_value}")
        page_size_key = self.impl.get_page_stride(layer_offset, Role.KEY)
        page_size_value = self.impl.get_page_stride(layer_offset, Role.VALUE)
        print(
            f"page_size_k: {page_size_key}, page_size_value: {page_size_value}")

        return convert_to_torch_tensor(
            TensorWrapper(
                addr_key + page_size_key * layer_idx * self.kv_factor,
                self.dtype,
                (
                    300,
                    self.kv_factor,
                    self.tokens_per_block,
                    self.num_kv_heads_per_layer[layer_offset],
                    self.head_dim,
                ),
            ))

    def add_dummy_requests(self, request_ids: List[int], token_nums: List[int]):
        print(f"add_dummy_requests: {request_ids}, {token_nums}")
        for i, req_id in enumerate(request_ids):
            print(f"create_kv_cache: {req_id}, {token_nums[i]}")
            tokens = [next_token() for _ in range(token_nums[i])]
            kv_cache = self.impl.create_kv_cache(None, tokens)
            success = kv_cache.resume(torch.cuda.current_stream().cuda_stream)
            assert success
            kv_cache.capacity = token_nums[i]
            kv_cache.commit(tokens)
            self.kv_cache_map[req_id] = kv_cache
            for layer_idx in range(self.num_local_layers):
                print(f"layer_idx: {layer_idx}")
                page_indices_key = kv_cache.get_page_indices(
                    layer_idx, Role.KEY)
                page_indices_value = kv_cache.get_page_indices(
                    layer_idx, Role.VALUE)
                import array

                if (isinstance(page_indices_key, array.array)
                        and page_indices_key.typecode == "i"):
                    print(f"KEY page_indices: {list(page_indices_key)}")
                else:
                    print(page_indices_key)
                if (isinstance(page_indices_value, array.array)
                        and page_indices_value.typecode == "i"):
                    print(f"VALUE page_indices: {list(page_indices_value)}")
                else:
                    print(page_indices_value)

        print(
            f"get_batch_cache_indices: {self.get_batch_cache_indices([0, 1, 2, 3, 4, 5, 6])}"
        )

    def get_batch_cache_indices(
            self,
            request_ids: List[int],
            layer_id: int = 0,
            window_size: Optional[int] = None) -> List[List[int]]:
        return list([
            i // self.kv_factor
            for i in self.kv_cache_map[req_id].get_page_indices(
                layer_id, Role.KEY)
        ] for req_id in request_ids)

    def get_cache_bytes_per_token(self,
                                  layer_idx: Optional[int] = None,
                                  data_role: Optional[DataRole] = None):
        kv_factor = 1 if data_role is not None else 2
        if layer_idx is None:
            cache_size_per_token = (kv_factor *
                                    sum(self.num_kv_heads_per_layer) *
                                    self.head_dim)
        else:
            cache_size_per_token = (kv_factor *
                                    self.num_kv_heads_per_layer[layer_idx] *
                                    self.head_dim)

        if self.dtype not in (
                DataType.FP8,
                DataType.HALF,
                DataType.BF16,
                DataType.FLOAT,
                DataType.NVFP4,
        ):
            raise ValueError(f"Cannot support {self.dtype} KV cache.")

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token,
                                                       self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8,
            )
        return cache_size_bytes_per_token

    def shutdown(self):
        for kv_cache in self.kv_cache_map.values():
            kv_cache.close()
        self.kv_cache_map.clear()
