# Currently, our nvfp4 kernels require that KV data and its corresponding KV block scale use the same block index, but different base address.
# As the ratio between KV data size and KV block scale size is fixed, we can simply use a pool with smaller block size and the same number of blocks for block scale.
import abc
import enum
import os
from dataclasses import dataclass
from typing import NamedTuple

from ._common import CacheTier, LayerId


class DataRole(enum.IntEnum):
    'The data role of a buffer inside one layer. The name does not really matter. You are free to convert arbitrary int into DataRole and use it.'
    KEY_DATA = 0
    VALUE_DATA = 1
    KEY_BLOCK_SCALE = 2
    VALUE_BLOCK_SCALE = 3


class CacheTierConfig(NamedTuple):
    quota: int  # in bytes

    @property
    @abc.abstractmethod
    def tier(self) -> CacheTier:
        ...

    def assert_valid(self):
        assert self.quota > 0, "Quota must be positive"


class GpuCacheTierConfig(CacheTierConfig):

    @property
    def tier(self) -> CacheTier:
        return CacheTier.GPU_MEM


class HostCacheTierConfig(CacheTierConfig):

    @property
    def tier(self) -> CacheTier:
        return CacheTier.HOST_MEM


class DiskCacheTierConfig(CacheTierConfig):
    path: str  # a folder where we will store data as files

    @property
    def tier(self) -> CacheTier:
        return CacheTier.DISK

    def assert_valid(self):
        assert os.path.isdir(
            self.path
        ), f"Disk path {self.path} does not exist or is not a directory"


@dataclass(slots=True)
class KVCacheManagerConfig:
    """
    Configuration for the KV cache manager.
    """
    tokens_per_block: int
    # if you have p-tuning tokens, include them. Only needed for multi-modal.
    vocab_size: int
    # cache tiers are sorted from warm to cold. The first one must be GPU memory.
    cache_tiers: list[CacheTierConfig]

    @dataclass(slots=True)
    class AttentionLayerConfig:
        layer_id: LayerId
        # Each page can have multiple sub-pages, e.g. separate K and V data, block quantization scales for K and/or V, etc.
        # KV cache manager will automatically group sub-pages of the same size, and redirect pages of different sizes to
        # different memory pools
        @dataclass(slots=True)
        class BufferConfig:
            role: DataRole
            size: int

        # BufferConfig.role should not duplicate
        buffers: list[BufferConfig]
        # Note that we use None to represent "no sliding window". Sink tokens are excluded.
        sliding_window_size: int | None = None
        num_sink_tokens: int = 0

        def __post_init__(self):
            assert len(set(buffer.role for buffer in self.buffers)) == len(
                self.buffers), "duplicate buffer role"

    # AttentionLayerConfig.layer_id should not duplicate
    layers: list[AttentionLayerConfig]

    @dataclass(slots=True)
    class HelixConfig:
        helix_group_size: int
        helix_gpu_rank: int
        # number of tokens in one helix shard
        helix_shard_size: int
        # must be the same for all ranks in the same helix group and different for different helix groups.
        shared_comm_port: int

    helix_config: HelixConfig | None = None

    def __post_init__(self):
        assert self.cache_tiers and self.cache_tiers[0].tier == CacheTier.GPU_MEM
        assert len(set(layer.layer_id for layer in self.layers)) == len(
            self.layers), "duplicate layer id"
