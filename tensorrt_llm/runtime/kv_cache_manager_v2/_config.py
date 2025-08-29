# Currently, our nvfp4 kernels require that KV data and its corresponding KV block scale use the same block index, but different base address.
# As the ratio between KV data size and KV block scale size is fixed, we can simply use a pool with smaller block size and the same number of blocks for block scale.
import enum
from dataclasses import dataclass
from typing import Sequence

from ._common import LayerId


class DataRole(enum.IntEnum):
    KEY_DATA = 0
    VALUE_DATA = 1
    KEY_BLOCK_SCALE = 2
    VALUE_BLOCK_SCALE = 3


@dataclass
class KVCacheManagerConfig:
    """
    Configuration for the KV cache manager.
    """
    tokens_per_block: int
    # if you have p-tuning tokens, include them. Only needed for multi-modal.
    vocab_size: int
    gpu_mem_quota: int
    host_mem_quota: int
    disk_quota: int
    disk_path: str  # a folder where we will store data as files

    @dataclass
    class AttentionLayerConfig:
        layer_id: LayerId
        # Each page can have multiple sub-pages, e.g. separate K and V data, block quantization scales for K and/or V, etc.
        # KV cache manager will automatically group sub-pages of the same size, and redirect pages of different sizes to
        # different memory pools
        @dataclass
        class BufferConfig:
            role: DataRole
            size: int

        buffers: Sequence[
            BufferConfig]  # BufferConfig.role should not duplicate
        sliding_win_size: int | None = None
        num_sink_tokens: int | None = None

        def __post_init__(self):
            assert len(set(buffer.role for buffer in self.buffers)) == len(
                self.buffers), "duplicate buffer role"

    layers: Sequence[
        AttentionLayerConfig]  # LayerConfig.layer_id should not duplicate

    @dataclass
    class HelixConfig:
        helix_group_size: int
        helix_gpu_rank: int
        helix_shard_size: int  # number of tokens in one helix shard
        shared_comm_port: int  # must be the same for all ranks in the same helix group and different for different helix groups.

    helix_config: HelixConfig | None = None

    def __post_init__(self):
        assert len(set(layer.layer_id for layer in self.layers)) == len(
            self.layers), "duplicate layer id"
