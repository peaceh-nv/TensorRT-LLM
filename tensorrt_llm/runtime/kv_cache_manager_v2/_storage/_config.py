from dataclasses import dataclass, field

from .._common import LayerId, SlidingWinSize
from .._config import DataRole, KVCacheManagerConfig
from .._life_cycle_registry import LifeCycle, LifeCycleId, LifeCycleRegistry


@dataclass
class BufferId:
    layer_id: LayerId
    role: DataRole


@dataclass
class CoalescedBuffer:
    life_cycle_id: LifeCycleId
    single_buffer_size: int  # identical for all buffers in the same coalesced buffer
    buffer_ids: list[BufferId]

    @property
    def size(self) -> int:
        return self.single_buffer_size * len(self.buffer_ids)


@dataclass
class PoolConfig:
    coalesced_size: int
    coalesced_buffers: list[CoalescedBuffer] = field(default_factory=list)

    def __post_init__(self):
        assert all(buffer.size == self.coalesced_size
                   for buffer in self.coalesced_buffers)
        assert len(self.coalesced_buffers) > 0


# A group of mirrored pools. Each pool has different page size, but they have the same number
# of slots and allocation/free is mirrored.
@dataclass
class PoolGroupConfig:
    pools: list[PoolConfig] = field(default_factory=list)


@dataclass
class StorageConfig:
    gpu_mem_quota: int
    host_mem_quota: int
    disk_quota: int
    disk_path: str
    pool_groups: list[PoolGroupConfig] = field(default_factory=list)


def create_storage_config(config: KVCacheManagerConfig) -> StorageConfig:
    # group buffers first by sliding window size, then by buffer size.
    buffer_groups = dict[SlidingWinSize, dict[int, list[BufferId]]]()
    life_cycle_registry = LifeCycleRegistry(config)
    for layer in config.layers:
        window_size = layer.sliding_win_size
        size_to_buffers = buffer_groups.setdefault(window_size,
                                                   dict[int, list[BufferId]]())
        for buffer in layer.buffers:
            size_to_buffers.setdefault(buffer.size, []).append(
                BufferId(layer.layer_id, buffer.role))
    pool_size_to_coalesced_buffers = dict[int, list[CoalescedBuffer]]()
    for window_size, size_to_buffers in buffer_groups.items():
        life_cycle_id = life_cycle_registry.get_id(
            LifeCycle(sliding_win_size=window_size))
        for size, buffer_ids in size_to_buffers.items():
            pool_size = size * len(buffer_ids)
            pool_size_to_coalesced_buffers.setdefault(pool_size, []).append(
                CoalescedBuffer(life_cycle_id=life_cycle_id,
                                single_buffer_size=size,
                                buffer_ids=buffer_ids))
    # create _StorageConfig from pool_size_to_coalesced_buffers. Group coalesced buffers by the coalesced size.
    storage_config = StorageConfig(gpu_mem_quota=config.gpu_mem_quota,
                                   host_mem_quota=config.host_mem_quota,
                                   disk_quota=config.disk_quota,
                                   disk_path=config.disk_path,
                                   pool_groups=[])
    for pool_size, coalesced_buffers in pool_size_to_coalesced_buffers.items():
        pool_group = PoolGroupConfig(pools=[])
        pool_group.pools.append(
            PoolConfig(coalesced_size=pool_size,
                       coalesced_buffers=coalesced_buffers))
        storage_config.pool_groups.append(pool_group)
    return storage_config
