from dataclasses import dataclass
from typing import NamedTuple

from .._common import LayerId
from .._config import CacheTierConfig, DataRole, KVCacheManagerConfig
from .._life_cycle_registry import LifeCycle, LifeCycleId, LifeCycleRegistry
from .._utils import HomoTuple, get_uniform_attribute, is_sorted


class BufferId(NamedTuple):
    layer_id: LayerId
    role: DataRole


@dataclass(slots=True)
class CoalescedBuffer:
    life_cycle_id: LifeCycleId
    single_buffer_size: int  # identical for all buffers in the same coalesced buffer
    buffer_ids: list[BufferId]

    @property
    def size(self) -> int:
        return self.single_buffer_size * len(self.buffer_ids)


@dataclass(slots=True)
class PageConfig:
    'A page is a group of coalesced buffers. Each coalesced buffer has multiple buffers with the same size. Multiple coalesced buffers can be in the same slotpage if they share the same life cycle and coalesced size.'
    coalesced_buffers: list[CoalescedBuffer]

    @property
    def _coalesced_size(self) -> int:
        return get_uniform_attribute(self.coalesced_buffers, lambda b: b.size)

    @property
    def slot_size(self) -> int:
        return self._coalesced_size * len(self.coalesced_buffers)

    @property
    def life_cycle_id(self) -> LifeCycleId:
        return get_uniform_attribute(self.coalesced_buffers,
                                     lambda b: b.life_cycle_id)


@dataclass(slots=True, frozen=True)
class SlotConfig:
    'A group of pages for the same life cycle.'
    pages: HomoTuple[PageConfig]

    def __post_init__(self) -> None:
        assert is_sorted(self.pages, key=lambda s: s.slot_size, reverse=True)

    @property
    def life_cycle_id(self) -> LifeCycleId:
        return get_uniform_attribute(self.pages, lambda s: s.life_cycle_id)

    @property
    def slot_size_list(self) -> HomoTuple[int]:
        return tuple(s.slot_size for s in self.pages)


@dataclass(slots=True, frozen=True)
class PoolGroupConfig:
    'A group of pools may contain slots (page groups) with different life cycles. They have identical slot size list, so we can put them in the same group of memory pools.'
    slots: HomoTuple[SlotConfig]

    @property
    def slot_size_list(self) -> HomoTuple[int]:
        return get_uniform_attribute(self.slots, lambda s: s.slot_size_list)


@dataclass(slots=True, frozen=True)
class StorageConfig:
    cache_tiers: HomoTuple[CacheTierConfig]
    pool_groups: HomoTuple[PoolGroupConfig]

    def life_cycle_grouping(self) -> HomoTuple[HomoTuple[LifeCycleId]]:
        return tuple(
            tuple(sg.life_cycle_id for sg in pg.slots)
            for pg in self.pool_groups)

    def __post_init__(self) -> None:
        all_life_cycle_ids = sum((g for g in self.life_cycle_grouping()), ())
        assert len(all_life_cycle_ids) == len(set(all_life_cycle_ids))


def create_storage_config(config: KVCacheManagerConfig) -> StorageConfig:
    # group buffers first by life cycle, then by single buffer size.
    buffer_groups = dict[LifeCycleId, dict[int, list[BufferId]]]()
    life_cycle_registry = LifeCycleRegistry(config)
    for layer in config.layers:
        life_cycle = LifeCycle.make(layer.sliding_window_size,
                                    layer.num_sink_tokens,
                                    config.tokens_per_block)
        life_cycle_id = life_cycle_registry.get_id(life_cycle)
        size_to_buffers = buffer_groups.setdefault(life_cycle_id,
                                                   dict[int, list[BufferId]]())
        for buffer in layer.buffers:
            size_to_buffers.setdefault(buffer.size, []).append(
                BufferId(layer.layer_id, buffer.role))
    # Create one slot group for each life cycle.
    # It's possible that buffers with different sizes form coalesced buffers with the same coalesced size.
    # @TODO: add test for this case.
    slot_groups: list[SlotConfig] = []
    for life_cycle_id, size_to_buffers in buffer_groups.items():
        size_to_coalesced_buffers = dict[int, list[CoalescedBuffer]]()
        for size, buffer_ids in size_to_buffers.items():
            coalesced_size = size * len(buffer_ids)
            coalesced_buffers = size_to_coalesced_buffers.setdefault(
                coalesced_size, [])
            coalesced_buffers.append(
                CoalescedBuffer(life_cycle_id=life_cycle_id,
                                single_buffer_size=size,
                                buffer_ids=buffer_ids))
        slots = [
            PageConfig(coalesced_buffers)
            for coalesced_buffers in size_to_coalesced_buffers.values()
        ]
        slots.sort(key=lambda p: p.slot_size, reverse=True)
        slot_groups.append(SlotConfig(tuple(slots)))
    # Merge slot groups with the same slot_size_list
    pool_groups_by_slot_size_list = dict[HomoTuple[int], list[SlotConfig]]()
    for slot_group in slot_groups:
        pool_groups_by_slot_size_list.setdefault(slot_group.slot_size_list,
                                                 []).append(slot_group)
    pool_groups = [
        PoolGroupConfig(tuple(slot_groups))
        for slot_groups in pool_groups_by_slot_size_list.values()
    ]
    return StorageConfig(cache_tiers=tuple(config.cache_tiers),
                         pool_groups=tuple(pool_groups))
