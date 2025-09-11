import os
import weakref
from typing import Sequence, cast

from tensorrt_llm.runtime.kv_cache_manager_v2._config import (
    CacheTierConfig, DiskCacheTierConfig)

from ._common import CacheLevel, CacheTier
from ._copy_engine import CopyTask, batched_copy
from ._eviction_controller import (EvictablePage, PageStatus,
                                   PerLevelEvictionController)
from ._exceptions import OutOfPagesError
from ._life_cycle_registry import LifeCycleId
from ._page import Page
from ._storage import CacheLevelStorage
from ._storage._config import BufferAttr, BufferId, StorageConfig
from ._storage._core import (DiskCacheLevelStorage, GpuCacheLevelStorage,
                             HostCacheLevelStorage, PoolGroupBase,
                             PoolGroupIndex, Slot)
from ._utils import (Array2D, CachedCudaEvent, HomoTuple, TemporaryCudaStream,
                     TypedIndexList, get_uniform_attribute, partition,
                     remove_if, typed_range)


class CacheLevelManager:
    __slots__ = ('parent', 'storage', 'controller')
    parent: weakref.ref['StorageManager']
    storage: CacheLevelStorage
    controller: PerLevelEvictionController

    @property
    def cache_tier(self) -> CacheTier:
        return self.storage.cache_tier

    def __init__(self, parent: 'StorageManager', config: CacheTierConfig,
                 slot_size_lists: Sequence[Sequence[int]],
                 init_ratio: Sequence[float]):
        self.parent = weakref.ref(parent)
        self.storage = self._create_cache_level_storage(config, slot_size_lists,
                                                        init_ratio)
        self.controller = PerLevelEvictionController(
            parent._life_cycle_grouping)

    @property
    def num_pool_groups(self) -> int:
        assert self.storage.num_pool_groups == self.controller.num_pool_groups
        return self.storage.num_pool_groups

    @staticmethod
    def _create_cache_level_storage(
            config: CacheTierConfig, slot_size_lists: Sequence[Sequence[int]],
            init_ratio: Sequence[float]) -> CacheLevelStorage:
        if config.tier == CacheTier.GPU_MEM:
            return GpuCacheLevelStorage(config.quota, slot_size_lists,
                                        init_ratio)
        elif config.tier == CacheTier.HOST_MEM:
            return HostCacheLevelStorage(config.quota, slot_size_lists,
                                         init_ratio)
        elif config.tier == CacheTier.DISK:
            assert isinstance(config, DiskCacheTierConfig)
            assert os.path.isdir(
                config.path
            ), f"Disk path {config.path} does not exist or is not a directory"
            filename_template = os.path.join(config.path, 'g{}p{}.bin')
            return DiskCacheLevelStorage(config.quota, slot_size_lists,
                                         init_ratio, filename_template)
        else:
            raise ValueError(f"Invalid cache tier: {config.tier}")


class StorageManager:
    _buffer_attr: dict[BufferId, BufferAttr]
    _life_cycle_grouping: dict[LifeCycleId, PoolGroupIndex]
    _levels: TypedIndexList[CacheLevel, CacheLevelManager]

    def __init__(self, config: StorageConfig):
        self._buffer_attr = config.buffer_attributes()
        self._life_cycle_grouping = config.life_cycle_grouping()
        slot_size_lists = [pg.slot_size_list for pg in config.pool_groups]
        # @TODO: accept an optional avg_seq_len param and consider sliding window.
        init_ratio = [
            sum(pg.slot_size_list) * len(pg.slots) for pg in config.pool_groups
        ]
        total = sum(init_ratio)
        init_ratio = [x / total for x in init_ratio]
        self._levels = cast(TypedIndexList, [
            CacheLevelManager(self, config.cache_tiers[i], slot_size_lists,
                              init_ratio)
            for i in range(len(config.cache_tiers))
        ])

    def get_pool_group_index(self, life_cycle: LifeCycleId) -> PoolGroupIndex:
        return self._life_cycle_grouping[life_cycle]

    def new_gpu_slots(self, life_cycle: LifeCycleId,
                      count: int) -> HomoTuple[Slot]:
        'Allocate new slots for a life cycle.'
        raise NotImplementedError("Not implemented")

    def new_gpu_slots_for_blocks(
            self, count: int) -> HomoTuple[TypedIndexList[LifeCycleId, Slot]]:
        'Allocate new slots for each life cycle, as new blocks require a slot for each life cycle.'
        raise NotImplementedError("Not implemented")

    def batched_migrate_to_gpu(self, pages: Sequence[Page]) -> None:
        raise NotImplementedError("Not implemented")

    @property
    def num_life_cycles(self) -> int:
        return len(self._life_cycle_grouping)

    @property
    def num_pool_groups(self) -> int:
        return get_uniform_attribute(self._levels,
                                     lambda l: l.storage.num_pool_groups)

    @property
    def num_cache_levels(self) -> int:
        return len(self._levels)

    def is_last_level(self, level: CacheLevel) -> bool:
        return level == self.num_cache_levels - 1

    @property
    def cache_tiers(self) -> HomoTuple[CacheTier]:
        return tuple(cache_level.cache_tier for cache_level in self._levels)

    def is_evictable(self,
                     page: EvictablePage,
                     level: CacheLevel | None = None) -> bool:
        'Check if a page is evictable. If level is specified, check if the page will be evictable after migrating to the given level.'
        status = page.status
        level = page.cache_level if level is None else level
        # droppable pages that are not committed should be dropped immediately.
        # held pages in last level cache can't be evicted.
        return (status == PageStatus.DROPPABLE and page.is_committed()) or (
            status == PageStatus.HELD and level < self.num_cache_levels - 1)

    def _prepare_free_slots(
            self, goals: Array2D[CacheLevel, PoolGroupIndex,
                                 int], lvl_id: CacheLevel,
            fallen_pages: TypedIndexList[PoolGroupIndex, list[Page]]) -> None:
        assert goals.rows == self.num_cache_levels and goals.cols == self.num_pool_groups
        assert all(
            all(p.cache_level < lvl_id for p in pages) for pages in
            fallen_pages), 'Fallen pages must come from upper cache levels'
        storage = self._levels[lvl_id].storage
        ctrl = self._levels[lvl_id].controller
        num_to_evict = cast(TypedIndexList[PoolGroupIndex, int],
                            [0] * self.num_pool_groups)
        held_pages = cast(TypedIndexList[PoolGroupIndex, list[Page]],
                          [[]] * self.num_pool_groups)
        for pg_idx in typed_range(PoolGroupIndex(self.num_pool_groups)):
            goal = goals[lvl_id, pg_idx]
            fallen = len(fallen_pages[pg_idx])
            free = storage.get_num_free_slots(pg_idx)
            evictable_cnt = ctrl.num_evictable_pages(pg_idx)
            num_to_evict[pg_idx] = max(0,
                                       min(goal + fallen - free, evictable_cnt))
            if self.is_last_level(lvl_id):
                held_pages[pg_idx] = remove_if(
                    fallen_pages[pg_idx], lambda p: p.status == PageStatus.HELD)
                held_cnt = len(held_pages[pg_idx])
                if held_cnt > free + evictable_cnt:
                    # need to revert the eviction we did before.
                    raise OutOfPagesError(
                        "Too many pages are held in the last-level cache for group {pg_idx}"
                    )
        evicted = cast(TypedIndexList[PoolGroupIndex, list[Page]],
                       ctrl.evict(num_to_evict))
        accepted_pages = cast(TypedIndexList[PoolGroupIndex, list[Page]],
                              [[]] * self.num_pool_groups)
        if self.is_last_level(lvl_id):
            for pg_idx in typed_range(PoolGroupIndex(self.num_pool_groups)):
                prev_free_cnt = storage.get_num_free_slots(pg_idx)
                num_evicted = len(evicted[pg_idx])
                assert all(p.status == PageStatus.DROPPABLE
                           for p in evicted[pg_idx])
                evicted[pg_idx].clear()
                free = storage.get_num_free_slots(pg_idx)
                assert num_evicted + prev_free_cnt == free
                assert len(held_pages[pg_idx]) <= free
                fallen_pages[pg_idx].extend(held_pages[pg_idx])
                held_pages[pg_idx].clear()
                accepted_pages[pg_idx] = fallen_pages[pg_idx][-free:]
                fallen_pages[pg_idx].clear()
        else:
            assert all(len(g) == 0 for g in held_pages)
            for pg_idx in typed_range(PoolGroupIndex(self.num_pool_groups)):
                free = storage.get_num_free_slots(pg_idx)
                e = evicted[pg_idx]
                num_evicted = len(e)
                fallen_pages[pg_idx][:0] = e
                e.clear()
                num_accepted = min(free + num_evicted,
                                   len(fallen_pages[pg_idx]))
                accepted_pages[pg_idx] = fallen_pages[pg_idx][-num_accepted:]
                del fallen_pages[pg_idx][-num_accepted:]
            self._prepare_free_slots(goals, CacheLevel(lvl_id + 1),
                                     fallen_pages)
        assert all(len(f) == 0 for f in fallen_pages)
        # migrate pages
        for pg_idx in typed_range(PoolGroupIndex(self.num_pool_groups)):
            partitioned = partition(
                accepted_pages[pg_idx], lambda p:
                (p.cache_level, self.get_pool_group_index(p.life_cycle)))
            accepted_pages[pg_idx].clear()
            for (src_lvl, pg_idx), pages in partitioned.items():
                self.batched_migrate(pg_idx,
                                     lvl_id,
                                     src_lvl,
                                     pages,
                                     update_src=True)
                for p in pages:
                    self._levels[src_lvl].controller.schedule_for_eviction(p)
        return

    def batched_migrate(self, pool_group_index: PoolGroupIndex,
                        dst_level: CacheLevel, src_level: CacheLevel,
                        src_pages: Sequence[Page],
                        update_src: bool) -> Sequence[Slot]:
        assert dst_level != src_level, "dst_level and src_level must be different"
        num_slots = len(src_pages)
        num_pools = self.num_pools(pool_group_index)
        src_pool_group = self._pool_group(src_level, pool_group_index)
        dst_pool_group = self._pool_group(dst_level, pool_group_index)
        if dst_pool_group.num_free_slots < num_slots:
            raise OutOfPagesError("Not enough free slots")
        dst_slots = dst_pool_group.allocate_multiple(num_slots)
        try:
            assert len(dst_slots) == num_slots
            prior_events: set[CachedCudaEvent] = set()
            tasks_per_pool: list[list[CopyTask]] = [[]] * num_pools
            for src, dst in zip(src_pages, dst_slots):
                prior_events.update((dst.ready_event, src.ready_event))
                dst_addresses = dst_pool_group.slot_address(dst)
                src_addresses = src_pool_group.slot_address(src)
                for pool_idx in range(num_pools):
                    tasks_per_pool[pool_idx].append(
                        CopyTask(dst_addresses[pool_idx],
                                 src_addresses[pool_idx]))
            dst_tier = self._levels[dst_level].cache_tier
            src_tier = self._levels[src_level].cache_tier
            with TemporaryCudaStream(prior_events) as stream:
                slot_sizes = self.slot_size(pool_group_index)
                for pool_idx, tasks in enumerate(tasks_per_pool):
                    batched_copy(dst_tier, src_tier, slot_sizes[pool_idx],
                                 tasks, stream.get())
                finish_event = stream.finish()
            for src, dst in zip(src_pages, dst_slots):
                dst.ready_event = finish_event
                src.ready_event = finish_event  # compulsory for the next owner getting this slot from the pool.
                if update_src:
                    src_pool_group.free(src)
                    src.set_slot(dst)
                    src.cache_level = dst_level
            return dst_slots
        except Exception:
            for s in dst_slots:
                dst_pool_group.free(s)
            raise

    def _pool_group(self, cache_level: CacheLevel,
                    pool_group_index: PoolGroupIndex) -> PoolGroupBase:
        return self._levels[cache_level].storage._pool_groups[pool_group_index]

    def num_pools(self, pool_group_index: PoolGroupIndex) -> int:
        return get_uniform_attribute(
            self._levels,
            lambda l: l.storage._pool_groups[pool_group_index].num_pools)

    def slot_size(self, pool_group_index: PoolGroupIndex) -> HomoTuple[int]:
        return get_uniform_attribute(
            self._levels, lambda l: l.storage.slot_size(pool_group_index))

    def free_slot(self, life_cycle: LifeCycleId, cache_level: CacheLevel,
                  slot: Slot) -> None:
        pg_idx = self.get_pool_group_index(life_cycle)
        self._levels[cache_level].storage.free(pg_idx, slot)

    def schedule_for_eviction(self, page: EvictablePage) -> None:
        self._levels[page.cache_level].controller.schedule_for_eviction(page)

    def exclude_from_eviction(self, page: EvictablePage) -> None:
        assert page.node_ref is not None
        self._levels[page.cache_level].controller.remove(page.node_ref)
