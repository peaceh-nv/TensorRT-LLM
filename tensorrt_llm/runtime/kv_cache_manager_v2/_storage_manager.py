import os
import warnings
import weakref
from typing import TYPE_CHECKING, Iterator, Sequence, cast

from ._buffer_registry import BufferRegistry
from ._common import (GPU_LEVEL, NDEBUG, Address, CacheLevel, CacheTier,
                      LayerId, MemAddress)
from ._config import CacheTierConfig, DataRole, DiskCacheTierConfig
from ._copy_engine import CopyTask, batched_copy

if TYPE_CHECKING:
    from ._core._kv_cache_manager import KVCacheManager

from ._eviction_controller import (EvictablePage, PageStatus,
                                   PerLevelEvictionController)
from ._exceptions import OutOfPagesError
from ._life_cycle_registry import LifeCycleId
from ._page import Page
from ._storage import CacheLevelStorage
from ._storage._config import (BufferAttr, BufferId, SlotToPageIndices,
                               StorageConfig)
from ._storage._core import (DiskCacheLevelStorage, GpuCacheLevelStorage,
                             HostCacheLevelStorage, PoolGroupBase,
                             PoolGroupIndex, PoolIndex, Slot, SlotId)
from ._utils import (Array2D, CachedCudaEvent, HomoTuple, TemporaryCudaStream,
                     TypedIndexList, exact_div, filled_array2d, filled_list,
                     get_uniform_attribute, make_typed, partition, remove_if,
                     typed_enumerate, typed_range, unwrap_weakref)


class CacheLevelManager:
    __slots__ = ('parent', 'cache_level', 'storage', 'controller')
    parent: weakref.ref['StorageManager']
    cache_level: CacheLevel
    storage: CacheLevelStorage
    controller: PerLevelEvictionController

    @property
    def cache_tier(self) -> CacheTier:
        return self.storage.cache_tier

    def __init__(self, parent: 'StorageManager', cache_level: CacheLevel,
                 config: CacheTierConfig,
                 slot_size_lists: Sequence[Sequence[int]],
                 init_ratio: Sequence[float]):
        self.parent = weakref.ref(parent)
        self.cache_level = cache_level
        self.storage = self._create_cache_level_storage(config, slot_size_lists,
                                                        init_ratio)
        self.controller = PerLevelEvictionController(
            parent._life_cycle_grouping, cache_level)

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
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
    __slots__ = ('_parent', '_slot_to_page_indices', 'buffer_registry',
                 '_buffer_attr', '_life_cycle_grouping', '_levels',
                 '__weakref__')
    _parent: weakref.ref['KVCacheManager']
    # Callables to convert slot_id to public page_indices.
    _slot_to_page_indices: TypedIndexList[LifeCycleId, list[SlotToPageIndices]]
    buffer_registry: BufferRegistry
    # Contains the same information as _slot_to_page_indices but in a inverse way. For ref-check only.
    _buffer_attr: dict[BufferId, BufferAttr]
    _life_cycle_grouping: TypedIndexList[LifeCycleId, PoolGroupIndex]
    _levels: TypedIndexList[CacheLevel, CacheLevelManager]

    def __init__(self, parent: 'KVCacheManager', config: StorageConfig):
        assert config.cache_tiers[
            GPU_LEVEL].tier == CacheTier.GPU_MEM, "The first cache tier must be GPU memory"
        self._parent = weakref.ref(parent)
        self._slot_to_page_indices = config.slot_to_page_indices()
        self.buffer_registry = BufferRegistry()
        for cvt in sum(self._slot_to_page_indices, []):
            self.buffer_registry.register_mirrored_buffers(cvt.buffers)
        self._buffer_attr = config.buffer_attributes()
        self._life_cycle_grouping = config.life_cycle_grouping()
        slot_size_lists = [pg.slot_size_list for pg in config.pool_groups]
        # @TODO: accept an optional avg_seq_len param and consider sliding window.
        init_ratio = [
            sum(pg.slot_size_list) * len(pg.slots) for pg in config.pool_groups
        ]
        total = sum(init_ratio)
        init_ratio = [x / total for x in init_ratio]
        num_levels = CacheLevel(len(config.cache_tiers))
        self._levels = cast(TypedIndexList, [
            CacheLevelManager(self, i, config.cache_tiers[i], slot_size_lists,
                              init_ratio) for i in typed_range(num_levels)
        ])

    def get_pool_group_index(self, life_cycle: LifeCycleId) -> PoolGroupIndex:
        return self._life_cycle_grouping[life_cycle]

    def new_gpu_slots(
        self, num_slots: TypedIndexList[LifeCycleId, int]
    ) -> TypedIndexList[LifeCycleId, HomoTuple[Slot]]:
        return self.new_slots(GPU_LEVEL, num_slots)

    def new_slots(
        self, level: CacheLevel, num_slots: TypedIndexList[LifeCycleId, int]
    ) -> TypedIndexList[LifeCycleId, HomoTuple[Slot]]:
        goals = filled_array2d(self.num_cache_levels, self.num_pool_groups, 0)
        for lc in typed_range(self.num_life_cycles):
            goals[level, self.get_pool_group_index(lc)] += num_slots[lc]
        fallen_pages = make_typed(lambda: list[Page](), self.num_pool_groups)
        self._prepare_free_slots(goals, level, fallen_pages)
        ret = filled_list(HomoTuple[Slot](), self.num_life_cycles)
        storage = self._levels[level].storage
        assert all(goals[level, pg] <= storage.get_num_free_slots(pg)
                   for pg in typed_range(self.num_pool_groups))
        try:
            for life_cycle in typed_range(self.num_life_cycles):
                pg_idx = self.get_pool_group_index(life_cycle)
                ret[life_cycle] = storage.allocate_multiple(
                    pg_idx, num_slots[life_cycle])
        except Exception:
            warnings.warn("Exception not expected here. Please report a bug.")
            for lc, slots in typed_enumerate(ret):
                pg_idx = self.get_pool_group_index(lc)
                for s in slots:
                    storage.release(pg_idx, s)
            raise
        return cast(TypedIndexList[LifeCycleId, HomoTuple[Slot]], ret)

    @property
    def kv_cache_manager(self) -> 'KVCacheManager':
        return unwrap_weakref(self._parent)

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return LifeCycleId(len(self._life_cycle_grouping))

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
        return get_uniform_attribute(self._levels,
                                     lambda l: l.storage.num_pool_groups)

    @property
    def num_cache_levels(self) -> CacheLevel:
        return CacheLevel(len(self._levels))

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
        assert NDEBUG or goals.rows == self.num_cache_levels and goals.cols == self.num_pool_groups
        assert NDEBUG or all(
            all(p.cache_level < lvl_id for p in pages) for pages in
            fallen_pages), 'Fallen pages must come from upper cache levels'
        storage = self._levels[lvl_id].storage
        ctrl = self._levels[lvl_id].controller
        num_to_evict = filled_list(0, self.num_pool_groups)
        held_pages = make_typed(lambda: list[Page](), self.num_pool_groups)
        for pg_idx in typed_range(self.num_pool_groups):
            goal = goals[lvl_id, pg_idx]
            fallen = len(fallen_pages[pg_idx])
            old_free_cnt = storage.get_num_free_slots(pg_idx)
            evictable_cnt = ctrl.num_evictable_pages(pg_idx)
            num_to_evict[pg_idx] = max(
                0, min(goal + fallen - old_free_cnt, evictable_cnt))
            fallen_held_cnt = 0  # fallen held pages we must accept in the current level.
            if self.is_last_level(lvl_id):
                held_pages[pg_idx] = remove_if(
                    fallen_pages[pg_idx], lambda p: p.status == PageStatus.HELD)
                fallen_held_cnt = len(held_pages[pg_idx])
                if fallen_held_cnt > old_free_cnt + evictable_cnt:
                    # Do we need to revert the eviction we did before? Maybe not.
                    raise OutOfPagesError(
                        "Too many held pages are being evicted to the last-level cache for group {pg_idx}"
                    )
            if old_free_cnt + evictable_cnt - fallen_held_cnt < goal:
                raise OutOfPagesError(
                    "Impossible to meet the goal ({goal} free slots) for group {pg_idx}"
                )
        evicted = ctrl.evict(num_to_evict)
        accepted_pages = make_typed(lambda: list[Page](), self.num_pool_groups)
        if self.is_last_level(lvl_id):
            for pg_idx in typed_range(self.num_pool_groups):
                old_free_cnt = storage.get_num_free_slots(pg_idx)
                num_evicted = len(evicted[pg_idx])
                assert NDEBUG or all(p.status == PageStatus.DROPPABLE
                                     for p in evicted[pg_idx])
                evicted[pg_idx].clear()
                new_free_cnt = storage.get_num_free_slots(pg_idx)
                assert num_evicted + old_free_cnt == new_free_cnt
                assert len(held_pages[pg_idx]) <= new_free_cnt
                fallen_pages[pg_idx].extend(held_pages[pg_idx])
                held_pages[pg_idx].clear()
                goal = goals[lvl_id, pg_idx]
                num_accepted = min(new_free_cnt - goal,
                                   len(fallen_pages[pg_idx]))
                assert num_accepted >= 0
                accepted_pages[pg_idx] = fallen_pages[pg_idx][-num_accepted:]
                fallen_pages[pg_idx].clear()
        else:
            assert all(len(g) == 0 for g in held_pages)
            for pg_idx in typed_range(self.num_pool_groups):
                old_free_cnt = storage.get_num_free_slots(pg_idx)
                e = evicted[pg_idx]
                num_evicted = len(e)
                fallen_pages[pg_idx][:0] = cast(list[Page], e)
                e.clear()
                num_accepted = min(
                    old_free_cnt + num_evicted - goals[lvl_id, pg_idx],
                    len(fallen_pages[pg_idx]))
                assert num_accepted >= 0
                accepted_pages[pg_idx] = fallen_pages[pg_idx][-num_accepted:]
                del fallen_pages[pg_idx][-num_accepted:]
            self._prepare_free_slots(goals, CacheLevel(lvl_id + 1),
                                     fallen_pages)
        assert all(len(f) == 0 for f in fallen_pages)
        # migrate pages
        for pg_idx in typed_range(self.num_pool_groups):
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
                        update_src: bool) -> Sequence[Slot] | None:
        assert dst_level != src_level, "dst_level and src_level must be different"
        assert not any(p.scheduled_for_eviction for p in src_pages
                       ), "Source pages must not be scheduled for eviction"
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
                dst_addresses = dst_pool_group.slot_address(dst.slot_id)
                src_addresses = src_pool_group.slot_address(src.slot_id)
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
            finish_event = stream.take_finish_event()
            for src, dst in zip(src_pages, dst_slots):
                dst.ready_event = finish_event
                src.ready_event = finish_event  # compulsory for the next owner getting this slot from the pool.
                if update_src:
                    src_pool_group.release(src)
                    src.set_slot(dst)
                    src.cache_level = dst_level
            return None if update_src else dst_slots
        except Exception:
            for s in dst_slots:
                dst_pool_group.release(s)
            raise

    def _pool_group(self, cache_level: CacheLevel,
                    pool_group_index: PoolGroupIndex) -> PoolGroupBase:
        return self._levels[cache_level].storage._pool_groups[pool_group_index]

    def num_pools(self, pool_group_index: PoolGroupIndex) -> PoolIndex:
        return get_uniform_attribute(
            self._levels,
            lambda l: l.storage._pool_groups[pool_group_index].num_pools)

    def slot_size(self, pool_group_index: PoolGroupIndex) -> HomoTuple[int]:
        return get_uniform_attribute(
            self._levels, lambda l: l.storage.slot_size(pool_group_index))

    def release_slot(self, life_cycle: LifeCycleId, cache_level: CacheLevel,
                     slot: Slot) -> None:
        pg_idx = self.get_pool_group_index(life_cycle)
        self._levels[cache_level].storage.release(pg_idx, slot)

    def schedule_for_eviction(self, page: EvictablePage) -> None:
        self._levels[page.cache_level].controller.schedule_for_eviction(page)

    def exclude_from_eviction(self, page: EvictablePage) -> None:
        assert page.node_ref is not None
        self._levels[page.cache_level].controller.remove(page.node_ref)

    def get_mem_pool_base_address(self, layer_id: LayerId,
                                  data_role: DataRole) -> MemAddress:
        storage = self._levels[GPU_LEVEL].storage
        attr = self.get_buffer_attr(layer_id, data_role)
        pool_group_index = self.get_pool_group_index(attr.life_cycle_id)
        return cast(
            MemAddress,
            storage.slot_address(pool_group_index, attr.pool_index, SlotId(0)))

    def get_page_indices(self, layer_id: LayerId, data_role: DataRole,
                         pages: Iterator[Page | None]) -> Iterator[int | None]:
        attr = self.get_buffer_attr(layer_id, data_role)
        pg_idx = self.get_pool_group_index(attr.life_cycle_id)
        pool_idx = attr.pool_index
        pool = self._levels[GPU_LEVEL].storage._pool(pg_idx, pool_idx)
        indice_offset = exact_div(attr.offset, attr.size)
        base = cast(int, pool.slot_address(SlotId(0)))
        assert NDEBUG or base == self.get_mem_pool_base_address(
            layer_id, data_role)
        for page in pages:
            if page is None:
                yield None
            else:
                offset = cast(int, pool.slot_address(page.slot_id)) - base
                yield exact_div(offset, attr.size) + indice_offset

    def get_buffer_attr(self, layer_id: LayerId,
                        data_role: DataRole) -> BufferAttr:
        return self._buffer_attr[BufferId(layer_id, data_role)]

    def slot_address(self, level: CacheLevel, pg_idx: PoolGroupIndex,
                     slot_id: SlotId, pool_idx: PoolIndex) -> Address:
        return self._levels[level].storage.slot_address(pg_idx, pool_idx,
                                                        slot_id)

    def get_page_indices_for_slot(self, life_cycle: LifeCycleId,
                                  slot_id: SlotId) -> Iterator[int]:
        converters = self._slot_to_page_indices[life_cycle]
        return (cvt(slot_id) for cvt in converters)

    def get_buffers(
        self, life_cycle: LifeCycleId
    ) -> Iterator[TypedIndexList[PoolIndex, BufferId]]:
        converters = self._slot_to_page_indices[life_cycle]
        return (cvt.buffers for cvt in converters)
