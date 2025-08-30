import abc
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO, override

from .. import CacheLevel, KVCacheManagerConfig
from .._common import Address, DiskAddress, MemAddress
from .._copy_engine import CopyTask, batched_copy
from .._cuda_virt_mem import PhysMem, VirtMem
from .._exceptions import LogicError, ResourceBusy
from .._life_cycle_registry import LifeCycleId
from .._utils import (CachedCudaEvent, DynamicBitset, HostMem,
                      TemporaryCudaStream, div_up, get_file_size,
                      query_total_gpu_memory, resize_file, round_down, round_up)
from ._config import StorageConfig, create_storage_config


class PhysMemPool:
    _total_num_phys_mem: int
    _phys_mem_pool: deque[PhysMem]

    def __init__(self, total_num_phys_mem: int):
        self._total_num_phys_mem = total_num_phys_mem
        self._phys_mem_pool = deque(PhysMem()
                                    for _ in range(self._total_num_phys_mem))

    def destroy(self):
        assert len(
            self._phys_mem_pool
        ) == self._total_num_phys_mem, "Some PhysMem are still in use"
        for phys_mem in self._phys_mem_pool:
            phys_mem.destroy()
        self._phys_mem_pool.clear()
        self._total_num_phys_mem = 0

    def push(self, phys_mem: PhysMem):
        self._phys_mem_pool.append(phys_mem)

    def pop(self) -> PhysMem:
        return self._phys_mem_pool.popleft()

    # raise exception if failed
    def resize(self, new_total_num_phys_mem: int) -> None:
        increase = new_total_num_phys_mem - self._total_num_phys_mem
        if increase > 0:
            # the [] is compulsory to avoid partial extension in case of exception
            self._phys_mem_pool.extend([PhysMem() for _ in range(increase)])
        elif increase < 0:
            delta = -increase
            if len(self._phys_mem_pool) < delta:
                raise ResourceBusy(
                    f"Not enough unused physical memory to shrink. Current {len(self._phys_mem_pool)} physical memory, need {delta}"
                )
            for _ in range(delta):
                self._phys_mem_pool.pop().destroy()
        self._total_num_phys_mem = new_total_num_phys_mem

    @property
    def num_free_phys_mem(self) -> int:
        return len(self._phys_mem_pool)

    @property
    def total_num_phys_mem(self) -> int:
        return self._total_num_phys_mem


class SlotPoolBase(abc.ABC):
    _slot_size: int
    _num_slots: int

    @property
    def slot_size(self) -> int:
        return self._slot_size

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def num_bytes(self) -> int:
        return self.slot_size * self.num_slots

    def __init__(self, slot_size: int, num_slots: int):
        self._slot_size = slot_size
        self._num_slots = num_slots

    @abc.abstractmethod
    def destroy(self):
        pass

    @abc.abstractmethod
    def resize(self, new_num_slots: int) -> None:
        pass

    @abc.abstractmethod
    def slot_address(self, slot: int) -> Address:
        pass


class GpuSlotPool(SlotPoolBase):
    _shared_phys_mem_pool: PhysMemPool
    _vm: VirtMem

    def __init__(self, slot_size: int, num_slots: int, vm_size: int,
                 shared_phys_mem_pool: PhysMemPool):
        super().__init__(slot_size, num_slots)
        self._shared_phys_mem_pool = shared_phys_mem_pool
        self._vm = VirtMem(vm_size)
        while self._vm.mapped_size() < self.num_bytes:
            self._vm.push(self._shared_phys_mem_pool.pop())

    @override
    def destroy(self):
        self._vm.destroy()
        self._shared_phys_mem_pool.destroy()

    @override
    def resize(self, new_num_slots: int) -> None:
        new_mapped_size = round_up(new_num_slots * self.slot_size, PhysMem.SIZE)
        if new_mapped_size > self._vm.mapped_size(
        ) and self._shared_phys_mem_pool.num_free_phys_mem * PhysMem.SIZE < new_mapped_size - self._vm.mapped_size(
        ):
            raise LogicError(
                "Physical memory preparation is done but not enough physical memory is available"
            )
        while self._vm.mapped_size() < new_mapped_size:
            self._vm.push(self._shared_phys_mem_pool.pop())
        while self._vm.mapped_size() - new_mapped_size > PhysMem.SIZE:
            self._shared_phys_mem_pool.push(self._vm.pop())
        self._num_slots = new_num_slots

    @override
    def slot_address(self, slot: int) -> MemAddress:
        return int(self._vm.address) + self.slot_size * slot


class HostSlotPool(SlotPoolBase):
    _host_mem: HostMem

    def __init__(self, slot_size: int, num_slots: int):
        super().__init__(slot_size, num_slots)
        self._host_mem = HostMem(self.num_bytes)

    @override
    def destroy(self):
        self._host_mem.destroy()

    @override
    def resize(self, new_num_slots: int) -> None:
        self._host_mem.resize(new_num_slots * self.slot_size)
        self._num_slots = new_num_slots

    @override
    def slot_address(self, slot: int) -> MemAddress:
        return self._host_mem._address + self.slot_size * slot


class DiskSlotPool(SlotPoolBase):
    filename: str
    file: BinaryIO

    def __init__(self, filename: str, slot_size: int, num_slots: int):
        super().__init__(slot_size, num_slots)
        self.filename = filename
        self.file = open(filename, "wb+")
        self.resize(num_slots)

    @override
    def destroy(self):
        self.file.close()
        os.remove(self.filename)

    @property
    def file_size(self) -> int:
        return get_file_size(self.file)

    @override
    def resize(self, new_num_slots: int) -> None:
        resize_file(self.file, new_num_slots * self.slot_size)
        self._num_slots = new_num_slots

    @override
    def slot_address(self, slot: int) -> DiskAddress:
        assert slot < self.num_slots
        return DiskAddress(self.file, slot * self.slot_size)


SlotId = int


class SlotAllocator:

    @dataclass(slots=True)
    class Slot:
        # ready_event indicates whether the slot is ready for use.
        #  For newly allocated BlockData, it indicates finish of the last usage by the previous owner of the slot (who returned the slot to the pool).
        #  When used in free(), it indicates finish of usage by the current owner of the slot.
        id: SlotId
        ready_event: CachedCudaEvent

    _capacity: int
    _num_active_slots: int  # active slots are either in use or recycled.
    _recycled_slots: deque[
        Slot]  # only store recycled slots to avoid excessive memory usage on program start
    _num_ready_recycled_slots: int  # number of recycled slots that are ready to be used immediately (no need for sync or wait in stream), i.e. their ready events are triggered.
    _occupied_mask: DynamicBitset

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._num_active_slots = 0
        self._recycled_slots = deque[self.Slot]()
        self._num_ready_recycled_slots = 0
        self._occupied_mask = DynamicBitset(capacity)

    def __del__(self):
        assert self._occupied_mask.num_set_bits == 0, "some slots are still in use"
        assert len(self._recycled_slots
                   ) == self._num_active_slots, "some slots are not free"

    @property
    def num_free_slots(self) -> int:
        return len(
            self._recycled_slots) + self._capacity - self._num_active_slots

    @property
    def num_occupied_slots(self) -> int:
        return self._occupied_mask.num_set_bits

    def allocate(self) -> Slot | None:
        if self.num_free_slots == 0:
            return None
        self._scrub_events()
        # prefererence: ready recycled slots > new slots > recycled slots that are not ready
        if self._num_ready_recycled_slots > 0:
            slot = self._recycled_slots.popleft()
            self._num_ready_recycled_slots -= 1
            assert slot.ready_event is CachedCudaEvent.NULL
        elif self._num_active_slots < self.num_slots:
            slot = self.Slot(self._num_active_slots, CachedCudaEvent.NULL)
            self._num_active_slots += 1
        else:
            slot = self._recycled_slots.popleft()
        self._occupied_mask.set(slot.id)
        return slot

    # The reason why we don't use allocate() multiple times is that if what user need is all or none, and when we don't have enough free slots, we will free these newly allocated slots by appending them to the back of the recycled slot queue, which may impact perf.
    def allocate_multiple(self, num_slots: int) -> list[Slot] | None:
        if self.num_free_slots < num_slots:
            return None
        slots: list[SlotAllocator.Slot] = [
            self.allocate() for _ in range(num_slots)
        ]  # type: ignore[assignment]
        assert not any(s is None for s in slots)
        return slots

    def free(self, slot: Slot):
        if not self._occupied_mask.get(slot.id):
            raise LogicError(f"Slot {slot.id} is not occupied")
        self._recycled_slots.append(slot)
        self._occupied_mask.clear(slot.id)
        self._scrub_events()
        assert self._check()

    @property
    def num_slots(self) -> int:
        return self._capacity

    def resize(self, new_num_slots: int) -> None:
        assert self._check()
        if new_num_slots < self.num_slots and self._occupied_mask.any_set(
                new_num_slots, self.num_slots):
            raise ResourceBusy("resize cannot remove occupied slots")
        self._occupied_mask.resize(new_num_slots)
        self._capacity = new_num_slots
        self._num_active_slots = min(self._num_active_slots, self._capacity)
        if new_num_slots < self._capacity:
            new_recycled_slots = deque[self.Slot]()
            for slot in self._recycled_slots:
                if slot.id >= new_num_slots:
                    slot.ready_event.synchronize()
                    slot.ready_event = CachedCudaEvent.NULL
                else:
                    new_recycled_slots.append(slot)
            self._recycled_slots = new_recycled_slots
            self._num_ready_recycled_slots = 0
            self._scrub_events()
        assert self._check()

    def _scrub_events(self) -> None:
        assert self._num_ready_recycled_slots <= len(self._recycled_slots)
        for i in range(self._num_ready_recycled_slots,
                       len(self._recycled_slots)):
            slot = self._recycled_slots[i]
            if slot.ready_event.query_complete():
                slot.ready_event = CachedCudaEvent.NULL
                self._num_ready_recycled_slots += 1
            else:
                break

    def _check(self) -> bool:
        return self.num_free_slots + self.num_occupied_slots == self.num_slots and self._num_active_slots == self._occupied_mask.num_set_bits + len(
            self._recycled_slots)


class PoolGroup:
    __slots__ = ('_slot_allocator', '_pools')

    _slot_allocator: SlotAllocator
    _pools: list[SlotPoolBase]

    def __init__(self, num_slots: int):
        self._slot_allocator = SlotAllocator(num_slots)

    def destroy(self):
        for pool in self._pools:
            pool.destroy()
        self._pools.clear()
        self._slot_allocator.resize(0)

    @property
    def num_slots(self) -> int:
        num_slots = self._slot_allocator._capacity
        for pool in self._pools:
            assert pool.num_slots == num_slots, f"Pool num_slots {pool.num_slots} does not match expected {num_slots}"
        return num_slots

    @property
    def num_free_slots(self) -> int:
        return self._slot_allocator.num_free_slots

    @property
    def num_bytes(self) -> int:
        return sum(pool.num_bytes for pool in self._pools)

    def resize(self, new_num_slots: int) -> None:
        self._slot_allocator.resize(new_num_slots)
        for pool in self._pools:
            pool.resize(new_num_slots)

    def allocate(self) -> SlotAllocator.Slot | None:
        return self._slot_allocator.allocate()

    def allocate_multiple(self,
                          num_slots: int) -> list[SlotAllocator.Slot] | None:
        return self._slot_allocator.allocate_multiple(num_slots)

    def free(self, slot: SlotAllocator.Slot):
        self._slot_allocator.free(slot)

    def slot_address(self, slot: SlotAllocator.Slot) -> tuple[Address, ...]:
        return tuple(pool.slot_address(slot.id) for pool in self._pools)

    def slot_size(self) -> tuple[int, ...]:
        return tuple(pool.slot_size for pool in self._pools)


class GpuPoolGroup(PoolGroup):

    def __init__(self, num_slots: int, slot_size_list: Sequence[int],
                 phys_mem_pool: PhysMemPool):
        super().__init__(num_slots)
        total_gpu_memory = query_total_gpu_memory()
        max_slot_size = max(slot_size_list)
        self._pools = [
            GpuSlotPool(slot_size, num_slots,
                        int(total_gpu_memory * slot_size / max_slot_size),
                        phys_mem_pool) for slot_size in slot_size_list
        ]


class HostPoolGroup(PoolGroup):

    def __init__(self, num_slots: int, slot_size_list: Sequence[int]):
        super().__init__(num_slots)
        self._pools = [
            HostSlotPool(slot_size, num_slots) for slot_size in slot_size_list
        ]


class DiskPoolGroup(PoolGroup):

    def __init__(self, num_slots: int, slot_size_list: Sequence[int],
                 filename_template: str):
        super().__init__(num_slots)
        self._pools = [
            DiskSlotPool(filename_template.format(i), slot_size, num_slots)
            for i, slot_size in enumerate(slot_size_list)
        ]


PoolGroupIndex = int
PoolIndex = int


class CacheLevelStorage:
    _total_quota: int
    _ratio_list: tuple[float, ...]
    _pool_groups: list[PoolGroup]
    POOL_SIZE_GRANULARITY: int = 1  # derived class can override this

    def __init__(self, total_quota: int, ratio_list: Sequence[float]):
        self._total_quota = total_quota
        self._ratio_list = tuple(ratio_list)

    def destroy(self):
        for pg in self._pool_groups:
            pg.destroy()
        self._pool_groups.clear()
        self._total_quota = 0
        self._ratio_list = ()

    def allocate(self,
                 pool_group_index: PoolGroupIndex) -> SlotAllocator.Slot | None:
        return self._pool_groups[pool_group_index].allocate()

    def free(self, pool_group_index: PoolGroupIndex, slot: SlotAllocator.Slot):
        self._pool_groups[pool_group_index].free(slot)

    @property
    def total_quota(self) -> int:
        return self._total_quota

    @property
    def ratio_list(self) -> tuple[float, ...]:
        return self._ratio_list

    def num_slots(self, pool_group_index: PoolGroupIndex) -> int:
        return self._pool_groups[pool_group_index].num_slots

    @property
    def slot_count_list(self) -> tuple[int, ...]:
        """
        Return a tuple of the number of slots in each pool group.
        """
        return tuple(pg.num_slots for pg in self._pool_groups)

    def slot_size(self, pool_group_index: PoolGroupIndex) -> tuple[int, ...]:
        """
        Return a tuple of the slot sizes of each pool in the pool group.
        """
        return self._pool_groups[pool_group_index].slot_size()

    @property
    def slot_size_lists(self) -> tuple[tuple[int, ...], ...]:
        """
        Return a tuple of tuples, each containing the slot sizes for a pool group.
        """
        return tuple(
            tuple(p.slot_size for p in pg._pools) for pg in self._pool_groups)

    # Calculate how many slots will there be in each pool group with the given total_quota and ratio_list. Use _ratio_to_slot_count_list for initialization.
    def compute_slot_count_list(
            self,
            total_quota: int | None = None,
            ratio_list: Sequence[float] | None = None) -> tuple[int, ...]:
        total_quota = total_quota or self.total_quota
        ratio_list = ratio_list or self.ratio_list
        assert len(ratio_list) == len(
            self._pool_groups
        ), f"Wrong ratio_list length. Expected {len(self._pool_groups)}, got {len(ratio_list)}"
        return self._ratio_to_slot_count_list(total_quota, self.slot_size_lists,
                                              ratio_list,
                                              self.POOL_SIZE_GRANULARITY)

    def resize(self,
               new_total_quota: int | None = None,
               new_ratio_list: Sequence[float] | None = None) -> None:
        new_slot_count_list = self.compute_slot_count_list(
            new_total_quota, new_ratio_list)
        self._pre_resize(new_slot_count_list)
        for pg, new_slot_count in zip(self._pool_groups, new_slot_count_list):
            pg.resize(new_slot_count)
        self._total_quota = new_total_quota or self.total_quota
        if new_ratio_list is not None:
            self._ratio_list = tuple(new_ratio_list)

    def _pre_resize(self, new_slot_count_list: Sequence[int]):
        return

    @staticmethod
    def _ratio_to_slot_count_list(
            total_quota: int,
            slot_size_lists: Sequence[Sequence[int]],
            ratio_list: Sequence[float],
            pool_size_granularity: int = 1) -> tuple[int, ...]:
        sum_ratio = sum(ratio_list)
        ret = []
        # divide total_quota into pool groups based on init_ratio, then divide quote for each pool_group into pools based on slot_size.
        for slot_size_list, ratio in zip(slot_size_lists, ratio_list):
            pool_group_quota = total_quota * ratio / sum_ratio
            sum_slot_size = sum(slot_size_list)
            num_slots = min(
                round_down(int(pool_group_quota *
                               (slot_size /
                                sum_slot_size)), pool_size_granularity) //
                slot_size for slot_size in slot_size_list)
            assert num_slots > 0, f"slot_size_list {slot_size_list} with ratio {ratio} is too small to fit in {total_quota} bytes"
            ret.append(num_slots)
        return tuple(ret)


class GpuCacheLevelStorage(CacheLevelStorage):
    _phys_mem_pool: PhysMemPool
    POOL_SIZE_GRANULARITY: int = PhysMem.SIZE

    def __init__(self, total_quota: int,
                 slot_size_lists: Sequence[Sequence[int]],
                 init_ratio: Sequence[float]):
        assert len(slot_size_lists) == len(
            init_ratio
        ), "slot_size_lists and init_ratio must have the same length"
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio, PhysMem.SIZE)
        self._phys_mem_pool = PhysMemPool(0)
        self._pre_resize(slot_count_list)
        assert self._phys_mem_pool.total_num_phys_mem * PhysMem.SIZE <= total_quota
        self._pool_groups = [
            GpuPoolGroup(num_slots, slot_size_list, self._phys_mem_pool) for
            slot_size_list, num_slots in zip(slot_size_lists, slot_count_list)
        ]

    @override
    def destroy(self):
        self._phys_mem_pool.destroy()
        super().destroy()

    @override
    def _pre_resize(self, new_slot_count_list: Sequence[int]):
        num_required_phys_mem = sum(
            sum(
                div_up(num_slots * slot_size, PhysMem.SIZE)
                for slot_size in slot_size_list) for num_slots, slot_size_list
            in zip(new_slot_count_list, self.slot_size_lists))
        if num_required_phys_mem > self._phys_mem_pool.total_num_phys_mem:
            self._phys_mem_pool.resize(num_required_phys_mem)


class HostCacheLevelStorage(CacheLevelStorage):

    def __init__(self, total_quota: int,
                 slot_size_lists: Sequence[Sequence[int]],
                 init_ratio: Sequence[float]):
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio, 1)
        self._pool_groups = [
            HostPoolGroup(num_slots,
                          slot_size_list) for slot_size_list, num_slots in zip(
                              slot_size_lists, slot_count_list)
        ]


class DiskCacheLevelStorage(CacheLevelStorage):

    def __init__(self, total_quota: int,
                 slot_size_lists: Sequence[Sequence[int]],
                 init_ratio: Sequence[float], filename_template: str):
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio, 1)
        self._pool_groups = [
            DiskPoolGroup(num_slots, slot_size_list, filename_template) for
            slot_size_list, num_slots in zip(slot_size_lists, slot_count_list)
        ]


class CacheStorage:

    @dataclass(slots=True)
    class Slot(SlotAllocator.Slot):
        pool_group_index: PoolGroupIndex
        cache_level: CacheLevel

    # different life cycle variants may map to the same pool group
    _life_cycle_id_to_pool_group_index: dict[LifeCycleId, int]
    # sorted by cache levels, None means no storage for this cache level
    _cache_levels: dict[CacheLevel, CacheLevelStorage]

    def __init__(self, config: StorageConfig | KVCacheManagerConfig):
        if isinstance(config, KVCacheManagerConfig):
            config = create_storage_config(config)
        slot_size_lists = [[p.coalesced_size for p in pg.pools]
                           for pg in config.pool_groups]
        # accept an optional avg_seq_len param and consider sliding window.
        init_ratio = [
            sum(p.coalesced_size for p in pg.pools) for pg in config.pool_groups
        ]
        total = sum(init_ratio)
        init_ratio = [x / total for x in init_ratio]
        if config.gpu_mem_quota > 0:
            self._cache_levels[CacheLevel.GPU_MEM] = GpuCacheLevelStorage(
                config.gpu_mem_quota, slot_size_lists, init_ratio)
        if config.host_mem_quota > 0:
            self._cache_levels[CacheLevel.HOST_MEM] = HostCacheLevelStorage(
                config.host_mem_quota, slot_size_lists, init_ratio)
        if config.disk_quota > 0:
            filename_template = os.path.join(config.disk_path, 'data_{}.bin')
            self._cache_levels[CacheLevel.DISK] = DiskCacheLevelStorage(
                config.disk_quota, slot_size_lists, init_ratio,
                filename_template)
        for pg_index, pg in enumerate(config.pool_groups):
            for pool in pg.pools:
                for cb in pool.coalesced_buffers:
                    if cb.life_cycle_id in self._life_cycle_id_to_pool_group_index and self._life_cycle_id_to_pool_group_index[
                            cb.life_cycle_id] != pg_index:
                        raise LogicError(
                            f"Life cycle variant {cb.life_cycle_id} is already mapped to pool group {self._life_cycle_id_to_pool_group_index[cb.life_cycle_id]}"
                        )
                    self._life_cycle_id_to_pool_group_index[
                        cb.life_cycle_id] = pg_index

    def get_pool_group_index(self, life_cycle_id: LifeCycleId) -> int:
        return self._life_cycle_id_to_pool_group_index[life_cycle_id]

    def allocate(self, pool_group_index: PoolGroupIndex,
                 cache_level: CacheLevel) -> Slot | None:
        assert cache_level in self._cache_levels, f"Cache level {cache_level} is not supported"
        slot = self._cache_levels[cache_level]._pool_groups[
            pool_group_index].allocate()
        if slot is None:
            return None
        return self.Slot(**vars(slot),
                         pool_group_index=pool_group_index,
                         cache_level=cache_level)

    def allocate_multiple(self, pool_group_index: PoolGroupIndex,
                          cache_level: CacheLevel,
                          num_slots: int) -> list[Slot] | None:
        assert cache_level in self._cache_levels, f"Cache level {cache_level} is not supported"
        slots = self._cache_levels[cache_level]._pool_groups[
            pool_group_index].allocate_multiple(num_slots)
        if slots is None:
            return None
        return [
            self.Slot(**vars(slot),
                      pool_group_index=pool_group_index,
                      cache_level=cache_level) for slot in slots
        ]

    def allocate_multiple_by_pool_group(
            self, cache_level: CacheLevel,
            num_slots_by_pool_group: list[int]) -> list[list[Slot]] | None:
        assert cache_level in self._cache_levels, f"Cache level {cache_level} is not supported"
        assert len(
            num_slots_by_pool_group
        ) == self.num_pool_groups, f"num_slots_by_pool_group must have {self.num_pool_groups} elements"
        if any(
                self._pool_group(cache_level, pg_idx).num_free_slots < num_slots
                for pg_idx, num_slots in enumerate(num_slots_by_pool_group)):
            return None
        slots_by_group: list[list[CacheStorage.Slot]] = [
            self.allocate_multiple(pg_idx, cache_level, num_slots)
            for pg_idx, num_slots in enumerate(num_slots_by_pool_group)
        ]  # type: ignore[assignment]
        assert all(
            slots is not None for slots in slots_by_group
        ), "we have enough free slots but allocate_multiple returned None"
        return slots_by_group

    def free(self, page: Slot):
        self._cache_levels[page.cache_level]._pool_groups[
            page.pool_group_index].free(
                SlotAllocator.Slot(page.id, page.ready_event))

    def slot_address(self, slot: Slot) -> tuple[Address, ...]:
        return self._cache_levels[slot.cache_level]._pool_groups[
            slot.pool_group_index].slot_address(slot)

    def num_slots(self, cache_level: CacheLevel,
                  pool_group_index: PoolGroupIndex) -> int:
        return self._pool_group(cache_level, pool_group_index).num_slots

    def slot_size(self, pool_group_index: PoolGroupIndex) -> tuple[int, ...]:
        assert self._cache_levels
        storage_iterator = iter(self._cache_levels.values())
        result = next(storage_iterator).slot_size(pool_group_index)
        assert all(
            s.slot_size(pool_group_index) == result for s in storage_iterator)
        return result

    def resize(self, cache_level: CacheLevel, new_quota: int) -> bool:
        ...

    def update_ratio(self, cache_level: CacheLevel,
                     ratio_list: Sequence[float]) -> bool:
        ...

    @property
    def num_pool_groups(self) -> int:
        iterator = iter(self._cache_levels.values())
        ret = len(next(iterator)._pool_groups)
        assert all(
            len(s._pool_groups) == ret for s in iterator
        ), "All cache levels must have the same number of pool groups"
        return ret

    def num_pools_in_group(self, pool_group_index: PoolGroupIndex) -> int:
        iterator = iter(self._cache_levels.values())
        ret = len(next(iterator)._pool_groups[pool_group_index]._pools)
        assert all(
            len(s._pool_groups[pool_group_index]._pools) == ret
            for s in iterator
        ), "All cache levels must have the same number of pools in each pool group"
        return ret

    def _pool_group(self, cache_level: CacheLevel,
                    pool_group_index: PoolGroupIndex) -> PoolGroup:
        assert cache_level in self._cache_levels, f"Cache level {cache_level} is not supported"
        return self._cache_levels[cache_level]._pool_groups[pool_group_index]


class Page:
    __slots__ = ('_cache_storage', '_slot')
    _cache_storage: CacheStorage
    _slot: CacheStorage.Slot | None

    # TODO: In the future, we may want a dedicated disk_slot to keep disk data even if data is ready in gpu/host memory. This helps reduce disk write, at cost of more disk space usage.

    def __init__(self, cache_storage: CacheStorage, slot: CacheStorage.Slot):
        self._cache_storage = cache_storage
        self._slot = slot

    def drop(self):
        if not self.valid:
            return
        self._cache_storage.free(self._slot)  # type: ignore[arg-type]
        self._slot = None

    def __del__(self):
        self.drop()

    @property
    def slot(self) -> CacheStorage.Slot:
        assert self._slot is not None
        return self._slot

    @slot.setter
    def slot(self, slot: CacheStorage.Slot):
        self.drop()
        self._slot = slot

    @property
    def valid(self) -> bool:
        return self._slot is not None

    # benefits of using batched migrate:
    # 1. reduce the number of cudaStreamWaitEvent calls by deduplicating ready_events.
    # 2. give us a chance to use a kernel to do batched gpu <-> host memory copy.
    # 3. after migration, the slots can share one ready_event. So further query/sync cost can be reduced.
    @staticmethod
    def batched_migrate(page_list: Sequence['Page'],
                        dst_cache_level: CacheLevel) -> bool:
        """
        Before calling this, for each page, all usage must be finished and the finish events must be merged and set as the page's ready_event.
        """
        assert all(page.valid
                   for page in page_list), f"All pages must have valid slots"
        if len(page_list) == 0:
            return True
        cache_storage = page_list[0]._cache_storage
        assert all(p._cache_storage is cache_storage for p in page_list)
        # classify pages by source cache level and pool group index.
        page_grouping: dict[CacheLevel, dict[PoolGroupIndex, list[Page]]] = {}
        num_pages_by_group: list[int] = [0] * cache_storage.num_pool_groups
        for p in page_list:
            if p.slot.cache_level == dst_cache_level:
                continue
            page_grouping.setdefault(p.slot.cache_level,
                                     {}).setdefault(p.slot.pool_group_index,
                                                    []).append(p)
            num_pages_by_group[p.slot.pool_group_index] += 1
        assert dst_cache_level not in page_grouping
        # allocate new slots for pages in dst_cache_level. indexed by pool group index.
        all_new_slots: list[
            list[CacheStorage.
                 Slot]] | None = cache_storage.allocate_multiple_by_pool_group(
                     dst_cache_level, num_pages_by_group)
        if all_new_slots is None:
            return False
        # transfer data from src_cache_level to dst_cache_level.
        for src_lv, pg_to_pages in page_grouping.items():
            for pg_idx, pages in pg_to_pages.items():
                dst_slots = all_new_slots[pg_idx]
                assert len(dst_slots) == len(pages)
                # must use the same stream for one pool group as a page has only one ready_event for buffers in all pools.
                prior_events: set[CachedCudaEvent] = set()
                tasks_by_pool: list[list[CopyTask]] = [
                    []
                ] * cache_storage.num_pools_in_group(pg_idx)
                for i, page in enumerate(pages):
                    prior_events.update(
                        (dst_slots[i].ready_event, page.slot.ready_event))
                    dst_addresses = cache_storage.slot_address(dst_slots[i])
                    src_addresses = cache_storage.slot_address(page.slot)
                    for pool_idx in range(
                            cache_storage.num_pools_in_group(pg_idx)):
                        tasks_by_pool[pool_idx].append(
                            CopyTask(dst_addresses[pool_idx],
                                     src_addresses[pool_idx]))
                with TemporaryCudaStream(prior_events) as stream:
                    slot_sizes = cache_storage.slot_size(pg_idx)
                    for pool_idx, tasks in enumerate(tasks_by_pool):
                        batched_copy(dst_cache_level, src_lv,
                                     slot_sizes[pool_idx], tasks, stream.get())
                    finish_event = stream.finish()
                for page, dst_slot in zip(pages, dst_slots):
                    dst_slot.ready_event = finish_event
                    page.slot.ready_event = finish_event  # compulsory for the next owner getting this slot from the pool.
                    page.slot = dst_slot
        return True
