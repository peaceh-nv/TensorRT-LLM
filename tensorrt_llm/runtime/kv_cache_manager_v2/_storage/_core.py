import abc
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO, ClassVar, NewType, final, override

from .._common import Address, CacheLevel, CacheTier, DiskAddress, MemAddress
from .._config import CacheTierConfig, DiskCacheTierConfig, KVCacheManagerConfig
from .._cuda_virt_mem import PhysMem, PooledPhysMemAllocator, VirtMem
from .._exceptions import LogicError, OutOfPagesError, ResourceBusyError
from .._life_cycle_registry import LifeCycleId
from .._utils import (CachedCudaEvent, DynamicBitset, HomoTuple, HostMem,
                      div_up, get_file_size, get_uniform_attribute,
                      query_total_gpu_memory, remove_if, resize_file,
                      round_down, unwrap_optional)
from ._config import StorageConfig, create_storage_config


class SlotPoolBase(abc.ABC):
    _slot_size: int

    @property
    def slot_size(self) -> int:
        return self._slot_size

    @property
    @abc.abstractmethod
    def num_slots(self) -> int:
        ...

    @property
    def num_bytes(self) -> int:
        return self.slot_size * self.num_slots

    def __init__(self, slot_size: int):
        self._slot_size = slot_size

    @abc.abstractmethod
    def destroy(self):
        pass

    @abc.abstractmethod
    def resize(self, new_num_slots: int) -> None:
        pass

    @abc.abstractmethod
    def slot_address(self, slot: int) -> Address:
        pass


@final
class GpuSlotPool(SlotPoolBase):
    __slots__ = ('_vm', )
    _vm: VirtMem

    def __init__(self, slot_size: int, vm_size: int,
                 shared_phys_mem_pool: PooledPhysMemAllocator, num_slots: int):
        super().__init__(slot_size)
        assert vm_size % PhysMem.SIZE == 0
        self._vm = VirtMem(vm_size, shared_phys_mem_pool)
        self.resize(num_slots)

    @override
    def destroy(self):
        self._vm.destroy()

    @override
    def resize(self, new_num_slots: int) -> None:
        new_num_phys_mem = self._compute_num_phys_mem(self.slot_size,
                                                      new_num_slots)
        self._vm.realloc(PhysMem.SIZE * new_num_phys_mem)

    def extend_by_one_phys_mem(self) -> int:
        self._vm.extend(1)
        return self.num_slots

    @override
    def slot_address(self, slot: int) -> MemAddress:
        return int(self._vm.address) + self.slot_size * slot

    @property
    @override
    def num_slots(self) -> int:
        return self._compute_num_slots(self.slot_size, self._vm.num_phys_mem)

    @staticmethod
    def _compute_num_phys_mem(slot_size: int, num_slots: int) -> int:
        return div_up(num_slots * slot_size, PhysMem.SIZE)

    @staticmethod
    def _compute_num_slots(slot_size: int, num_phys_mem: int) -> int:
        return num_phys_mem * PhysMem.SIZE // slot_size


class HostSlotPool(SlotPoolBase):
    __slots__ = ('_host_mem', )
    _host_mem: HostMem

    def __init__(self, slot_size: int, num_slots: int):
        super().__init__(slot_size)
        self._host_mem = HostMem(0)
        self.resize(num_slots)

    @override
    def destroy(self):
        self._host_mem.destroy()

    @override
    def resize(self, new_num_slots: int) -> None:
        self._host_mem.resize(new_num_slots * self.slot_size)

    @override
    def slot_address(self, slot: int) -> MemAddress:
        return self._host_mem._address + self.slot_size * slot

    @property
    @override
    def num_slots(self) -> int:
        return self._host_mem.size // self.slot_size


class DiskSlotPool(SlotPoolBase):
    __slots__ = ('_filename', '_file')
    filename: str
    file: BinaryIO

    def __init__(self, filename: str, slot_size: int, num_slots: int):
        super().__init__(slot_size)
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

    @override
    def slot_address(self, slot: int) -> DiskAddress:
        assert slot < self.num_slots
        return DiskAddress(self.file, slot * self.slot_size)

    @property
    @override
    def num_slots(self) -> int:
        return self.file_size // self.slot_size


SlotId = NewType('SlotId', int)


@dataclass(slots=True)
class Slot:
    # ready_event indicates whether the slot is ready for use.
    #  For newly allocated BlockData, it indicates finish of the last usage by the previous owners of the slot (who returned the slot to the pool).
    #  After data migration, it indicates finish of data migration.
    #  When passed to free(), it indicates finish of usage by the current owners of the slot.
    _slot_id: SlotId | None
    ready_event: CachedCudaEvent

    @property
    def slot_id(self) -> SlotId:
        return unwrap_optional(self._slot_id)

    def query_ready(self) -> bool:
        ret = self.ready_event.query_complete()
        if ret:
            self.ready_event = CachedCudaEvent.NULL
        return ret

    @property
    def has_valid_slot(self) -> bool:
        return self._slot_id is not None

    def set_slot(self, slot: 'Slot'):
        if self.has_valid_slot:
            raise LogicError("Slot is already set.")
        self._slot_id = slot.slot_id
        self.ready_event = slot.ready_event


class SlotAllocator:
    __slots__ = ('_capacity', '_num_active_slots', '_recycled_slots',
                 '_num_ready_recycled_slots', '_occupied_mask',
                 '_target_capacity', '_overflow_slots',
                 '_num_ready_overflow_slots')
    _capacity: int
    _num_active_slots: int  # active slots are either in use or recycled.
    _recycled_slots: deque[
        Slot]  # only store recycled slots to avoid excessive memory usage on program start
    _num_ready_recycled_slots: int  # number of recycled slots that are ready to be used immediately (no need for sync or wait in stream), i.e. their ready events are triggered.
    _occupied_mask: DynamicBitset

    # for scheduled shrinking resize
    _target_capacity: int  # _target_capacity <= _capacity. Inequal if a shrinking resize is in progress.
    _overflow_slots: list[
        Slot]  # slots that will be out-of-range after a in-progress resize. scheduled for removal.
    _num_ready_overflow_slots: int  # similar to _num_ready_recycled_slots, but for _overflow_slots.

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._num_active_slots = 0
        self._recycled_slots = deque[Slot]()
        self._num_ready_recycled_slots = 0
        self._occupied_mask = DynamicBitset(capacity)
        self._target_capacity = capacity
        self._overflow_slots = []

    def __del__(self):
        assert (self._target_capacity == self._capacity
                and not self._overflow_slots), "resize is in progress"
        assert self._occupied_mask.num_set_bits == 0, "some slots are still in use"
        assert (len(self._recycled_slots) == self._num_active_slots
                ), "some slots are not free"

    @property
    def num_free_slots(self) -> int:
        return (len(self._recycled_slots) +
                max(self._target_capacity - self._num_active_slots, 0))

    @property
    def num_occupied_slots(self) -> int:
        return self._occupied_mask.num_set_bits

    def allocate(self) -> Slot:
        if self.num_free_slots == 0:
            raise OutOfPagesError("No free slots")
        self._scrub_events()
        # prefererence: ready recycled slots > new slots > recycled slots that are not ready
        if self._num_ready_recycled_slots > 0:
            slot = self._recycled_slots.popleft()
            self._num_ready_recycled_slots -= 1
            assert slot.ready_event is CachedCudaEvent.NULL
        elif self._num_active_slots < self.num_slots:
            slot = Slot(SlotId(self._num_active_slots), CachedCudaEvent.NULL)
            self._num_active_slots += 1
        else:
            slot = self._recycled_slots.popleft()
        self._occupied_mask.set(slot.slot_id)
        return slot

    # The reason why we don't use allocate() multiple times is that if what user need is all or none, and when we don't have enough free slots, we will free these newly allocated slots by appending them to the back of the recycled slot queue, which may impact perf.
    def allocate_multiple(self, num_slots: int) -> list[Slot]:
        if self.num_free_slots < num_slots:
            raise OutOfPagesError("Not enough free slots")
        return [self.allocate() for _ in range(num_slots)]

    def free(self, slot: Slot):
        if slot.slot_id >= self._capacity or not self._occupied_mask.get(
                slot.slot_id):
            raise LogicError(f"Slot {slot.slot_id} is not occupied")
        if slot.slot_id < self._target_capacity:
            self._recycled_slots.append(slot)
        else:
            self._overflow_slots.append(slot)
            self._try_trigger_shrink()
        self._occupied_mask.clear(slot.slot_id)
        self._scrub_events()
        assert self._check()
        slot._slot_id = None
        slot.ready_event = CachedCudaEvent.NULL

    @property
    def num_slots(self) -> int:
        return self._capacity

    def resize(self, new_num_slots: int) -> None:
        assert self._check()
        if new_num_slots < self.num_slots and self._occupied_mask.any_set(
                new_num_slots, self.num_slots):
            raise ResourceBusyError("resize cannot remove occupied slots")
        self._occupied_mask.resize(new_num_slots)
        self._capacity = new_num_slots
        self._num_active_slots = min(self._num_active_slots, self._capacity)
        if new_num_slots < self._capacity:
            new_recycled_slots = deque[Slot]()
            new_num_ready_recycled_slots = 0
            for idx_recycled, slot in enumerate(self._recycled_slots):
                if slot.slot_id >= new_num_slots:
                    slot.ready_event.synchronize()
                    slot.ready_event = CachedCudaEvent.NULL
                else:
                    new_recycled_slots.append(slot)
                    if idx_recycled < self._num_ready_recycled_slots:
                        new_num_ready_recycled_slots += 1
            self._recycled_slots = new_recycled_slots
            self._num_ready_recycled_slots = new_num_ready_recycled_slots
            self._scrub_events()
        assert self._check()

    def schedule_resize(self, new_num_slots: int) -> None:
        assert self._check()
        if new_num_slots >= self.num_slots:
            self.cancel_scheduled_resize()
            self.resize(new_num_slots)
            return
        old_target_capacity = self._target_capacity
        if new_num_slots > old_target_capacity:
            self._recycled_slots.extend(
                remove_if(
                    self._overflow_slots, lambda slot: old_target_capacity <=
                    slot.slot_id < new_num_slots))
            self._num_ready_overflow_slots = 0
        if new_num_slots < old_target_capacity:
            self._overflow_slots.extend(
                remove_if(self._recycled_slots,
                          lambda slot: slot.slot_id >= new_num_slots))
            self._num_ready_recycled_slots = 0
        self._target_capacity = new_num_slots
        self._try_trigger_shrink()
        self._scrub_events()
        assert self._check()

    def cancel_scheduled_resize(self) -> None:
        assert self._check()
        self._target_capacity = self._capacity
        self._recycled_slots.extend(
            remove_if(self._overflow_slots, lambda slot: True))
        self._num_ready_overflow_slots = 0

    def shrink_in_progress(self) -> bool:
        'Indicates if a scheduled shrink is in progress.'
        assert self._target_capacity <= self._capacity
        return self._target_capacity < self._capacity

    def get_slots_blocking_shrink(self) -> HomoTuple[SlotId]:
        return tuple(
            SlotId(id) for id in range(self._target_capacity, self._capacity)
            if self._occupied_mask.get(id))

    def _try_trigger_shrink(self) -> bool:
        assert self._check()
        if self.shrink_in_progress() and self._target_capacity + len(
                self._overflow_slots) == self._capacity:
            assert len(set(s.slot_id for s in self._overflow_slots)) == len(
                self._overflow_slots)
            for slot in self._overflow_slots:
                slot.ready_event.synchronize()
                slot.ready_event = CachedCudaEvent.NULL
            self._overflow_slots.clear()
            self._num_ready_overflow_slots = 0
            self._capacity = self._target_capacity
            self._num_active_slots = min(self._num_active_slots, self._capacity)
            self._scrub_events()
            assert self._check()
            return True
        return False

    def _scrub_events(self) -> None:
        self._num_ready_recycled_slots = self._scrub_events_impl(
            self._recycled_slots, self._num_ready_recycled_slots)
        self._num_ready_overflow_slots = self._scrub_events_impl(
            self._overflow_slots, self._num_ready_overflow_slots)

    def _check(self) -> bool:
        return (self._num_active_slots <= self._capacity
                and self._target_capacity <= self._capacity and
                (self.shrink_in_progress() or len(self._overflow_slots) == 0)
                and all(self._target_capacity <= slot.slot_id < self._capacity
                        for slot in self._overflow_slots)
                and len(self._recycled_slots) + len(self._overflow_slots) +
                self.num_occupied_slots == self._num_active_slots)

    @staticmethod
    def _scrub_events_impl(slots: Sequence[Slot], num_ready: int) -> int:
        assert num_ready <= len(slots)
        for i in range(num_ready, len(slots)):
            slot = slots[i]
            if slot.ready_event.query_complete():
                slot.ready_event = CachedCudaEvent.NULL
                num_ready += 1
            else:
                break
        return num_ready


class PoolGroupBase:
    __slots__ = ('_slot_allocator', '_pools')

    _slot_allocator: SlotAllocator
    _pools: HomoTuple[SlotPoolBase]

    def __init__(self, num_slots: int):
        self._slot_allocator = SlotAllocator(num_slots)

    def destroy(self):
        for pool in self._pools:
            pool.destroy()
        self._slot_allocator.resize(0)

    @property
    def num_pools(self) -> int:
        return len(self._pools)

    @property
    def num_slots(self) -> int:
        num_slots = self._slot_allocator._capacity
        assert num_slots <= self._get_num_slots_from_pools()
        return num_slots

    @property
    def num_free_slots(self) -> int:
        return self._slot_allocator.num_free_slots

    @property
    def num_bytes(self) -> int:
        return sum(pool.num_bytes for pool in self._pools)

    def resize_slot_allocator(self, new_num_slots: int | None) -> None:
        """
        Resize the slot allocator, but not pools. If new_num_slots is None, make slot allocator match the pool sizes.
        """
        if new_num_slots is None:
            new_num_slots = self._get_num_slots_from_pools()
        self._slot_allocator.resize(new_num_slots)
        assert self._check(True)

    def resize_pools(self, new_num_slots: int | None) -> None:
        """
        Resize the pools, but not the slot allocator. If new_num_slots is None, make pool sizes match the slot allocator.
        If exception is raised, size of pools may be imbalanced. Call resize_pools() again with None or self._get_num_slots_from_pools() to fix it.
        """
        if new_num_slots is None:
            new_num_slots = self._slot_allocator.num_slots
        for pool in self._pools:
            pool.resize(new_num_slots)
        assert self._check(True)

    def allocate(self) -> Slot:
        return self._slot_allocator.allocate()

    def allocate_multiple(self, num_slots: int) -> list[Slot]:
        return self._slot_allocator.allocate_multiple(num_slots)

    def free(self, slot: Slot):
        self._slot_allocator.free(slot)

    def slot_address(self, slot: Slot) -> HomoTuple[Address]:
        return tuple(pool.slot_address(slot.slot_id) for pool in self._pools)

    @property
    def slot_size(self) -> HomoTuple[int]:
        return tuple(pool.slot_size for pool in self._pools)

    def _check(self, allow_mismatch: bool = False) -> bool:
        pool_num_slots = self._get_num_slots_from_pools()
        return (self._slot_allocator.num_slots <= pool_num_slots
                if allow_mismatch else self._slot_allocator.num_slots
                == pool_num_slots)

    def _get_num_slots_from_pools(self) -> int:
        return min(p.num_slots for p in self._pools)

    @staticmethod
    def _compute_num_phys_mem(slot_size_list: Sequence[int],
                              num_slots: int) -> HomoTuple[int]:
        return tuple(
            GpuSlotPool._compute_num_phys_mem(slot_size, num_slots)
            for slot_size in slot_size_list)


class GpuPoolGroup(PoolGroupBase):
    __slots__ = ()

    def __init__(self, num_slots: int, slot_size_list: Sequence[int],
                 shared_phys_mem_pool: PooledPhysMemAllocator):
        super().__init__(num_slots)
        total_gpu_memory = query_total_gpu_memory()
        max_slot_size = max(slot_size_list)
        self._pools = tuple(
            GpuSlotPool(slot_size,
                        int(total_gpu_memory * slot_size /
                            max_slot_size), shared_phys_mem_pool, num_slots)
            for slot_size in slot_size_list)


class HostPoolGroup(PoolGroupBase):
    __slots__ = ()

    def __init__(self, num_slots: int, slot_size_list: Sequence[int]):
        super().__init__(num_slots)
        self._pools = tuple(
            HostSlotPool(slot_size, num_slots) for slot_size in slot_size_list)


class DiskPoolGroup(PoolGroupBase):
    __slots__ = ()

    def __init__(self, num_slots: int, slot_size_list: Sequence[int],
                 filename_template: str):
        super().__init__(num_slots)
        self._pools = tuple(
            DiskSlotPool(filename_template.format(i), slot_size, num_slots)
            for i, slot_size in enumerate(slot_size_list))


PoolGroupIndex = NewType("PoolGroupIndex", int)
PoolIndex = NewType("PoolIndex", int)


class CacheLevelStorage:
    POOL_SIZE_GRANULARITY: ClassVar[int] = 1  # derived class can override this
    __slots__ = ('_total_quota', '_ratio_list', '_pool_groups')
    _total_quota: int  # fixme: remove _total_quota and _ratio_list and compute from _pool_groups
    _ratio_list: HomoTuple[float]
    _pool_groups: HomoTuple[PoolGroupBase]

    def __init__(self, total_quota: int, ratio_list: Sequence[float]):
        self._total_quota = total_quota
        self._ratio_list = tuple(ratio_list)

    @property
    @abc.abstractmethod
    def cache_tier(self) -> CacheTier:
        ...

    def destroy(self):
        for pg in self._pool_groups:
            pg.destroy()
        self._total_quota = 0
        self._ratio_list = ()

    def allocate(self, pool_group_index: PoolGroupIndex) -> Slot:
        return self._pool_groups[pool_group_index].allocate()

    def allocate_multiple(self, pool_group_index: PoolGroupIndex,
                          num_slots: int) -> list[Slot]:
        return self._pool_groups[pool_group_index].allocate_multiple(num_slots)

    def free(self, pool_group_index: PoolGroupIndex, slot: Slot):
        self._pool_groups[pool_group_index].free(slot)

    @property
    def total_quota(self) -> int:
        return self._total_quota

    @property
    def ratio_list(self) -> HomoTuple[float]:
        return self._ratio_list

    def num_slots(self, pool_group_index: PoolGroupIndex) -> int:
        return self._pool_groups[pool_group_index].num_slots

    def get_num_free_slots(self, pool_group_index: PoolGroupIndex) -> int:
        return self._pool_groups[pool_group_index].num_free_slots

    @property
    def slot_count_list(self) -> HomoTuple[int]:
        """
        The number of slots in each pool group.
        """
        return tuple(pg.num_slots for pg in self._pool_groups)

    def slot_size(self, pool_group_index: PoolGroupIndex) -> HomoTuple[int]:
        """
        The slot sizes of each pool in the pool group.
        """
        return self._pool_groups[pool_group_index].slot_size

    @property
    def slot_size_lists(self) -> HomoTuple[HomoTuple[int]]:
        """
        A tuple of tuples, each containing the slot sizes for a pool group.
        """
        return tuple(
            tuple(p.slot_size for p in pg._pools) for pg in self._pool_groups)

    @property
    def num_pool_groups(self) -> int:
        return len(self._pool_groups)

    def resize(self,
               new_total_quota: int | None = None,
               new_ratio_list: Sequence[float] | None = None) -> None:
        new_slot_count_list = self._compute_slot_count_list(
            new_total_quota, new_ratio_list)
        self._resize_impl(new_slot_count_list)
        if new_total_quota is not None:
            self._total_quota = new_total_quota
        if new_ratio_list is not None:
            self._ratio_list = tuple(new_ratio_list)

    def _resize_impl(self, new_slot_count_list: Sequence[int]):
        old_slot_count_list = self.slot_count_list
        assert old_slot_count_list == self._compute_slot_count_list(
            self.total_quota, self.ratio_list)
        try:
            # shrink first to avoid intermediate state with excessive memory usage
            for pg, new_slot_count, old_slot_count in zip(
                    self._pool_groups, new_slot_count_list,
                    old_slot_count_list):
                if new_slot_count < old_slot_count:
                    pg.resize_slot_allocator(
                        new_slot_count
                    )  # shrink slot allocators first as it can fail for shrinking
                    pg.resize_pools(new_slot_count)
            for pg, new_slot_count, old_slot_count in zip(
                    self._pool_groups, new_slot_count_list,
                    old_slot_count_list):
                if new_slot_count > old_slot_count:
                    pg.resize_pools(
                        new_slot_count
                    )  # expand pools first as it can fail for expanding
                    pg.resize_slot_allocator(new_slot_count)
        except Exception:
            self._resize_impl(old_slot_count_list)
            raise

    # Calculate how many slots will there be in each pool group with the given total_quota and ratio_list. Use _ratio_to_slot_count_list for initialization.
    def _compute_slot_count_list(
            self,
            total_quota: int | None = None,
            ratio_list: Sequence[float] | None = None) -> HomoTuple[int]:
        if total_quota is None:
            total_quota = self.total_quota
        if ratio_list is None:
            ratio_list = self.ratio_list
        assert len(ratio_list) == len(
            self._pool_groups
        ), f"Wrong ratio_list length. Expected {len(self._pool_groups)}, got {len(ratio_list)}"
        return self._ratio_to_slot_count_list(total_quota, self.slot_size_lists,
                                              ratio_list,
                                              self.POOL_SIZE_GRANULARITY)

    @staticmethod
    def _ratio_to_slot_count_list(
            total_quota: int,
            slot_size_lists: Sequence[Sequence[int]],
            ratio_list: Sequence[float],
            pool_size_granularity: int = 1) -> HomoTuple[int]:
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
    POOL_SIZE_GRANULARITY: ClassVar[int] = PhysMem.SIZE
    __slots__ = ('shared_phys_mem_pool', )
    shared_phys_mem_pool: PooledPhysMemAllocator

    def __init__(self, total_quota: int,
                 slot_size_lists: Sequence[Sequence[int]],
                 init_ratio: Sequence[float]):
        assert len(slot_size_lists) == len(
            init_ratio
        ), "slot_size_lists and init_ratio must have the same length"
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio,
            self.POOL_SIZE_GRANULARITY)
        self.shared_phys_mem_pool = PooledPhysMemAllocator()
        self._pool_groups = tuple(
            GpuPoolGroup(num_slots, slot_size_list, self.shared_phys_mem_pool)
            for slot_size_list, num_slots in zip(slot_size_lists,
                                                 slot_count_list))

    @property
    @override
    def cache_tier(self) -> CacheTier:
        return CacheTier.GPU_MEM

    @override
    def resize(self,
               new_total_quota: int | None = None,
               new_ratio_list: Sequence[float] | None = None):
        super().resize(new_total_quota, new_ratio_list)
        self.shared_phys_mem_pool.clear()  # clear cached unused phys mem


class HostCacheLevelStorage(CacheLevelStorage):
    __slots__ = ()

    def __init__(self, total_quota: int,
                 slot_size_lists: Sequence[Sequence[int]],
                 init_ratio: Sequence[float]):
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio,
            self.POOL_SIZE_GRANULARITY)
        self._pool_groups = tuple(
            HostPoolGroup(num_slots, slot_size_list) for slot_size_list,
            num_slots in zip(slot_size_lists, slot_count_list))

    @property
    @override
    def cache_tier(self) -> CacheTier:
        return CacheTier.HOST_MEM


class DiskCacheLevelStorage(CacheLevelStorage):
    __slots__ = ()

    def __init__(self, total_quota: int,
                 slot_size_lists: Sequence[Sequence[int]],
                 init_ratio: Sequence[float], filename_template: str):
        super().__init__(total_quota, init_ratio)
        slot_count_list = self._ratio_to_slot_count_list(
            total_quota, slot_size_lists, init_ratio,
            self.POOL_SIZE_GRANULARITY)
        self._pool_groups = tuple(
            DiskPoolGroup(num_slots, slot_size_list,
                          filename_template.format(pg_idx, '{}'))
            for pg_idx, (
                slot_size_list,
                num_slots) in enumerate(zip(slot_size_lists, slot_count_list)))

    @property
    @override
    def cache_tier(self) -> CacheTier:
        return CacheTier.DISK


class CacheStorage:
    __slots__ = ('_life_cycle_id_to_pool_group_index', '_cache_levels')

    # different life cycle variants may map to the same pool group
    _life_cycle_id_to_pool_group_index: dict[LifeCycleId, PoolGroupIndex]
    # sorted by cache levels, None means no storage for this cache level
    _cache_levels: HomoTuple[CacheLevelStorage]

    def __init__(self, config: StorageConfig | KVCacheManagerConfig):
        if isinstance(config, KVCacheManagerConfig):
            config = create_storage_config(config)
        slot_size_lists = [pg.slot_size_list for pg in config.pool_groups]
        # @TODO: accept an optional avg_seq_len param and consider sliding window.
        init_ratio = [
            sum(pg.slot_size_list) * len(pg.slots) for pg in config.pool_groups
        ]
        total = sum(init_ratio)
        init_ratio = [x / total for x in init_ratio]
        self._cache_levels = tuple(
            self._create_cache_level_storage(tier_config, slot_size_lists,
                                             init_ratio)
            for tier_config in config.cache_tiers)
        self._life_cycle_id_to_pool_group_index = config.life_cycle_grouping()

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

    @property
    def num_cache_levels(self) -> int:
        return len(self._cache_levels)

    @property
    def cache_tiers(self) -> HomoTuple[CacheTier]:
        return tuple(cache_level.cache_tier
                     for cache_level in self._cache_levels)

    def get_pool_group_index(self, life_cycle: LifeCycleId) -> PoolGroupIndex:
        return PoolGroupIndex(
            self._life_cycle_id_to_pool_group_index[life_cycle])

    def get_num_free_slots(self, cache_level: CacheLevel,
                           pg_idx: PoolGroupIndex) -> int:
        return self._pool_group(cache_level, pg_idx).num_free_slots

    def allocate(self, life_cycle: LifeCycleId,
                 cache_level: CacheLevel) -> Slot:
        assert 0 <= cache_level < self.num_cache_levels, f"Cache level {cache_level} is invalid"
        pool_group_index = self.get_pool_group_index(life_cycle)
        return self._pool_group(cache_level, pool_group_index).allocate()

    def allocate_multiple(self, life_cycle: LifeCycleId,
                          cache_level: CacheLevel,
                          num_slots: int) -> list[Slot]:
        assert 0 <= cache_level < self.num_cache_levels, f"Cache level {cache_level} is invalid"
        pool_group_index = self.get_pool_group_index(life_cycle)
        return self._pool_group(cache_level,
                                pool_group_index).allocate_multiple(num_slots)

    def allocate_multiple_for_all_life_cycles(
            self, cache_level: CacheLevel,
            num_slots_per_life_cycle: list[int]) -> list[list[Slot]]:
        assert 0 <= cache_level < self.num_cache_levels, f"Cache level {cache_level} is invalid"
        assert (
            len(num_slots_per_life_cycle) == self.num_life_cycles
        ), f"num_slots_per_life_cycle must have {self.num_life_cycles} elements"
        num_slots_per_pool_group = [0] * self.num_pool_groups
        for i in range(self.num_life_cycles):
            num_slots_per_pool_group[self.get_pool_group_index(
                LifeCycleId(i))] += num_slots_per_life_cycle[i]
        if any(
                self.get_num_free_slots(cache_level, PoolGroupIndex(pg_idx)) <
                num_slots
                for pg_idx, num_slots in enumerate(num_slots_per_pool_group)):
            raise OutOfPagesError("Not enough free slots")
        return [
            self.allocate_multiple(LifeCycleId(life_cycle_id), cache_level,
                                   num_slots)
            for life_cycle_id, num_slots in enumerate(num_slots_per_life_cycle)
        ]

    def free(self, life_cycle: LifeCycleId, cache_level: CacheLevel,
             slot: Slot):
        self._pool_group(cache_level,
                         self.get_pool_group_index(life_cycle)).free(slot)

    def slot_address(self, life_cycle: LifeCycleId, cache_level: CacheLevel,
                     slot: Slot) -> HomoTuple[Address]:
        return self._pool_group(
            cache_level,
            self.get_pool_group_index(life_cycle)).slot_address(slot)

    def num_slots(self, cache_level: CacheLevel,
                  pool_group_index: PoolGroupIndex) -> int:
        return self._pool_group(cache_level, pool_group_index).num_slots

    def slot_size(self, pool_group_index: PoolGroupIndex) -> HomoTuple[int]:
        assert self._cache_levels
        return get_uniform_attribute(self._cache_levels,
                                     lambda s: s.slot_size(pool_group_index))

    def resize(self, cache_level: CacheLevel, new_quota: int) -> bool:
        self._cache_levels[cache_level].resize(new_quota, None)
        raise NotImplementedError("Not implemented")

    def update_ratio(self, cache_level: CacheLevel,
                     ratio_list: Sequence[float]) -> bool:
        self._cache_levels[cache_level].resize(None, ratio_list)
        raise NotImplementedError("Not implemented")

    @property
    def num_life_cycles(self) -> int:
        return len(self._life_cycle_id_to_pool_group_index)

    @property
    def num_pool_groups(self) -> int:
        return get_uniform_attribute(self._cache_levels,
                                     lambda s: len(s._pool_groups))

    def num_pools_in_group(self, pool_group_index: PoolGroupIndex) -> int:
        return get_uniform_attribute(
            self._cache_levels,
            lambda s: s._pool_groups[pool_group_index].num_pools)

    def _pool_group(self, cache_level: CacheLevel,
                    pool_group_index: PoolGroupIndex) -> PoolGroupBase:
        assert 0 <= cache_level < self.num_cache_levels, f"Cache level {cache_level} is invalid"
        return self._cache_levels[cache_level]._pool_groups[pool_group_index]
