import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, cast

from ._block_radix_tree import Block
from ._common import (BAD_PAGE_INDEX, GPU_LEVEL, NDEBUG, BeamIndex,
                      BlockOrdinal, CacheLevel, PageIndex, Priority, TokenIdExt)

if TYPE_CHECKING:
    from ._core._kv_cache import _KVCache
    from ._storage_manager import StorageManager

import numba as nb
import numpy as np
import numpy.typing as npt

from ._eviction_controller import EvictionPolicy, PageStatus
from ._exceptions import LogicError
from ._life_cycle_registry import LifeCycleId
from ._storage._core import Slot, SlotId
from ._utils import (CachedCudaEvent, assert_critical, filled_list,
                     get_uniform_attribute, merge_events, partition,
                     stream_wait_events, unwrap_optional, unwrap_weakref)


# We will have a huge amount of pages for large storage capacity.
# So we prefer inheritance over composition to save some memory.
@dataclass(slots=True)
class Page(Slot):
    _manager: weakref.ref['StorageManager']
    life_cycle: LifeCycleId
    cache_level: CacheLevel
    _priority: Priority
    # _holder is either None or a valid weakref.
    _holder: weakref.ref['_PageHolder'] | None
    node_ref: EvictionPolicy.NodeRef | None

    def __del__(self):
        if not NDEBUG:
            assert_critical(self.status == PageStatus.DROPPABLE
                            and not self.scheduled_for_eviction)
        if self.has_valid_slot:
            self.manager.release_slot(self.life_cycle, self.cache_level, self)

    @property
    def manager(self) -> 'StorageManager':
        return unwrap_weakref(self._manager)

    @property
    def priority(self) -> Priority:
        return self._priority

    # prevent dropping
    def hold(self) -> '_PageHolder':
        if self._holder is not None:
            return unwrap_weakref(self._holder)
        holder = object.__new__(_PageHolder)
        holder._setup(self)
        self._holder = weakref.ref(holder)
        controller = self.manager
        if self.scheduled_for_eviction and not controller.is_evictable(self):
            controller.exclude_from_eviction(self)
            assert not self.scheduled_for_eviction
        return holder

    # Prevent eviction. You need to migrate the page to GPU later.
    def lock(self,
             kv_cache: '_KVCache',
             beam_index: BeamIndex,
             ordinal: BlockOrdinal,
             life_cycle: LifeCycleId,
             skip_wait: bool = False) -> '_SharedPageLock':
        'If skip wait, you are responsible for making the page ready in kv_cache.cuda_stream.'
        return self.hold().lock(kv_cache, beam_index, ordinal, life_cycle,
                                skip_wait)

    @property
    def status(self) -> PageStatus:
        if self._holder is None:
            return PageStatus.DROPPABLE
        lock_ref = unwrap_weakref(self._holder)._lock
        if lock_ref is None:
            return PageStatus.HELD
        assert unwrap_weakref(lock_ref) is not None
        return PageStatus.LOCKED

    @property
    def scheduled_for_eviction(self) -> bool:
        return self.node_ref is not None

    @staticmethod
    def is_committed() -> bool:
        raise LogicError("Unexpected call to this implementation.")


@dataclass(slots=True)
class UncommittedPage(Page):
    #@TODO: consider move this to _PageHolder
    kv_cache: weakref.ref['_KVCache']
    ordinal: BlockOrdinal
    beam_index: BeamIndex

    tokens: list[TokenIdExt] = field(default_factory=list)

    @staticmethod
    def is_committed() -> bool:
        return False

    def __init__(self,
                 kv_cache: '_KVCache',
                 ordinal: BlockOrdinal,
                 life_cycle: LifeCycleId,
                 cache_level: CacheLevel,
                 slot: Slot,
                 beam_index: BeamIndex = BeamIndex(0)):
        self.kv_cache = weakref.ref(kv_cache)
        self.ordinal = ordinal
        self.beam_index = beam_index
        manager = kv_cache.manager
        priority = kv_cache._get_priority(
            ordinal, manager._life_cycles.get_life_cycle(life_cycle))
        Page.__init__(self, None, CachedCudaEvent.NULL,
                      weakref.ref(manager._storage), life_cycle, cache_level,
                      priority, None, None)
        self.set_slot(slot)

    def convert_to_committed(self, block: Block) -> 'CommittedPage':
        'Moves the slot to a new committed page and add the new page to the block. The uncommitted page becomes invalid.'
        assert not self.scheduled_for_eviction
        assert block.storage[self.life_cycle] is None
        # If you hit this assertion failure, it's likely because you are using debugpy, which delayed GC for _KVCache._take_uncommitted_page(). Disable breakpoints on exceptions to avoid this issue.
        assert self.status == PageStatus.DROPPABLE, "Release holder/lock first"
        committed_page = CommittedPage(self.manager, block, self.life_cycle,
                                       self.cache_level, self, self.priority)
        self._slot_id = None
        self.ready_event = CachedCudaEvent.NULL
        assert committed_page.has_valid_slot
        block.storage[self.life_cycle] = weakref.ref(committed_page)
        return committed_page

    def __del__(self):
        check_page = lambda p: p is None or isinstance(p.page, CommittedPage)
        if not NDEBUG:
            assert_critical(
                len(unwrap_weakref(self.kv_cache)._blocks) <= self.ordinal
                or check_page(
                    unwrap_weakref(self.kv_cache)._blocks[self.ordinal].pages[
                        self.beam_index][self.life_cycle]))
        Page.__del__(self)


@dataclass(slots=True, weakref_slot=True)
class CommittedPage(Page):
    block: weakref.ref['Block']

    @staticmethod
    def is_committed() -> bool:
        return True

    def __init__(self, storage: 'StorageManager', block: Block,
                 life_cycle: LifeCycleId, cache_level: CacheLevel, slot: Slot,
                 priority: Priority):
        self.block = weakref.ref(block)
        Page.__init__(self, None, CachedCudaEvent.NULL, weakref.ref(storage),
                      life_cycle, cache_level, priority, None, None)
        self.set_slot(slot)

    def __del__(self):
        block = self.block()
        # block may be None when rebase happens, i.e. another block with the same key is committed, replacing it, but the page is still used by a _KVCache.
        if block is not None:
            block.unset_page(
                self.life_cycle,
                self.manager.kv_cache_manager._life_cycles.get_life_cycle(
                    self.life_cycle))
        Page.__del__(self)


@dataclass(slots=True, init=False, weakref_slot=True)
class _PageHolder:
    'Prevents pages from being dropped.'
    page: Page
    _lock: weakref.ref['_UniqPageLock'] | None

    def __init__(self):
        raise LogicError("Use page.hold() instead")

    def _setup(self, page: Page):
        self.page = page
        self._lock = None

    def __del__(self):
        if not NDEBUG:
            assert_critical(self._lock is None)
        self.page._holder = None
        # If a held page was in last level cache, it was not scheduled for eviction.
        if self.page.is_committed():
            page = cast(CommittedPage, self.page)
            if not page.scheduled_for_eviction:
                page.manager.schedule_for_eviction(page)
            block = page.block()
            if block is None or block.is_orphan:
                page.manager.exclude_from_eviction(page)
        elif self.page.scheduled_for_eviction:
            page = cast(UncommittedPage, self.page)
            page.manager.exclude_from_eviction(page)

    # Prevent eviction. You need to migrate the page to GPU later.
    def lock(self,
             kv_cache: '_KVCache',
             beam_index: BeamIndex,
             ordinal: BlockOrdinal,
             life_cycle: LifeCycleId,
             skip_wait: bool = False) -> '_SharedPageLock':
        if self._lock is None:
            lock = object.__new__(_UniqPageLock)
            lock._setup(self)
            self._lock = weakref.ref(lock)
        else:
            lock = unwrap_weakref(self._lock)
        if self.page.scheduled_for_eviction:
            manager = self.page.manager
            manager.exclude_from_eviction(self.page)
            assert not self.page.scheduled_for_eviction
        return lock.share(kv_cache, beam_index, ordinal, life_cycle, skip_wait)


@dataclass(slots=True, init=False, weakref_slot=True)
class _UniqPageLock:
    'Locks pages to prevent eviction.'
    holder: _PageHolder
    finish_events: list[CachedCudaEvent]
    users: weakref.WeakSet['_SharedPageLock']

    def __init__(self):
        raise LogicError("Use page.lock() or holder.lock() instead")

    def _setup(self, holder: _PageHolder):
        if holder.page.cache_level != CacheLevel(0):
            raise ValueError("Lock can be applied only on GPU memory pages.")
        self.holder = holder
        self.finish_events = []
        self.users = weakref.WeakSet()

    def share(self, kv_cache: '_KVCache', beam_index: BeamIndex,
              ordinal: BlockOrdinal, life_cycle: LifeCycleId,
              skip_wait: bool) -> '_SharedPageLock':
        ret = _SharedPageLock(self, kv_cache, beam_index, ordinal, life_cycle,
                              skip_wait)
        self.users.add(ret)
        return ret

    @property
    def page(self) -> Page:
        return self.holder.page

    def __del__(self):
        page = self.page
        if not NDEBUG:
            assert_critical(page.cache_level == CacheLevel(0)
                            and not page.scheduled_for_eviction)
        page.ready_event = merge_events(self.finish_events)
        self.holder._lock = None
        # delete holder first, so if nobody holds the page elsewhere, it becomes droppable immediately, before we hand it over to eviction controller.
        del self.holder
        if page.manager.is_evictable(page):
            page.manager.schedule_for_eviction(page)


class LockOwner(NamedTuple):
    kv_cache: weakref.ref['_KVCache']
    beam_index: BeamIndex
    ordinal: BlockOrdinal
    life_cycle: LifeCycleId


@nb.njit(cache=True)
def _compute_page_indices(slot_id: SlotId, scale: npt.NDArray[np.int32],
                          bias: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    return slot_id * scale + bias


@dataclass(slots=True, init=False, weakref_slot=True)
class _SharedPageLock:
    _uniq_lock: _UniqPageLock | None
    _user: LockOwner

    @property
    def page(self) -> Page:
        return unwrap_optional(self._uniq_lock).page

    @property
    def holder(self) -> _PageHolder:
        return unwrap_optional(self._uniq_lock).holder

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def __init__(self, uniq_lock: _UniqPageLock, kv_cache: '_KVCache',
                 beam_index: BeamIndex, ordinal: BlockOrdinal,
                 life_cycle: LifeCycleId, skip_wait: bool):
        self._uniq_lock = uniq_lock
        if not skip_wait:
            self.page.ready_event.wait_in_stream(kv_cache.cuda_stream)
        self._user = LockOwner(weakref.ref(kv_cache), beam_index, ordinal,
                               life_cycle)
        new_index = self._get_page_index()
        old_index = kv_cache._update_page_index(beam_index, ordinal, life_cycle,
                                                new_index)
        assert old_index == BAD_PAGE_INDEX

    def __del__(self):
        if self._uniq_lock is not None:
            self.unlock()

    def unlock(self) -> Page:
        assert self._uniq_lock is not None
        page = self.page
        self._uniq_lock.finish_events.append(
            unwrap_weakref(self._user.kv_cache).finish_event)
        new_index = BAD_PAGE_INDEX
        old_index = unwrap_weakref(self._user.kv_cache)._update_page_index(
            self._user.beam_index, self._user.ordinal, self._user.life_cycle,
            new_index)
        assert NDEBUG or old_index == self._get_page_index()
        self._uniq_lock = None
        return page

    def _get_page_index(self) -> PageIndex:
        storage = unwrap_weakref(self._user.kv_cache).manager._storage
        num_buffers_per_slot = storage._slot_to_page_indices[
            self._user.life_cycle]
        return PageIndex(self.page.slot_id * num_buffers_per_slot)


class BatchedLockTarget(NamedTuple):
    page: Page
    beam_index: BeamIndex
    ordinal: BlockOrdinal
    life_cycle: LifeCycleId


def batched_lock_to_gpu(
        kv_cache: '_KVCache',
        tasks: Sequence[BatchedLockTarget]) -> list['_SharedPageLock']:
    'Lock pages after migrating all pages to GPU. If migration fails, no locking happens.'
    storage = kv_cache.manager._storage
    assert not tasks or storage is get_uniform_attribute(
        tasks, lambda p: p.page.manager)
    requirements = filled_list(0, storage.num_pool_groups)
    scheduled_for_eviction = [t.page.scheduled_for_eviction for t in tasks]
    for t, e in zip(tasks, scheduled_for_eviction):
        if e:
            storage.exclude_from_eviction(t.page)
        if t.page.cache_level == GPU_LEVEL:
            continue
        requirements[storage.get_pool_group_index(t.life_cycle)] += 1

    try:
        storage.prepare_free_slots(GPU_LEVEL, requirements)
        partitioned = partition(
            tasks, lambda p:
            (p.page.cache_level, storage.get_pool_group_index(p.life_cycle)))
        for (lvl, pg_idx), part in partitioned.items():
            if lvl == GPU_LEVEL:
                continue
            storage._batched_migrate(pg_idx,
                                     GPU_LEVEL,
                                     lvl, [p.page for p in part],
                                     update_src=True)
    except Exception:
        for t, e in zip(tasks, scheduled_for_eviction):
            if e:
                storage.schedule_for_eviction(t.page)
        raise
    stream_wait_events(kv_cache.cuda_stream,
                       (p.page.ready_event for p in tasks))
    return [
        page.lock(kv_cache, beam_index, ordinal, life_cycle, skip_wait=True)
        for page, beam_index, ordinal, life_cycle in tasks
    ]
