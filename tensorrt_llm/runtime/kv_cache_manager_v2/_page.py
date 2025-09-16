import warnings
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from ._block_radix_tree import Block
from ._common import BlockOrdinal, CacheLevel, Priority, TokenIdExt
from ._core._kv_cache import BeamIndex, _KVCache
from ._eviction_controller import EvictionPolicy, PageStatus
from ._exceptions import LogicError
from ._life_cycle_registry import LifeCycleId
from ._storage._core import Slot
from ._storage_manager import GPU_LEVEL, StorageManager
from ._utils import (CachedCudaEvent, get_uniform_attribute, merge_events,
                     partition, stream_wait_events, unwrap_weakref)


# We will have a huge amount of pages for large storage capacity.
# So we prefer inheritance over composition to save some memory.
@dataclass(slots=True)
class Page(Slot):
    _manager: weakref.ref[StorageManager]
    life_cycle: LifeCycleId
    cache_level: CacheLevel
    _priority: Priority
    # _holder is either None or a valid weakref.
    _holder: weakref.ref['_PageHolder'] | None = field(default=None)
    node_ref: EvictionPolicy.NodeRef | None = field(default=None)

    def __del__(self):
        assert self.status == PageStatus.DROPPABLE
        assert not self.scheduled_for_eviction
        if self.has_valid_slot:
            self.manager.release_slot(self.life_cycle, self.cache_level, self)
            assert not self.has_valid_slot

    @property
    def manager(self) -> StorageManager:
        return unwrap_weakref(self._manager)

    @property
    def priority(self) -> Priority:
        return self._priority

    # prevent dropping
    def hold(self) -> '_PageHolder':
        if self._holder is not None:
            return unwrap_weakref(self._holder)
        controller = self.manager
        holder = object.__new__(_PageHolder)
        holder._setup(self)
        self._holder = weakref.ref(holder)
        if self.scheduled_for_eviction and not controller.is_evictable(self):
            controller.exclude_from_eviction(self)
            assert not self.scheduled_for_eviction
        return holder

    # Prevent eviction. You need to migrate the page to GPU later.
    def lock(self,
             kv_cache: _KVCache,
             skip_wait: bool = False) -> '_SharedPageLock':
        'If skip wait, you are responsible for making the page ready in kv_cache.cuda_stream.'
        return self.hold().lock(kv_cache, skip_wait)

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


# @TODO: may be unnecessary. Consider removing.
@dataclass(slots=True)
class UncommittedPageKey:
    #@TODO: consider move this to _PageHolder
    kv_cache: weakref.ref[_KVCache]
    ordinal: BlockOrdinal
    life_cycle: LifeCycleId
    beam_index: BeamIndex

    @staticmethod
    def is_committed() -> bool:
        return False


@dataclass(slots=True)
class CommittedPageKey:
    #@TODO: consider move this to _PageHolder
    block: weakref.ref['Block']
    life_cycle: LifeCycleId

    @staticmethod
    def is_committed() -> bool:
        return True


@dataclass(slots=True)
class UncommittedPage(UncommittedPageKey, Page):
    tokens: list[TokenIdExt] = []

    def __init__(self,
                 kv_cache: _KVCache,
                 ordinal: BlockOrdinal,
                 life_cycle: LifeCycleId,
                 cache_level: CacheLevel,
                 slot: Slot,
                 beam_index: BeamIndex = BeamIndex(0)):
        UncommittedPageKey.__init__(self, weakref.ref(kv_cache), ordinal,
                                    life_cycle, beam_index)
        manager = kv_cache.manager
        priority = kv_cache._get_priority(
            ordinal, manager._life_cycles.get_life_cycle(life_cycle))
        Page.__init__(self, slot.slot_id, slot.ready_event,
                      weakref.ref(manager._storage), life_cycle, cache_level,
                      priority)

    def convert_to_committed(self, block: Block) -> 'CommittedPage':
        'Moves the slot to a new committed page and add the new page to the block. The uncommitted page becomes invalid.'
        assert not self.scheduled_for_eviction
        assert block.storage[self.life_cycle] is None
        assert self.status == PageStatus.DROPPABLE, "Release holder/lock first"
        committed_page = CommittedPage(self.manager, block, self.life_cycle,
                                       self.cache_level, self, self.priority)
        self._slot_id = None
        self.ready_event = CachedCudaEvent.NULL
        assert committed_page.has_valid_slot
        block.storage[self.life_cycle] = weakref.ref(committed_page)
        return committed_page

    def __del__(self):
        assert unwrap_weakref(self.kv_cache)._blocks[self.ordinal].pages[
            self.beam_index][self.life_cycle] is None
        Page.__del__(self)


@dataclass(slots=True)
class CommittedPage(CommittedPageKey, Page):

    def __init__(self, storage: StorageManager, block: Block,
                 life_cycle: LifeCycleId, cache_level: CacheLevel, slot: Slot,
                 priority: Priority):
        CommittedPageKey.__init__(self, weakref.ref(block), life_cycle)
        Page.__init__(self, slot.slot_id, slot.ready_event,
                      weakref.ref(storage), life_cycle, cache_level, priority)

    def __del__(self):
        block = unwrap_weakref(self.block)
        block.storage[self.life_cycle] = None
        # if this makes the block unusable, remove it from the radix tree.
        warnings.warn(
            "[KVCacheManager] Implement a better way to detect and remove ununsable blocks, which should consider SWA layers."
        )
        # For now, we use a very simple approach to avoid accumulation of radix tree nodes.
        block.remove_if_unusable()
        Page.__del__(self)


@dataclass(slots=True, init=False)
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
        assert self._lock is None
        self.page._holder = None
        # If a held page was in last level cache, it was not scheduled for eviction.
        if self.page.is_committed():
            page = cast(CommittedPage, self.page)
            if not page.scheduled_for_eviction:
                page.manager.schedule_for_eviction(page)
            block = page.block()
            if block is None or block.is_orphan:
                page.manager.exclude_from_eviction(page)

    # Prevent eviction. You need to migrate the page to GPU later.
    def lock(self,
             kv_cache: _KVCache,
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
        return lock.share(kv_cache, skip_wait)


@dataclass(slots=True, init=False)
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

    def share(self, kv_cache: _KVCache, skip_wait) -> '_SharedPageLock':
        ret = _SharedPageLock(self, kv_cache, skip_wait)
        self.users.add(ret)
        return ret

    @property
    def page(self) -> Page:
        return self.holder.page

    def __del__(self):
        page = self.page
        assert page.cache_level == CacheLevel(
            0) and not page.scheduled_for_eviction
        page.ready_event = merge_events(self.finish_events)
        self.holder._lock = None
        # delete holder first, so if nobody holds the page elsewhere, it becomes droppable immediately, before we hand it over to eviction controller.
        del self.holder
        if page.manager.is_evictable(page):
            page.manager.schedule_for_eviction(page)


@dataclass(slots=True, init=False)
class _SharedPageLock:
    _uniq_lock: _UniqPageLock
    _user: weakref.ref[_KVCache]

    @property
    def page(self) -> Page:
        return self._uniq_lock.page

    @property
    def holder(self) -> _PageHolder:
        return self._uniq_lock.holder

    def __init__(self, uniq_lock: _UniqPageLock, user: _KVCache,
                 skip_wait: bool):
        self._uniq_lock = uniq_lock
        if not skip_wait:
            self.page.ready_event.wait_in_stream(user.cuda_stream)
        self._user = weakref.ref(user)

    def __del__(self):
        self._uniq_lock.finish_events.append(
            unwrap_weakref(self._user).finish_event)


def batched_lock_to_gpu(kv_cache: _KVCache,
                        pages: Sequence[Page]) -> list[_SharedPageLock]:
    'Lock pages after migrating all pages to GPU. If migration fails, no locking happens.'
    storage = get_uniform_attribute(pages, lambda p: p.manager)
    partitioned = partition(
        pages, lambda p:
        (p.cache_level, storage.get_pool_group_index(p.life_cycle)))
    for (lvl, pg_idx), part in partitioned.items():
        if lvl == GPU_LEVEL:
            continue
        storage.batched_migrate(pg_idx, GPU_LEVEL, lvl, part, update_src=True)
    stream_wait_events(kv_cache.cuda_stream, (p.ready_event for p in pages))
    return [p.lock(kv_cache, skip_wait=True) for p in pages]
