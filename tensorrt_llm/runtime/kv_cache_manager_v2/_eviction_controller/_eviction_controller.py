import weakref
from typing import Callable, Iterable, Protocol, runtime_checkable

from llist import sllist, sllistnode

from .._block_radix_tree import BlockRadixTree
from .._common import CacheLevel, Priority
from .._core._kv_cache_manager import KVCacheManager
from .._eviction_controller import PageStatus
from .._exceptions import OutOfPagesError
from .._life_cycle_registry import LifeCycle, LifeCycleId
from .._storage._core import CacheStorage, PoolGroupIndex
from .._utils import HomoTuple, unwrap_weakref


@runtime_checkable
class EvictablePage(Protocol):

    @property
    def cache_level(self) -> CacheLevel:
        ...

    @property
    def priority(self) -> Priority:
        ...

    @property
    def life_cycle(self) -> LifeCycleId:
        ...

    @property
    def status(self) -> PageStatus:
        ...

    @staticmethod
    def is_committed() -> bool:
        ...

    node_ref: 'EvictionPolicy.NodeRef | None'


@runtime_checkable
class EvictionPolicy(Protocol):

    @runtime_checkable
    class NodeRef(Protocol):

        @property
        def value(self) -> EvictablePage:
            ...

    def push(self, page: EvictablePage) -> NodeRef:
        ...

    def front(self) -> NodeRef:
        ...

    # Remove a node so we no longer consider it for eviction. Like pop() but allow removing a node that is not the first.
    def remove(self, node: NodeRef) -> EvictablePage:
        ...

    def __len__(self) -> int:
        ...


class LRUEvictionPolicy:
    __slots__ = ('_queue', )
    _queue: sllist

    NodeRef = sllistnode

    def __init__(self):
        self._queue = sllist()

    def push(self, page: EvictablePage) -> sllistnode:
        return self._queue.append(page)

    def front(self) -> sllistnode:
        return self._queue.first

    def remove(self, node: EvictionPolicy.NodeRef) -> EvictablePage:
        assert isinstance(node, self.NodeRef)
        assert node == node.value.node_ref
        return self._queue.remove(node)

    def __len__(self) -> int:
        return len(self._queue)


# helper class to help add support for priority-based eviction
class PrioritizedEvictionPolicy:
    __slots__ = (
        '_policy_creator',
        '_policies',
    )
    _policy_creator: Callable[[Priority], EvictionPolicy]
    _policies: dict[Priority, EvictionPolicy]

    NodeRef = EvictionPolicy.NodeRef

    def __init__(self, policy_creator: Callable[[Priority], EvictionPolicy]):
        self._policy_creator = policy_creator
        self._policies = {}

    def __len__(self) -> int:
        return sum(len(policy) for policy in self._policies.values())

    def get_policy(self, priority: Priority) -> EvictionPolicy:
        if priority not in self._policies:
            self._policies[priority] = self._policy_creator(priority)
            self._policies = dict(sorted(self._policies.items()))
        return self._policies[priority]

    def _front_policy(self) -> EvictionPolicy:
        return next(iter(self._policies.values()))

    def push(self, page: EvictablePage) -> EvictionPolicy.NodeRef:
        return self.get_policy(page.priority).push(page)

    def front(self) -> EvictionPolicy.NodeRef:
        return self._front_policy().front()

    def remove(self, node: EvictionPolicy.NodeRef) -> EvictablePage:
        page = node.value
        assert page.node_ref == node
        policy = self._policies[page.priority]
        policy.remove(node)
        if not policy:
            self._policies.pop(page.priority)
        return page


class PrioritizedLRUEvictionPolicy(PrioritizedEvictionPolicy):
    __slots__ = ()
    NodeRef = LRUEvictionPolicy.NodeRef

    def __init__(self):
        super().__init__(lambda priority: LRUEvictionPolicy())


class PerLevelEvictionController:  # for one cache level
    __slots__ = ('_controller', '_level', '_radix_tree', '_policies')
    _controller: weakref.ref['EvictionController']
    _level: CacheLevel
    _policies: dict[PoolGroupIndex, EvictionPolicy]

    def __init__(self, controller: 'EvictionController', level: CacheLevel):
        self._controller = weakref.ref(controller)
        self._level = level
        manager = controller.manager
        self._policies = {
            PoolGroupIndex(pg_idx): PrioritizedLRUEvictionPolicy()
            for pg_idx in range(manager.storage.num_pool_groups)
        }

    def _get_policy(self, life_cycle: LifeCycleId) -> EvictionPolicy:
        pg_idx = self.storage.get_pool_group_index(life_cycle)
        return self._policies[pg_idx]

    def schedule_for_eviction(self, page: EvictablePage):
        assert page.node_ref is None
        page.node_ref = self._get_policy(page.life_cycle).push(page)

    # Usually only evict at most num_pages pages, but if evicting a node makes some other nodes useless, those nodes will be returned as well.
    # One example: for SWA, if the number of blocks just makes up one window size, then evicting any of them makes the remaining blocks useless.
    # Raise if no enough pages to evict.
    def get_eviction_candidates(
            self, pg_idx: PoolGroupIndex,
            min_num_pages: int) -> list[EvictionPolicy.NodeRef]:
        policy = self._policies[pg_idx]
        if len(policy) < min_num_pages:
            raise OutOfPagesError(
                f"Not enough pages to evict. {len(policy)} < {min_num_pages}")
        pages: list[EvictionPolicy.NodeRef] = []
        while len(pages) < min_num_pages:
            nodes = self._next_to_evict(pg_idx)
            assert len(nodes) > 0
            pages.extend(nodes)
        return pages

    def remove(
            self, nodes: Iterable[EvictionPolicy.NodeRef]
    ) -> HomoTuple[EvictablePage]:
        ret = tuple(
            self._get_policy(n.value.life_cycle).remove(n) for n in nodes)
        for page in ret:
            page.node_ref = None
        return ret

    @property
    def radix_tree(self) -> BlockRadixTree:
        return self.manager._radix_tree

    def _next_to_evict(self,
                       pg_idx: PoolGroupIndex) -> list[EvictionPolicy.NodeRef]:
        policy = self._policies[pg_idx]
        if len(policy) == 0:
            return []
        seed = policy.front()
        ret = [seed]
        # For now, we ignore dependency check. This should be good enough for LRU policy.
        return ret
        # @todo: Check if evicting this page makes any other pages useless.
        front_page = seed.value
        front_page.cache_level
        tree = self.radix_tree

        seed_block = unwrap_weakref(front_page.block)
        life_cycle: LifeCycle = tree.life_cycles.get_life_cycle(
            front_page.life_cycle_id)
        if seed_block.depth < life_cycle.num_sink_blocks or life_cycle.sliding_win_size is None:
            for block in tree.traverse_subtree(seed_block):
                raise NotImplementedError("Not implemented")
        else:
            raise NotImplementedError("Not implemented")

    @property
    def controller(self) -> 'EvictionController':
        return unwrap_weakref(self._controller)

    @property
    def manager(self) -> KVCacheManager:
        return self.controller.manager

    @property
    def storage(self) -> CacheStorage:
        return self.manager.storage


class HeldNodeRef:
    __slots__ = ('_page', )
    _page: EvictablePage

    def __init__(self, page: EvictablePage):
        self._page = page

    @property
    def value(self) -> EvictablePage:
        return self._page


assert issubclass(HeldNodeRef, EvictionPolicy.NodeRef)


class EvictionController:
    __slots__ = ('_storage', '_radix_tree', '_per_level_controllers',
                 '_held_pages')
    _manager: weakref.ref[KVCacheManager]
    _radix_tree: weakref.ref[BlockRadixTree]
    # sorted by CacheLevel from warm to cold
    _per_level_controllers: tuple[PerLevelEvictionController]

    def schedule_for_eviction(self, page: EvictablePage) -> bool:
        'If success, sets page.node_ref.'
        assert page.node_ref is None
        if not self.is_evictable(page):
            return False
        self._per_level_controllers[page.cache_level].schedule_for_eviction(
            page)
        return True

    def is_evictable(self, page: EvictablePage) -> bool:
        status = page.status
        # droppable pages that are not committed should be dropped immediately.
        # held pages in last level cache can't be evicted.
        return (status == PageStatus.DROPPABLE and page.is_committed()) or (
            status == PageStatus.HELD
            and page.cache_level < self.num_cache_levels - 1)

    def get_eviction_candidates(self, cache_level: CacheLevel,
                                pg_idx: PoolGroupIndex,
                                num_pages: int) -> list[EvictionPolicy.NodeRef]:
        return self._per_level_controllers[cache_level].get_eviction_candidates(
            pg_idx, num_pages)

    def remove(
            self, nodes: Iterable[EvictionPolicy.NodeRef]
    ) -> HomoTuple[EvictablePage]:
        return tuple(
            self._per_level_controllers[n.value.cache_level].remove((n, ))[0]
            for n in nodes)

    @property
    def radix_tree(self) -> BlockRadixTree:
        return unwrap_weakref(self._radix_tree)

    @property
    def num_cache_levels(self) -> int:
        return len(self._per_level_controllers)

    @property
    def manager(self) -> KVCacheManager:
        return unwrap_weakref(self._manager)

    @property
    def storage(self) -> CacheStorage:
        return self.manager.storage
