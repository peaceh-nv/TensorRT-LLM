import hashlib
import warnings
import weakref
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence, TypeVar, cast

from ._common import NDEBUG, BlockOrdinal, TokenId, TokenIdExt
from ._eviction_controller import PageStatus
from ._life_cycle_registry import LifeCycleId, LifeCycleRegistry
from ._utils import (HomoTuple, TypedIndexList, chunked, filled_list,
                     unwrap_weakref)

if TYPE_CHECKING:
    from ._page import CommittedPage

BlockKey = bytes


# id_offset is usually vocab_size
def gen_multi_modal_tokens(id_offset: int, multi_modal_data_digest: bytes,
                           num_tokens: int) -> list[TokenIdExt]:
    assert num_tokens > 0
    # Alternatively, we could also use (multi_modal_data_digest + i.to_bytes(8, 'little')) or its hash digest as token id.
    # The implementation below is faster and also works because KV cache reuse of a token is with a precondition that all previous tokens also match. So only the first multi-modal token id needs to be unique.
    return [
        multi_modal_data_digest if i == 0 else TokenId(id_offset + i)
        for i in range(num_tokens)
    ]


class Hasher:
    __slots__ = ('_hasher')
    _hasher: 'hashlib._Hash'

    def __init__(self, data: int | bytes | None | Sequence[int | bytes] = None):
        self._hasher = hashlib.sha256()
        if data is not None:
            self.update(data)

    # This function is perf-critical. Expect compromised code quality.
    def update(self, data: int | bytes | Sequence[int | bytes]) -> 'Hasher':
        data_type = type(data)
        if data_type is int:
            assert NDEBUG or (data >= 0 and data <  # type: ignore[operator]
                              (1 << 64))
            self._hasher.update(
                data.to_bytes(  # type: ignore[attr-defined]
                    8, 'little'))
        elif data_type is bytes:
            self._hasher.update(data)  # type: ignore[attr-defined]
        else:
            assert NDEBUG or isinstance(data, Iterable)
            for item in data:  # type: ignore[attr-defined]
                assert NDEBUG or (type(item) is int and
                                  (item >= 0
                                   and item < (1 << 64))) or type(item) is bytes
                self._hasher.update(
                    item.to_bytes(8, 'little') if type(item) is int else
                    item)  # type: ignore[attr-defined]
        return self

    @property
    def digest(self) -> bytes:
        return self._hasher.digest()


TokenBlock = HomoTuple[TokenIdExt]


def sequence_to_blockchain_keys(
        tokens_per_block: int, lora_task_id: int | None,
        tokens: Sequence[TokenIdExt]) -> Iterator[tuple[TokenBlock, BlockKey]]:
    digest = Hasher(lora_task_id).digest
    yield (), digest
    for token_block in chunked(tokens, tokens_per_block):
        digest = Hasher(digest).update(token_block).digest
        yield token_block, digest


Child = TypeVar('Child', bound='Block | RootBlock')
Children = dict[BlockKey, Child]


def get_tree(block: 'RootBlock | Block') -> 'BlockRadixTree':
    node = block
    while not isinstance(node, BlockRadixTree):
        node = node.prev
    return node


def remove_subtree(
        root: 'RootBlock | Block',
        yield_pages: bool = False) -> Iterator[weakref.ref['CommittedPage']]:
    # taking O(1) space
    # remove leaf blocks one by one, in post-order
    block: 'RootBlock | Block' = root
    while True:
        if block.next:
            block = next(iter(block.next.values()))
        else:
            if yield_pages and isinstance(block, Block):
                yield from (page for page in block.storage if page is not None)
                block.storage = filled_list(None, block.num_life_cycles)
            assert (isinstance(block, RootBlock) or all(
                page is None
                for page in block.storage)), "Storage is not cleared, yet"
            prev_block: Block | RootBlock | BlockRadixTree = block.prev
            prev_block.next.pop(block.key)
            if block is root:
                break
            assert not isinstance(prev_block, BlockRadixTree)
            block = prev_block


def traverse_subtree(root: 'Block') -> Iterator['Block']:
    'post-order traversal of the subtree rooted at root'
    stack: list[Iterator[Block]] = []
    block = root
    while True:
        if block.next:
            child_iter = iter(block.next.values())
            stack.append(child_iter)
            block = next(child_iter)
        else:
            yield (last_yielded := block)
            while stack and (block := next(stack[-1], None)) is None:
                yield (last_yielded := cast(Block, last_yielded.prev))
                stack.pop()
            if not stack:
                break


def find_best_partial_match_in_next_nodes(
        block: 'Block | RootBlock',
        tokens: TokenBlock) -> tuple['Block | None', int]:
    """
    Among all child nodes (self.next), finds the one whose tokens have the longest leading match with the given tokens.
    Returns a tuple of (best_block, num_matched_tokens).
    If no child matches any tokens, returns (None, 0).
    """
    if len(block.next) >= 32:
        warnings.warn(
            "[KVCacheManager] Not Implemented: build a database to accelerate partial matching."
        )
    best_block = None
    best_match_len = 0
    for b in block.next.values():
        match_len = b._partial_match_this_node(tokens)
        if match_len > best_match_len:
            best_match_len = match_len
            best_block = b
    return best_block, best_match_len


class DuplicateKeyError(RuntimeError):
    key: BlockKey

    def __init__(self, key: BlockKey):
        super().__init__(f"Block with key {key.hex()} already exists")
        self.key = key


class UselessBlockError(RuntimeError):
    block: 'Block'

    def __init__(self, block: 'Block'):
        super().__init__(
            f"Block is useless because all its tokens are covered by another block with key = {block.key.hex()}"
        )
        self.block = block


def _add_or_get_existing(parent: 'RootBlock | Block',
                         tokens: Sequence[TokenIdExt]) -> 'Block | None':
    try:
        return Block(tokens, parent)
    except DuplicateKeyError as e:
        return parent.next[e.key]
    except UselessBlockError:
        return None


class RootBlock:
    __slots__ = ('_prev', 'key', 'next', 'lora_task_id', '__weakref__')
    key: BlockKey
    lora_task_id: int | None
    _prev: weakref.ref['BlockRadixTree']
    next: Children['Block']

    def __init__(self, lora_task_id: int | None, prev: 'BlockRadixTree'):
        self.key = self.make_key(lora_task_id)
        assert self.key not in prev.next, "Root block already exists"
        self.lora_task_id = lora_task_id
        self._prev = weakref.ref(prev)
        self.next = {}
        prev.next[self.key] = self

    @property
    def ordinal(self) -> BlockOrdinal:
        return BlockOrdinal(-1)

    @property
    def prev(self) -> 'BlockRadixTree':
        return unwrap_weakref(self._prev)

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return self.prev.num_life_cycles

    @property
    def tokens_per_block(self) -> int:
        return self.prev.tokens_per_block

    @staticmethod
    def make_key(lora_task_id: int | None) -> BlockKey:
        return Hasher(lora_task_id).digest


class Block:
    """
    A block of tokens. Manages data for all layers.
    """
    __slots__ = ('key', 'tokens', 'ordinal', '_prev', 'next', 'storage',
                 '__weakref__')
    key: BlockKey
    tokens: Sequence[TokenIdExt]
    ordinal: BlockOrdinal
    _prev: weakref.ref['Block | RootBlock']
    next: Children['Block']

    # indexed with LifeCycleId
    storage: TypedIndexList[LifeCycleId, weakref.ref['CommittedPage'] | None]

    @staticmethod
    def make_key(prev_key: BlockKey, tokens: Sequence[TokenIdExt]) -> BlockKey:
        return Hasher(prev_key).update(tokens).digest

    def __init__(self, tokens: Sequence[TokenIdExt], prev: 'Block | RootBlock'):
        assert prev.tokens_per_block == prev.prev.tokens_per_block, 'prev must be a full block'
        self.key = self.make_key(prev.key, tokens)
        # a Block is useless if all its tokens are covered by a sibling block. Raise UselessBlockError if so.
        if self.key in prev.next:
            raise UselessBlockError(prev.next[self.key])
        # @TODO: when we have the database for find_best_partial_match_in_next_nodes, we may use that for faster check.
        for b in prev.next.values():
            if b.tokens[:len(tokens)] == tokens:
                raise UselessBlockError(b)
        self.tokens = tokens
        self.ordinal = BlockOrdinal(prev.ordinal + 1)
        self._prev = weakref.ref(prev)
        # prev.next keeps a strong ref to this _Block, so no need to remove self from prev.next in __del__().
        prev.next[self.key] = self
        self.next = {}
        self.storage = filled_list(None, prev.num_life_cycles)
        # If there are sibling blocks fully covered by this block, remove them.
        for k, b in prev.next.items():
            if len(b.tokens) < len(
                    tokens) and tokens[:len(b.tokens)] == b.tokens:
                assert NDEBUG or (not b.is_full and b is not self and b.key == k
                                  and not b.next)
                prev.next.pop(k)
                assert b.is_orphan  # _KVCache may still hold it.

    def __del__(self):
        for ref in self.storage:
            if ref is not None and ref() is not None:
                page = unwrap_weakref(ref)
                if page.status == PageStatus.DROPPABLE:
                    if page.scheduled_for_eviction:
                        page.manager.exclude_from_eviction(page)
                    assert not page.scheduled_for_eviction
                else:
                    warnings.warn(
                        "[KVCacheManager] Block is being deleted, but its pages are still in use!"
                    )

    def _partial_match_this_node(self, tokens: TokenBlock) -> int:
        """
        Returns the number of leading tokens that match between the given tokens and this block's tokens.
        """
        assert len(tokens) <= len(self.tokens), "too many tokens"
        for i, (a, b) in enumerate(zip(tokens, self.tokens)):
            if a != b:
                return i
        return len(tokens)

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return LifeCycleId(len(self.storage))

    @property
    def prev(self) -> 'Block | RootBlock':
        return unwrap_weakref(self._prev)

    def remove_if_unusable(self) -> None:
        warnings.warn(
            "[KVCacheManager] Not Implemented: Block.remove_if_unusable() is currently just a naive implementation."
        )
        has_empty_storage = lambda b: all(page is None for page in b.storage)
        if all(has_empty_storage(b) for b in traverse_subtree(self)):
            remove_subtree(self)

    @property
    def tokens_per_block(self) -> int:
        # we assume non-leaf blocks are always full.
        prev = self.prev
        return prev.tokens_per_block if isinstance(prev, RootBlock) else len(
            prev.tokens)

    @property
    def is_full(self) -> bool:
        return len(self.tokens) == self.tokens_per_block

    @property
    def is_orphan(self) -> bool:
        return self.key not in self.prev.next


class BlockRadixTree:
    __slots__ = ('_life_cycles', '_tokens_per_block', 'next', '__weakref__')
    _life_cycles: weakref.ref[LifeCycleRegistry]
    _tokens_per_block: int
    next: Children[RootBlock]

    def __init__(self, life_cycles: LifeCycleRegistry, tokens_per_block: int):
        self._life_cycles = weakref.ref(life_cycles)
        self._tokens_per_block = tokens_per_block
        self.next = {}

    def add_or_get_existing(self, lora_task_id: int | None) -> RootBlock:
        key = RootBlock.make_key(lora_task_id)
        if key in self.next:
            return self.next[key]
        return RootBlock(lora_task_id, self)

    @property
    def tokens_per_block(self) -> int:
        return self._tokens_per_block

    @property
    def life_cycles(self) -> LifeCycleRegistry:
        return unwrap_weakref(self._life_cycles)

    @property
    def num_life_cycles(self) -> LifeCycleId:
        return self.life_cycles.size

    def clear(
            self,
            yield_pages: bool = False
    ) -> Iterator[weakref.ref['CommittedPage']]:
        # taking O(1) space
        # remove leaf blocks one by one, in post-order
        while self.next:
            block = next(iter(self.next.values()))
            yield from remove_subtree(block, yield_pages)
        assert not self.next

    # yields tuples of (block, num_matched_tokens). num_matched_tokens should be equal to tokens_per_block except the last one.
    def match(
            self,
            lora_task_id: int | None,
            tokens: Sequence[TokenIdExt],
            enable_partial_match: bool = False) -> Iterator[tuple[Block, int]]:
        block = self
        mismatched_token_block = ()
        for token_block, key in sequence_to_blockchain_keys(
                self._tokens_per_block, lora_task_id, tokens):
            if key in block.next:
                block = block.next[key]
                if token_block:
                    assert isinstance(block, Block)
                    yield block, len(token_block)
            else:
                mismatched_token_block = token_block
                break
        if mismatched_token_block and enable_partial_match:
            block, match_len = find_best_partial_match_in_next_nodes(
                cast(Block | RootBlock, block), mismatched_token_block)
            if block is not None:
                yield block, match_len

    def _check_sanity(self) -> bool:
        raise NotImplementedError(
            "[KVCacheManager] Check if there are any unusable blocks that should have been removed."
        )
