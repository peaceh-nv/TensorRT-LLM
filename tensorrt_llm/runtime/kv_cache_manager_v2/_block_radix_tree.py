import hashlib
import weakref
from typing import Iterable, Iterator, Sequence

from ._common import BlockOrdinal, TokenIdExt
from ._core._kv_cache_manager import KVCacheManager
from ._life_cycle_registry import LifeCycleRegistry
from ._page import CommittedPage
from ._utils import unwrap_weakref

BlockKey = bytes


# id_offset is usually vocab_size
def gen_multi_modal_tokens(id_offset: int, multi_modal_data_digest: bytes,
                           num_tokens: int) -> list[TokenIdExt]:
    assert num_tokens > 0
    # Alternatively, we could also use (multi_modal_data_digest + i.to_bytes(8, 'little')) or its hash digest as token id.
    # The implementation below is faster and also works because KV cache reuse of a token is with a precondition that all previous tokens also match. So only the first multi-modal token id needs to be unique.
    return [
        multi_modal_data_digest if i == 0 else id_offset + i
        for i in range(num_tokens)
    ]


class Hasher:
    __slots__ = ('_hasher')
    _hasher: hashlib._Hash

    def __init__(self, data: int | bytes | None | Iterable = None):
        self._hasher = hashlib.sha256()
        self.update(data)

    def update(self, data: int | bytes | None | Iterable) -> 'Hasher':
        if isinstance(data, int):
            assert data >= 0 and data < (1 << 64)
            self._hasher.update(data.to_bytes(8, 'little'))
        elif isinstance(data, bytes):
            self._hasher.update(data)
        elif data is None:
            pass
        elif isinstance(data, Iterable):
            for item in data:
                self.update(item)
        else:
            raise ValueError(f"Unsupported type: {type(data)}")
        return self

    @property
    def digest(self) -> bytes:
        return self._hasher.digest()


TokenBlock = Sequence[TokenIdExt]


def sequence_to_blockchain_keys(
        tokens_per_block: int, lora_task_id: int | None,
        tokens: Sequence[TokenIdExt]) -> Iterator[tuple[TokenBlock, BlockKey]]:
    digest = Hasher(lora_task_id).digest
    for i in range(0, len(tokens), tokens_per_block):
        token_block = tokens[i:i + tokens_per_block]
        yield token_block, Hasher((digest, token_block)).digest


class ChildrenDict(dict[BlockKey, 'Block']):
    __slots__ = ()

    def remove_child_tree(self, child_key: BlockKey):
        # taking O(1) space
        # remove leaf blocks one by one, in post-order
        block: Block = self[child_key]
        while True:
            if block.next:
                block = next(iter(block.next.values()))
            else:
                assert all(
                    page is None or page() is None
                    for page in block.storage), "Storage is not cleared, yet"
                prev_block: Block | None = block.prev(
                ) if block.prev is not None else None
                if prev_block is None:
                    break
                prev_block.next.pop(block.key)
                block = prev_block

    def _find_best_partial_match_in_next_nodes(
            self, tokens: TokenBlock) -> tuple['Block | None', int]:
        """
        Among all child nodes (self.next), finds the one whose tokens have the longest leading match with the given tokens.
        Returns a tuple of (best_block, num_matched_tokens).
        If no child matches any tokens, returns (None, 0).
        """
        # @todo: when we have many children, we need to build a database to accelerate partial matching.
        best_block = None
        best_match_len = 0
        for block in self.values():
            match_len = block._partial_match_this_node(tokens)
            if match_len > best_match_len:
                best_match_len = match_len
                best_block = block
        return best_block, best_match_len


class RootPrevInfo:
    __slots__ = ('_manager', 'key')
    _manager: weakref.ref[KVCacheManager]
    key: BlockKey  # hash of lora_task_id
    ordinal: BlockOrdinal = -1


class Block:
    """
    A block of tokens. Manages data for all layers.
    """
    __slots__ = ('_manager', 'key', 'tokens', 'ordinal', 'prev', 'next',
                 'storage')
    _manager: weakref.ref[KVCacheManager]
    key: BlockKey
    tokens: Sequence[TokenIdExt]
    ordinal: BlockOrdinal
    prev: weakref.ref['Block'] | None
    next: ChildrenDict

    # indexed with LifeCycleId
    storage: list[weakref.ref[CommittedPage] | None]

    def __init__(self, tokens: Sequence[TokenIdExt],
                 prev: 'Block | RootPrevInfo'):
        self._manager = prev._manager
        self.key = Hasher((prev.key, tokens)).digest
        self.tokens = tokens
        self.ordinal = prev.ordinal + 1
        if isinstance(prev, Block):
            self.prev = weakref.ref(prev)
            # prev.next keeps a strong ref to this _Block, so no need to remove self from prev.next in __del__().
            prev.next[self.key] = self
        else:
            assert isinstance(prev, RootPrevInfo)
            self.prev = None
        self.next = ChildrenDict()
        self.storage = []

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
    def manager(self) -> KVCacheManager:
        return unwrap_weakref(self._manager)


class BlockRadixTree:
    __slots__ = ('_tokens_per_block', '_root_blocks', '_life_cycles')
    _tokens_per_block: int
    _life_cycles: LifeCycleRegistry
    _root_blocks: ChildrenDict

    def __init__(self, tokens_per_block: int, life_cycles: LifeCycleRegistry):
        self._tokens_per_block = tokens_per_block
        self._life_cycles = life_cycles
        self._root_blocks = ChildrenDict()

    @property
    def life_cycles(self) -> LifeCycleRegistry:
        return self._life_cycles

    def clear(self):
        # taking O(1) space
        # remove leaf blocks one by one, in post-order
        for root_key in self._root_blocks:
            self._root_blocks.remove_child_tree(root_key)
        self._root_blocks.clear()

    @staticmethod
    def traverse_subtree(root: Block) -> Iterator[Block]:
        stack: list[Iterator[Block]] = []
        brother_iter: Iterator[Block] = iter((root, ))
        while True:
            block: Block | None = next(brother_iter, None)
            if block is None:
                if not stack:
                    break
                brother_iter = stack.pop()
            else:
                yield block
                if block.next:
                    stack.append(brother_iter)
                    brother_iter = iter(block.next.values())

    # yields tuples of (block, num_matched_tokens). num_matched_tokens should be equal to tokens_per_block except the last one.
    def match(
            self,
            lora_task_id: int | None,
            tokens: Sequence[TokenIdExt],
            enable_partial_match: bool = False) -> Iterator[tuple[Block, int]]:
        candidates: ChildrenDict = self._root_blocks
        mismatched_token_block = None
        for token_block, key in sequence_to_blockchain_keys(
                self._tokens_per_block, lora_task_id, tokens):
            if key in candidates:
                block = candidates[key]
                yield block, self._tokens_per_block
                candidates = block.next
            else:
                mismatched_token_block = token_block
                break
        if mismatched_token_block is not None and enable_partial_match:
            block, match_len = candidates._find_best_partial_match_in_next_nodes(
                mismatched_token_block)
            if block is not None:
                yield block, match_len
