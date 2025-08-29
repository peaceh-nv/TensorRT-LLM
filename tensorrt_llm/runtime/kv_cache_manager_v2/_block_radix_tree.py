import hashlib
import weakref
from typing import Iterator, Sequence

from ._common import TokenIdExt
from ._storage import Page

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


class _Block:
    """
    A block of tokens. Manages data for all layers.
    """
    _partial_match_index_build_threshold: int = 100  # if we have more child nodes (self.next) than this, we will build a index to accelerate partial matching.
    _partial_match_index_removal_threshold: int = 50  # if we have less child nodes (self.next) than this, we will remove the index.
    __slots__ = ('key', 'tokens', 'prev', 'next', 'storage')
    key: BlockKey
    tokens: Sequence[TokenIdExt]
    prev: weakref.ref['_Block'] | None
    next: dict[BlockKey, '_Block']

    # indexed with LifeCycleId
    storage: list[Page | None]

    def __init__(self, key: BlockKey, tokens: Sequence[TokenIdExt],
                 prev: '_Block | None'):
        self.key = key
        self.tokens = tokens
        if prev is not None:
            self.prev = weakref.ref(prev)
            # prev.next keeps a strong ref to this _Block, so no need to remove self from prev.next in __del__().
            prev.next[key] = self
        else:
            self.prev = None
        self.next = {}
        self.storage = []

    def drop_storage(self):
        for page in self.storage:
            if page is not None:
                page.drop()
        self.storage.clear()

    def _partial_match_this_node(self, tokens: Sequence[TokenIdExt]) -> int:
        """
        Returns the number of leading tokens that match between the given tokens and this block's tokens.
        """
        assert len(tokens) <= len(self.tokens), "too many tokens"
        for i, (a, b) in enumerate(zip(tokens, self.tokens)):
            if a != b:
                return i
        return len(tokens)

    def _find_best_partial_match_in_next_nodes(
            self, tokens: Sequence[TokenIdExt]) -> tuple['_Block | None', int]:
        """
        Among all child nodes (self.next), finds the one whose tokens have the longest leading match with the given tokens.
        Returns a tuple of (best_block, num_matched_tokens).
        If no child matches any tokens, returns (None, 0).
        """
        if len(self.next) > type(self)._partial_match_index_build_threshold:
            print(
                "Warning: should build index to accelerate partial matching, but this is not yet implemented."
            )
        best_block = None
        best_match_len = 0
        for block in self.next.values():
            match_len = block._partial_match_this_node(tokens)
            if match_len > best_match_len:
                best_match_len = match_len
                best_block = block
        return best_block, best_match_len


class _BlockRadixTree:
    tokens_per_block: int
    root_blocks: dict[BlockKey, _Block]

    def __init__(self, tokens_per_block: int):
        self.tokens_per_block = tokens_per_block
        self.root_blocks = dict[BlockKey, _Block]()

    def clear(self):
        for root_key in self.root_blocks:
            key = root_key
            block: _Block = self.root_blocks[key]
            while True:
                if len(block.next) > 0:
                    key = next(iter(block.next.keys()))
                    block = block.next[key]
                else:
                    block.drop_storage()
                    prev_block: _Block | None = block.prev(
                    ) if block.prev is not None else None
                    if prev_block is None:
                        break
                    block = prev_block
                    key = block.key
        self.root_blocks.clear()

    # yields tuples of (block, num_matched_tokens). num_matched_tokens should be equal to tokens_per_block except the last one.
    def match(
            self,
            lora_task_id: int | None,
            tokens: Sequence[TokenIdExt],
            enable_partial_match: bool = False) -> Iterator[tuple[_Block, int]]:
        child_blocks: dict[BlockKey, _Block] = self.root_blocks
        block = None
        for key in self._sequence_to_blockchain_keys(self.tokens_per_block,
                                                     lora_task_id, tokens):
            if key in child_blocks:
                block = child_blocks[key]
                yield block, self.tokens_per_block
            else:
                break
        if block is not None and enable_partial_match:
            block, match_len = block._find_best_partial_match_in_next_nodes(
                tokens)
            if block is not None:
                yield block, match_len

    @staticmethod
    def _sequence_to_blockchain_keys(
            tokens_per_block: int, lora_task_id: int | None,
            tokens: Sequence[TokenIdExt]) -> Iterator[BlockKey]:
        hasher = hashlib.sha256()
        if lora_task_id is not None:
            assert isinstance(lora_task_id, int)
            hasher.update(lora_task_id.to_bytes(8, 'little'))
        digest = hasher.digest()
        for i in range(0, len(tokens), tokens_per_block):
            hasher = hashlib.sha256(digest)
            block = tokens[i:i + tokens_per_block]
            for token in block:
                hasher.update(
                    token.to_bytes(8, 'little') if isinstance(token, int
                                                              ) else token)
            digest = hasher.digest()
            yield digest
