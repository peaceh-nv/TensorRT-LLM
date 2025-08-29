import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

import cuda.bindings.driver as drv


# Can extend to more levels in the future, e.g. object storage like AWS S3.
class CacheLevel(enum.IntEnum):
    GPU_MEM = 0
    HOST_MEM = 1
    DISK = 2


# Normal token id that falls in the tokenizer vocabulary.
TokenId = int

# For multi-modal tokens, we can handle it in either of the following ways:
#   1. Hash combine image digest and local_token_id, then use digest for every multi-modal token.
#   2. Use digest only for the first multi-modal token, and use int(vocab_size + local_token_id) for the rest.
#   3. Hash the multi-modal token embedding data and use the digest as TokenIdExt for every multi-modal token. If we do this, we can't skip the encoder.
TokenIdExt = TokenId | bytes

LayerId = int
if TYPE_CHECKING:

    class CudaStream(int):
        ...
else:
    CudaStream = drv.CUstream

UserId = int

MemAddress = int


@dataclass(slots=True)
class DiskAddress:
    file: BinaryIO
    offset: int


Address = MemAddress | DiskAddress

SlidingWinSize = int | None


class Priority(int):
    __slots__ = ()
    _DEFAULT_VALUE = 35
    _MIN_VALUE = 0
    _MAX_VALUE = 100

    def __new__(cls, value: int):
        if value < cls._MIN_VALUE or value > cls._MAX_VALUE:
            raise ValueError(
                f"Priority must be between {cls._MIN_VALUE} and {cls._MAX_VALUE}, got {value}"
            )
        return super().__new__(cls, value)

    def __int__(self):
        return super().__int__()

    def __str__(self):
        return f"Priority({super().__int__()})"

    @classmethod
    def _init_constants(cls):
        cls.DEFAULT = cls(cls._DEFAULT_VALUE)
        cls.MIN = cls(cls._MIN_VALUE)
        cls.MAX = cls(cls._MAX_VALUE)


Priority._init_constants()
