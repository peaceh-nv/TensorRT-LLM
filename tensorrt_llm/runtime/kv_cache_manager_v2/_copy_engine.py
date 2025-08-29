import atexit
from typing import NamedTuple

from ._common import Address, CacheLevel, CudaStream


class CopyTask(NamedTuple):
    dst: Address
    src: Address


class _CopyEngine:

    def close(self):
        pass

    def transfer(self, dst_cache_level: CacheLevel, src_cache_level: CacheLevel,
                 num_bytes: int, tasks: list[CopyTask], stream: CudaStream):
        pass


_copy_engine = _CopyEngine()
atexit.register(_copy_engine.close)


def batched_copy(dst_cache_level: CacheLevel, src_cache_level: CacheLevel,
                 num_bytes: int, tasks: list[CopyTask],
                 stream: CudaStream) -> None:
    _copy_engine.transfer(dst_cache_level, src_cache_level, num_bytes, tasks,
                          stream)
