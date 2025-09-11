import array
import atexit
import ctypes
import functools
import operator
import os
import traceback
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from ctypes.util import find_library
from itertools import pairwise
from typing import (Any, BinaryIO, Callable, ClassVar, Generic, Iterable,
                    Iterator, MutableSequence, Protocol, Sequence, Type,
                    TypeVar, cast)

import cuda.bindings.driver as drv

from ._exceptions import (CuError, CuOOMError, DiskOOMError, HostOOMError,
                          LogicError)


def _unwrap(ret: drv.CUresult | tuple[drv.CUresult, Any]
            | tuple[drv.CUresult, Any, Any]):
    if isinstance(ret, drv.CUresult):
        if ret != drv.CUDA_SUCCESS:
            if ret == drv.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
                raise CuOOMError()
            raise CuError(ret)
    else:
        _unwrap(ret[0])
        return ret[1:]


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    return div_up(x, y) * y


def round_down(x: int, y: int) -> int:
    return x // y * y


def in_range(x: int, lower: int, upper: int) -> bool:
    return lower <= x < upper


T = TypeVar('T')
U = TypeVar('U')
Index = TypeVar('Index', bound=int, contravariant=True)
Row = TypeVar('Row', bound=int, contravariant=True)
Col = TypeVar('Col', bound=int, contravariant=True)


def unwrap_optional(value: T | None) -> T:
    if value is None:
        raise ValueError("Expected non-None value")
    return value


def unwrap_weakref(value: weakref.ref[T]) -> T:
    return unwrap_optional(value())


def coalesce(value: T | None, fallback: T) -> T:
    return value if value is not None else fallback


def remove_if(original: MutableSequence[T],
              predicate: Callable[[T], bool]) -> list[T]:
    'Remove items from original that satisfy the predicate and return the removed items.'
    removed = []
    for idx, item in enumerate(original):
        if predicate(item):
            removed.append(item)
        else:
            original[idx - len(removed)] = item
    del original[len(original) - len(removed):]
    return removed


def partition(original: Iterable[T],
              classifier: Callable[[T], U]) -> defaultdict[U, list[T]]:
    ret = defaultdict(list)
    for item in original:
        ret[classifier(item)].append(item)
    return ret


def get_uniform_attribute(iterable: Iterable[T],
                          attribute_func: Callable[[T], U]) -> U:
    ret = attribute_func(next(iter(iterable)))
    assert all(attribute_func(item) == ret for item in iterable)
    return ret


def noexcept(func: Callable[..., Any]) -> Callable[..., Any]:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise AssertionError(
                f"Function {func.__name__} raised an exception: {e}") from e

    return wrapper


def expect_type(value: Any, ExpectedType: Type[T]) -> T:
    if not isinstance(value, ExpectedType):
        raise ValueError(
            f"Expected {ExpectedType.__name__}, got {type(value).__name__}")
    return value


def is_sorted(iterable: Iterable[T],
              key: Callable[[T], Any] = lambda x: x,
              reverse: bool = False) -> bool:
    comp = operator.ge if reverse else operator.le
    return all(comp(key(a), key(b)) for a, b in pairwise(iterable))


HomoTuple = tuple[T, ...]


class TypedIndexList(Protocol[Index, T]):

    def __getitem__(self, index: Index) -> T:
        ...

    def __setitem__(self, index: Index, value: T) -> None:
        ...

    def __iter__(self) -> Iterator[T]:
        ...

    def __len__(self) -> int:
        ...

    def __reversed__(self) -> Iterator[T]:
        ...


class Array2D(Generic[Row, Col, T]):
    __slots__ = ('_data', '_cols')
    _data: list[T]
    _cols: int

    def __init__(self, rows: int, cols: int, init_val: Iterable[T]):
        self._data = list(init_val)
        self._cols = cols

    def __getitem__(self, index: tuple[Row, Col]) -> T:
        return self._data[index[0] * self._cols + index[1]]

    def __setitem__(self, index: tuple[Row, Col], value: T) -> None:
        self._data[index[0] * self._cols + index[1]] = value

    @property
    def rows(self) -> int:
        assert len(self._data) % self._cols == 0
        return len(self._data) // self._cols

    def row(self, row: Row) -> TypedIndexList[Col, T]:
        return cast(TypedIndexList[Col, T],
                    self._data[row * self._cols:(row + 1) * self._cols])

    def col(self, col: Col) -> TypedIndexList[Row, T]:
        return cast(TypedIndexList[Row, T], self._data[col::self._cols])

    @property
    def cols(self) -> int:
        return self._cols

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._data)


def typed_range(count: Index) -> Iterator[Index]:
    return (cast(Index, i) for i in range(int(count)))


mem_alignment = 2 << 20  # 2MB

_libc = ctypes.CDLL(find_library('c'))


def _aligned_alloc(alignment: int, size: int) -> int:
    """
    Allocates size bytes of uninitialized storage whose alignment is specified by alignment.
    Returns the address as an integer.
    Raises HostOOMError on failure.
    """
    assert size % alignment == 0
    memptr = _libc.aligned_alloc(alignment, size)
    if memptr == ctypes.c_void_p(0):
        raise HostOOMError("aligned_alloc failed")
    assert memptr.value is not None and memptr.value != 0
    return memptr.value


def _memadvise(ptr: int, size: int, advice: int):
    if os.name == "nt":
        return
    ret = _libc.madvise(ptr, size, advice)
    if ret != 0:
        raise HostOOMError("memadvise failed")


MADV_HUGEPAGE = 14


def _realloc(ptr: int, size: int) -> int:
    """
    Reallocates size bytes of storage whose alignment is specified by alignment.
    Returns the address as an integer.
    Raises OSError on failure.
    """
    ret = _libc.realloc(ptr, size)
    if ret == ctypes.c_void_p(0):
        raise HostOOMError("realloc failed.")
    return ret


class HostMem:
    """
    Host memory aligned to 2MB, reallocable for low-cost resizing and registered to CUDA as page-locked memory.
    Resizing will keep the original memory content, like `realloc` in C.
    """
    __slots__ = ('_address', '_size')
    _address: int
    _size: int

    @property
    def address(self) -> int:
        return self._address

    @property
    def size(self) -> int:
        return self._size

    def __init__(self, size: int):
        if size == 0:
            self._address = 0
            self._size = 0
            return
        self._address = _aligned_alloc(mem_alignment, size)
        self._size = size
        _memadvise(self._address, size, MADV_HUGEPAGE)

    def resize(self, new_size: int):
        self._unregister_from_cuda()
        try:
            self._address = _realloc(self._address, new_size)
            self._size = new_size
            _memadvise(self._address, new_size, MADV_HUGEPAGE)
        finally:
            self._register_to_cuda()

    def destroy(self):
        self._unregister_from_cuda()
        _libc.free(self._address)
        self._address = 0
        self._size = 0

    def __del__(self):
        if self._address != 0:
            self.destroy()

    def _register_to_cuda(self):
        _unwrap(
            drv.cuMemHostRegister(
                self._address, self._size, drv.CU_MEMHOSTREGISTER_PORTABLE
                | drv.CU_MEMHOSTREGISTER_DEVICEMAP))

    def _unregister_from_cuda(self):
        _unwrap(drv.cuMemHostUnregister(self._address))


def _posix_fallocate(fd: int, offset: int, length: int):
    ret = _libc.posix_fallocate(fd, offset, length)
    if ret != 0:
        raise DiskOOMError(ret, "posix_fallocate failed")


def get_file_size(file: BinaryIO) -> int:
    pos = file.tell()
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(pos)
    return size


def resize_file(file: BinaryIO, new_size: int):
    old_size = get_file_size(file)
    if new_size > old_size:
        _posix_fallocate(file.fileno(), old_size, new_size - old_size)
    elif new_size < old_size:
        file.truncate(new_size)


class DynamicBitset:
    """
    A memory efficient bitset that can be resized.
    """
    __slots__ = ('_bits', '_num_set_bits')
    _bits: array.array
    _num_set_bits: int

    TYPE_CODE = 'Q'
    ALL_SET_MASK = (1 << 64) - 1

    def __init__(self, capacity: int):
        self._bits = array.array(self.TYPE_CODE, [0] * ((capacity + 63) // 64))
        self._num_set_bits = 0

    def set(self, index: int):
        if not self.get(index):
            self._bits[index // 64] |= 1 << (index % 64)
            self._num_set_bits += 1

    def get(self, index: int) -> bool:
        return self._bits[index // 64] & (1 << (index % 64)) != 0

    def clear(self, index: int):
        if self.get(index):
            self._bits[index // 64] &= ~(1 << (index % 64))
            self._num_set_bits -= 1

    @property
    def num_set_bits(self) -> int:
        return self._num_set_bits

    def resize(self, new_capacity: int):
        extra_elems = div_up(new_capacity, 64) - len(self._bits)
        if extra_elems > 0:
            self._bits.extend(array.array(self.TYPE_CODE, [0] * extra_elems))
        elif extra_elems < 0:
            self._bits = self._bits[:extra_elems]

    # check if any bit in the range [start, end) is set
    def any_set(self, start: int, end: int) -> bool:
        if start >= end:
            return False
        start_word_mask = self.ALL_SET_MASK << (start % 64)
        end_word_mask = self.ALL_SET_MASK >> (64 - (end % 64))
        if start // 64 == end // 64:
            if (start_word_mask & end_word_mask & self._bits[start // 64]) != 0:
                return True
        else:
            if (start_word_mask & self._bits[start // 64]) != 0 or (
                    end_word_mask & self._bits[end // 64]) != 0:
                return True
        return any(self._bits[i] != 0
                   for i in range(start // 64 + 1, end // 64))


class SimplePool(Generic[T]):
    __slots__ = ('_create_func', '_destroy_func', '_items', '_max_size',
                 '_outstanding_count')
    _create_func: Callable[[], T]
    _destroy_func: Callable[[T], None]
    _items: deque[T]
    _max_size: int | None
    _outstanding_count: int  # number of items currently we gave out but not returned, i.e. get() but not put()

    def __init__(self,
                 create_func: Callable[[], T],
                 destroy_func: Callable[[T], None],
                 init_size: int = 0,
                 max_size: int | None = None):
        self._create_func = create_func
        self._destroy_func = destroy_func
        self._items = deque[T]((create_func() for _ in range(init_size)),
                               maxlen=max_size)
        self._max_size = max_size

    def clear(self):
        while True:
            self._destroy_func(self._items.popleft())

    def __del__(self):
        self.clear()

    def get(self) -> T:
        ret = self._items.popleft() if self._items else self._create_func()
        self._outstanding_count += 1
        return ret

    def put(self, item: T):
        self._outstanding_count -= 1
        if self._max_size is not None and len(self._items) >= self._max_size:
            self._destroy_func(item)
        self._items.appendleft(item)

    @property
    def outstanding_count(self) -> int:
        'number of items currently we get() but not put()'
        return self._outstanding_count

    @property
    def cached_count(self) -> int:
        'number of items currently in the pool'
        return len(self._items)

    @property
    def total_count(self) -> int:
        'total number of items created, including both outstanding and cached'
        return self.outstanding_count + self.cached_count


class GlobalPoolProvider(Generic[T]):
    __slots__ = ()
    _pool: ClassVar[SimplePool]

    @classmethod
    def register_pool(cls, pool: SimplePool[T]):
        cls._pool = pool
        atexit.register(cls.clear_pool)

    @classmethod
    def clear_pool(cls):
        if cls._pool is not None:
            cls._pool.clear()

    def pool(self) -> SimplePool[T]:
        return self._pool


class ItemHolderBase(Generic[T], ABC):
    __slots__ = ('_item', )
    _item: T | None

    def __init__(self):
        self._item = self.pool().get()

    def close(self):
        if not self.is_closed():
            self.pool().put(self._item)  # type: ignore
            self._item = None

    def __del__(self):
        self.close()

    def is_closed(self) -> bool:
        return self._item is None

    def get(self) -> T:
        assert not self.is_closed()
        return self._item  # type: ignore

    @property
    def handle(self) -> T:
        return self.get()

    @abstractmethod
    def pool(self) -> SimplePool[T]:
        ...


class ItemHolderWithGlobalPool(GlobalPoolProvider[T], ItemHolderBase[T]):
    __slots__ = ()


class CachedCudaEvent(ItemHolderWithGlobalPool[drv.CUevent]):
    """
    A cached CUDA event without support for timing. Recorded to a stream when created.
    """
    NULL: ClassVar['_NullCudaEvent']

    __slots__ = ()

    def __init__(self, stream: drv.CUstream):
        super().__init__()
        self._record(stream)

    def query_complete(self) -> bool:
        """
        Query the event. If complete, also close the event. Closed events are always considered complete.
        """
        if self.is_closed():
            return True
        err = drv.cuEventQuery(self.get())
        if err == drv.CUDA_SUCCESS:
            self.close()
            return True
        elif err == drv.CUDA_ERROR_NOT_READY:
            return False
        else:
            raise CuError(err)

    def synchronize(self):
        if self.is_closed():
            return
        _unwrap(drv.cuEventSynchronize(self.get()))
        self.close()

    def wait_in_stream(self, stream: drv.CUstream):
        if self.is_closed():
            return
        _unwrap(
            drv.cuStreamWaitEvent(stream, self.get(),
                                  drv.CU_STREAM_WAIT_VALUE_COMPLETED))

    def _record(self, stream: drv.CUstream):
        """
        Prefer new event instead of recording an existing event.
        """
        _unwrap(drv.cuEventRecord(self.get(), stream))


CachedCudaEvent.register_pool(SimplePool[drv.CUevent](
    lambda: _unwrap(drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DISABLE_TIMING)
                    ),
    lambda ev: _unwrap(drv.cuEventDestroy(ev)),  # type: ignore[arg-type]
    init_size=1024))


class _NullCudaEvent(CachedCudaEvent):
    """
    A null CUDA event that is closed (and always complete).
    """
    __slots__ = ()

    def __init__(self):
        # do not call super().__init__(). We don't need an event here.
        self._item = None


CachedCudaEvent.NULL = _NullCudaEvent()


class CachedCudaStream(ItemHolderWithGlobalPool[drv.CUstream]):
    """
    A cached non-blocking CUDA stream.
    """
    __slots__ = ()

    def wait_event(self, event: drv.CUevent) -> None:
        _unwrap(
            drv.cuStreamWaitEvent(self.get(), event,
                                  drv.CU_STREAM_WAIT_VALUE_COMPLETED))

    def wait_events(
            self,
            events: Sequence[CachedCudaEvent] | set[CachedCudaEvent]) -> None:
        '''
        Wait for events with deduplication first.
        '''
        for ev in (set(events) if isinstance(events, Sequence) else events):
            ev.wait_in_stream(self.get())

    def record_event(self) -> CachedCudaEvent:
        return CachedCudaEvent(self.get())


CachedCudaStream.register_pool(SimplePool[drv.CUstream](
    lambda: _unwrap(
        drv.cuStreamCreate(drv.CUstream_flags.CU_STREAM_NON_BLOCKING)),
    lambda stream: _unwrap(drv.cuStreamDestroy(stream)
                           ),  # type: ignore[arg-type]
    init_size=128))


class TemporaryCudaStream(CachedCudaStream):
    """
    A cached non-blocking CUDA stream. Mainly used as temporary worker streams.
    Requires a list of prior events to wait for dependencies, and a finish event must be recorded before closing.
    """
    __slots__ = ('_finish_event_recorded', )
    _finish_event_recorded: bool

    def __init__(self, prior_events: Sequence[CachedCudaEvent]
                 | set[CachedCudaEvent]):
        super().__init__()
        self.wait_events(prior_events)
        self._finish_event_recorded = False

    def close(self):
        if not self._finish_event_recorded:
            raise LogicError("finish event not recorded" +
                             "".join(traceback.format_stack()))
        super().close()

    def finish(self) -> CachedCudaEvent:
        if self._finish_event_recorded:
            raise LogicError("finish event already recorded")
        self._finish_event_recorded = True
        return self.record_event()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def merge_events(
    events: Sequence[CachedCudaEvent] | set[CachedCudaEvent]
) -> CachedCudaEvent:
    if len(events) == 0:
        return CachedCudaEvent.NULL
    if len(events) == 1:
        ev = next(iter(events))
        return ev if not ev.is_closed() else CachedCudaEvent.NULL
    with TemporaryCudaStream(events) as stream:
        return stream.finish()


class SharedPoolProvider(Generic[T]):
    __slots__ = ('_pool', )
    _pool: SimplePool[T]

    def __init__(self, pool: SimplePool[T]):
        self._pool = pool

    def pool(self) -> SimplePool[T]:
        return self._pool


class ItemHolderWithSharedPool(SharedPoolProvider[T], ItemHolderBase[T]):
    __slots__ = ()

    def __init__(self, pool: SimplePool[T]):
        SharedPoolProvider.__init__(self, pool)
        ItemHolderBase.__init__(self)


HolderT = TypeVar('HolderT', bound=ItemHolderWithSharedPool)


# For subclassing if holder needs to be customized
class PooledFactoryBase(Generic[T, HolderT]):
    _Holder: Type[HolderT]  # subclasses must initialize this static attribute
    __slots__ = ('_pool', )
    _pool: SimplePool[T]

    def __init__(self,
                 create_func: Callable[[], T],
                 destroy_func: Callable[[T], None],
                 init_size: int = 0,
                 max_cache_size: int | None = None):
        self._pool = SimplePool[T](create_func, destroy_func, init_size,
                                   max_cache_size)

    def create(self) -> HolderT:
        return self._Holder(self._pool)

    def clear(self):
        self._pool.clear()


# For directly use
class PooledFactory(PooledFactoryBase[T, ItemHolderWithSharedPool]):
    _Holder = ItemHolderWithSharedPool
    __slots__ = ()

    def __init__(self,
                 create_func: Callable[[], T],
                 destroy_func: Callable[[T], None],
                 init_size: int = 0,
                 max_cache_size: int | None = None):
        super().__init__(create_func, destroy_func, init_size, max_cache_size)


def query_total_gpu_memory() -> int:
    _, total = _unwrap(drv.cuMemGetInfo())  # type: ignore[assignment]
    return total


def query_free_gpu_memory() -> int:
    free, _ = _unwrap(drv.cuMemGetInfo())  # type: ignore[assignment]
    return free
