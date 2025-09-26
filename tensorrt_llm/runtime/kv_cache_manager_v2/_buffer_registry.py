from typing import cast

from ._common import MirroredBufGroupId
from ._storage._config import BufferId
from ._storage._core import PoolIndex
from ._utils import TypedIndexList, typed_len


class BufferRegistry:
    _mirrored_buffer_groups: TypedIndexList[MirroredBufGroupId,
                                            TypedIndexList[PoolIndex, BufferId]]
    _buffer_to_mirrored_index: dict[BufferId, MirroredBufGroupId]

    def __init__(self):
        self._mirrored_buffer_groups = cast(
            TypedIndexList[MirroredBufGroupId, TypedIndexList[PoolIndex,
                                                              BufferId]], [])
        self._buffer_to_mirrored_index = dict[BufferId, MirroredBufGroupId]()

    def register_mirrored_buffers(self, buffers: TypedIndexList[PoolIndex,
                                                                BufferId]):
        index = self.num_mirrored_buffer_groups
        self._mirrored_buffer_groups.append(buffers)
        for buffer in buffers:
            assert buffer not in self._buffer_to_mirrored_index
            self._buffer_to_mirrored_index[buffer] = index

    @property
    def num_mirrored_buffer_groups(self) -> MirroredBufGroupId:
        return typed_len(self._mirrored_buffer_groups)

    def get_mirrored_buffer_group(
            self,
            index: MirroredBufGroupId) -> TypedIndexList[PoolIndex, BufferId]:
        return self._mirrored_buffer_groups[index]

    def get_mirrored_buffer_group_index(self,
                                        buffer: BufferId) -> MirroredBufGroupId:
        return self._buffer_to_mirrored_index[buffer]
