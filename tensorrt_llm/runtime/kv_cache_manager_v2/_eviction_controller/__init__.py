import enum


class PageStatus(enum.IntEnum):
    LOCKED = 0  # Required in GPU. Eviction/dropping not allowed
    HELD = 1  # Allow eviction but not dropping
    DROPPABLE = 2  # Allow eviction and dropping


from ._eviction_controller import Candidate, EvictionController, EvictionPolicy

__all__ = ['EvictionController', 'Candidate', 'EvictionPolicy']
