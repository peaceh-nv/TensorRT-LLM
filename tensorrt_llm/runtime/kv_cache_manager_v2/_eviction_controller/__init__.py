import enum


class PageStatus(enum.IntEnum):
    LOCKED = 0  # Required in GPU. Eviction/dropping not allowed
    EVICTABLE = 1  # Allow eviction to host or disk but not dropping
    UNRESTRICTED = 2  # Allow eviction and dropping
