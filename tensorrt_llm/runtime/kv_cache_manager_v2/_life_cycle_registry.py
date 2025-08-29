from typing import NamedTuple

from ._common import SlidingWinSize
from ._config import KVCacheManagerConfig


class LifeCycle(NamedTuple):
    sliding_win_size: SlidingWinSize


class LifeCycleId(int):
    pass


class LifeCycleRegistry:
    __slots__ = ('life_cycle_list', 'life_cycle_id_dict')
    life_cycle_list: list[LifeCycle]
    life_cycle_id_dict: dict[LifeCycle, LifeCycleId]

    def __init__(self, config: KVCacheManagerConfig):
        for layer in config.layers:
            details = LifeCycle(sliding_win_size=layer.sliding_win_size)
            if details not in self.life_cycle_id_dict:
                assert len(self.life_cycle_id_dict) == len(
                    self.life_cycle_list), "corrupted life cycle registry"
                self.life_cycle_list.append(details)
                self.life_cycle_id_dict[details] = LifeCycleId(
                    len(self.life_cycle_list) - 1)

    def get_life_cycle(self, id: LifeCycleId) -> LifeCycle:
        return self.life_cycle_list[id]

    def get_id(self, life_cycle_details: LifeCycle) -> LifeCycleId:
        return self.life_cycle_id_dict[life_cycle_details]
