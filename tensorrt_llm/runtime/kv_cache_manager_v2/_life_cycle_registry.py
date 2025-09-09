from typing import NamedTuple, NewType

from ._common import SlidingWindowSize
from ._config import KVCacheManagerConfig
from ._utils import div_up


class LifeCycle(NamedTuple):
    sliding_win_size: SlidingWindowSize
    num_sink_blocks: int  # div_up(num_sink_tokens, tokens_per_block)

    @staticmethod
    def make(sliding_win_size: SlidingWindowSize, num_sink_tokens: int,
             tokens_per_block: int) -> 'LifeCycle':
        return LifeCycle(sliding_win_size,
                         div_up(num_sink_tokens, tokens_per_block))


LifeCycleId = NewType("LifeCycleId", int)


class LifeCycleRegistry:
    __slots__ = ('_life_cycle_list', '_life_cycle_id_dict')
    _life_cycle_list: list[LifeCycle]
    _life_cycle_id_dict: dict[LifeCycle, LifeCycleId]

    def __init__(self, config: KVCacheManagerConfig):
        for layer in config.layers:
            num_sink_blocks = div_up(layer.num_sink_tokens,
                                     config.tokens_per_block)
            details = LifeCycle(layer.sliding_window_size, num_sink_blocks)
            if details not in self._life_cycle_id_dict:
                assert len(self._life_cycle_id_dict) == len(
                    self._life_cycle_list), "corrupted life cycle registry"
                self._life_cycle_list.append(details)
                self._life_cycle_id_dict[details] = LifeCycleId(
                    len(self._life_cycle_list) - 1)

    def get_life_cycle(self, id: LifeCycleId) -> LifeCycle:
        return self._life_cycle_list[id]

    def get_id(self, life_cycle_details: LifeCycle) -> LifeCycleId:
        return self._life_cycle_id_dict[life_cycle_details]

    @property
    def num_life_cycles(self) -> int:
        assert len(self._life_cycle_list) == len(
            self._life_cycle_id_dict), "corrupted life cycle registry"
        return len(self._life_cycle_list)
