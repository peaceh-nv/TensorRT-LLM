import itertools
import time
import unittest
from random import randbytes
from statistics import median
from typing import NamedTuple

import cuda.bindings.runtime as cudart
from kv_cache_manager_v2 import (AttentionLayerConfig, BufferConfig,
                                 DiskCacheTierConfig, GpuCacheTierConfig,
                                 HostCacheTierConfig, KVCacheManager,
                                 KVCacheManagerConfig, LayerId, TokenId,
                                 TokenIdExt, _KVCache)
from kv_cache_manager_v2._utils import (TemporaryCudaStream, round_up,
                                        typed_range)
from kv_cache_manager_v2.tests.fake_engine import FakeEngine, Role
from parameterized import parameterized


class TestNaive(unittest.TestCase):
    cfg: KVCacheManagerConfig
    engine: FakeEngine
    manager: KVCacheManager

    def setUp(self):
        err, = cudart.cudaFree(0)
        assert int(err) == int(cudart.cudaError_t.cudaSuccess)
        self._init_cfg(gpu_quota=256 << 20,
                       host_quota=256 << 20,
                       disk_quota=1 << 30,
                       num_layers=36,
                       window_size=128,
                       sink_tokens=48)
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)

    def _init_cfg(self, gpu_quota: int, host_quota: int, disk_quota: int,
                  num_layers: int, window_size: int, sink_tokens: int):
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=32,
            vocab_size=4096,
            cache_tiers=[
                GpuCacheTierConfig(quota=gpu_quota),
                HostCacheTierConfig(quota=host_quota),
                DiskCacheTierConfig(quota=disk_quota, path="/workspace/"),
            ],
            layers=[
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(role=Role.KEY, size=8192),
                        BufferConfig(role=Role.VALUE, size=8192),
                        # BufferConfig(role=Role.KEY_BLOCK_QUANT, size=512),
                        # BufferConfig(role=Role.VALUE_BLOCK_QUANT, size=512),
                    ],
                    sliding_window_size=window_size if layer_id %
                    2 == 0 else None,
                    num_sink_tokens=sink_tokens if layer_id % 2 == 0 else None)
                for layer_id in typed_range(LayerId(num_layers))
            ])

    def run_naive(self, interval: int = 1) -> float:
        prompt_len = 1
        decode_len = 1024 * 10 - prompt_len

        class Request(NamedTuple):
            id: int
            kv_cache: _KVCache
            prompt: list[TokenIdExt]

        lora_task_id = None
        token_gen = itertools.count()
        prompt0 = [
            TokenId(next(token_gen)) if i != 100 else
            (next(token_gen), randbytes(32))[1] for i in range(prompt_len)
        ]
        req0 = Request(0, self.manager.create_kv_cache(lora_task_id, prompt0),
                       prompt0)
        with TemporaryCudaStream([]) as s:
            stream = s.handle
            req_id, kv_cache, prompt = req0
            success = kv_cache.resume(stream)
            assert success
            tic = time.perf_counter()
            # prefill
            num_reused = kv_cache.num_committed_tokens
            kv_cache.capacity = round_up(len(prompt), interval)
            capacity = kv_cache.capacity
            history = prompt[:num_reused]
            input = prompt[num_reused:]
            # self.engine.execute([Step(req_id, kv_cache, input, history)], stream)
            kv_cache.commit(input)
            history.extend(input)
            # decode
            for _ in range(decode_len):
                required_capacity = len(history) + 1
                if required_capacity > capacity:
                    kv_cache.commit(history[kv_cache.history_length:])
                    kv_cache.capacity = round_up(required_capacity, interval)
                    capacity = kv_cache.capacity
                input_token = TokenId(next(token_gen))
                # self.engine.execute([Step(req_id, kv_cache, [input_token], history)], stream)
                history.append(input_token)
            kv_cache.commit(history[kv_cache.history_length:])
            # last check
            # self.engine.execute([Step(req_id, kv_cache, [], history)], stream)
            toc = time.perf_counter()
            time_taken = toc - tic
            # print(f"Time taken: {time_taken} seconds")
        s.take_finish_event().synchronize()
        kv_cache.close()
        self.manager.clear_reusable_blocks()
        return time_taken

    @parameterized.expand([(2**i, ) for i in range(12)])
    def test_naive(self, interval):
        profiler = None
        if True:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        #time_taken = [self.run_naive(interval) for _ in range(11)]
        time_taken = [self.run_naive(32)]
        median_time_taken = median(time_taken)
        print(f"Median time taken: {median_time_taken}")
        if profiler is not None:
            profiler.disable()
            profiler.print_stats(sort='cumtime')
            profiler.dump_stats('profiler.prof')


if __name__ == "__main__":
    unittest.main()
