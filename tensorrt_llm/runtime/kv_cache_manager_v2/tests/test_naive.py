import time
import unittest
from random import randbytes, randint
from typing import NamedTuple

import cuda.bindings.runtime as cudart
from kv_cache_manager_v2 import (AttentionLayerConfig, BufferConfig,
                                 DiskCacheTierConfig, GpuCacheTierConfig,
                                 HostCacheTierConfig, KVCacheManager,
                                 KVCacheManagerConfig, LayerId, TokenId,
                                 TokenIdExt, _KVCache)
from kv_cache_manager_v2._utils import TemporaryCudaStream, typed_range
from kv_cache_manager_v2.tests.fake_engine import FakeEngine, Role, Step


class TestNaive(unittest.TestCase):
    cfg: KVCacheManagerConfig
    engine: FakeEngine
    manager: KVCacheManager

    def setUp(self):
        err, = cudart.cudaFree(0)
        assert int(err) == int(cudart.cudaError_t.cudaSuccess)
        self._init_cfg(gpu_quota=128 << 20,
                       host_quota=128 << 20,
                       disk_quota=1 << 30,
                       num_layers=8,
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
                        BufferConfig(role=Role.KEY_BLOCK_QUANT, size=512),
                        BufferConfig(role=Role.VALUE_BLOCK_QUANT, size=512),
                    ],
                    sliding_window_size=window_size if layer_id %
                    2 == 0 else None,
                    num_sink_tokens=sink_tokens if layer_id % 2 == 0 else None)
                for layer_id in typed_range(LayerId(num_layers))
            ])

    def test_naive(self):
        prompt_len = 1
        decode_len = 256 * 4

        class Request(NamedTuple):
            id: int
            kv_cache: _KVCache
            prompt: list[TokenIdExt]

        lora_task_id = None
        prompt0 = [
            TokenId(randint(0, self.cfg.vocab_size -
                            1)) if i != 100 else randbytes(32)
            for i in range(prompt_len)
        ]
        req0 = Request(0, self.manager.create_kv_cache(lora_task_id, prompt0),
                       prompt0)
        with TemporaryCudaStream([]) as s:
            stream = s.handle
            req_id, kv_cache, prompt = req0
            success = kv_cache.resume(stream)
            assert success
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
            tic = time.perf_counter()
            # prefill
            num_reused = kv_cache.num_committed_tokens
            kv_cache.capacity = len(prompt)
            history = prompt[:num_reused]
            input = prompt[num_reused:]
            self.engine.execute([Step(req_id, kv_cache, input, history)],
                                stream)
            kv_cache.commit(input)
            history.extend(input)
            # decode
            for _ in range(decode_len):
                input = [TokenId(randint(0, self.cfg.vocab_size - 1))]
                kv_cache.capacity = kv_cache.history_length + 1
                self.engine.execute([Step(req_id, kv_cache, input, history)],
                                    stream)
                kv_cache.commit(input)
                history.extend(input)
            # last check
            self.engine.execute([Step(req_id, kv_cache, [], history)], stream)
            toc = time.perf_counter()
            profiler.disable()
            profiler.print_stats(sort='cumtime')
            profiler.dump_stats('profiler.prof')
            print(f"Time taken: {toc - tic} seconds")
        s.take_finish_event().synchronize()
        kv_cache.close()
        self.manager.clear_reusable_blocks()


if __name__ == "__main__":
    unittest.main()
