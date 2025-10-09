import itertools
import time
import unittest
from random import randbytes
from statistics import median
from typing import Iterator, NamedTuple

import cuda.bindings.runtime as cudart
from kv_cache_manager_v2 import (AttentionLayerConfig, BufferConfig,
                                 DiskCacheTierConfig, GpuCacheTierConfig,
                                 HostCacheTierConfig, KVCacheManager,
                                 KVCacheManagerConfig, LayerId, TokenId,
                                 TokenIdExt, _KVCache)
from kv_cache_manager_v2._block_radix_tree import traverse_subtree
from kv_cache_manager_v2._eviction_controller import PageStatus
from kv_cache_manager_v2._exceptions import OutOfPagesError
from kv_cache_manager_v2._utils import (TemporaryCudaStream, round_up,
                                        typed_range, unwrap_weakref)
from kv_cache_manager_v2.tests.fake_engine import FakeEngine, Role, Step
from parameterized import parameterized


class TestNaive(unittest.TestCase):
    engine: FakeEngine
    cfg: KVCacheManagerConfig
    manager: KVCacheManager
    _token_id_gen: Iterator[int]

    def setUp(self):
        err, = cudart.cudaFree(0)
        assert int(err) == int(cudart.cudaError_t.cudaSuccess)
        self._token_id_gen = itertools.count()

    def next_token(self) -> TokenIdExt:
        token_id = next(self._token_id_gen)
        if token_id % 100 == 99:
            return randbytes(32)
        else:
            return TokenId(token_id)

    def prepare(self, gpu_quota: int, host_quota: int, disk_quota: int,
                num_layers: int, window_size: int, sink_tokens: int):
        self._init_cfg(gpu_quota, host_quota, disk_quota, num_layers,
                       window_size, sink_tokens)
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

    class Request(NamedTuple):
        id: int
        kv_cache: _KVCache
        prompt: list[TokenIdExt]
        decode_len: int

    def run_request(self, req: Request, interval: int, refcheck: bool) -> float:
        req_id, kv_cache, prompt, decode_len = req
        assert kv_cache.status == _KVCache.Status.ACTIVE
        stream = kv_cache.cuda_stream
        tic = time.perf_counter()
        # prefill
        num_reused = kv_cache.num_committed_tokens
        kv_cache.capacity = round_up(len(prompt), interval)
        capacity = kv_cache.capacity
        history = prompt[:num_reused]
        input = prompt[num_reused:]
        if refcheck:
            self.engine.execute([Step(req_id, kv_cache, input, history)],
                                stream)
        if input:
            kv_cache.commit(input)
            history.extend(input)
        # decode
        for _ in range(decode_len):
            required_capacity = len(history) + 1
            if required_capacity > capacity:
                kv_cache.commit(history[kv_cache.history_length:])
                kv_cache.capacity = round_up(required_capacity, interval)
                capacity = kv_cache.capacity
            input_token = self.next_token()
            if refcheck:
                self.engine.execute(
                    [Step(req_id, kv_cache, [input_token], history)], stream)
            history.append(input_token)
        kv_cache.commit(history[kv_cache.history_length:])
        # last check
        if refcheck:
            self.engine.execute([Step(req_id, kv_cache, [], history)], stream)
        toc = time.perf_counter()
        time_taken = toc - tic
        # print(f"Time taken: {time_taken} seconds")
        return time_taken

    def new_request(self, req_id: int, lora_task_id: int | None,
                    prompt_len: int, decode_len: int) -> Request:
        prompt = [self.next_token() for _ in range(prompt_len)]
        return self.Request(req_id,
                            self.manager.create_kv_cache(lora_task_id, prompt),
                            prompt, decode_len)

    def run_naive(self,
                  seq_len: int,
                  interval: int = 1,
                  refcheck: bool = True) -> float:
        prompt_len = 1
        decode_len = seq_len - prompt_len

        req_id = 0
        lora_task_id = None
        req0 = self.new_request(req_id, lora_task_id, prompt_len, decode_len)
        with TemporaryCudaStream([]) as s:
            stream = s.handle
            kv_cache = req0.kv_cache
            success = kv_cache.resume(stream)
            assert success
            time_taken = self.run_request(req0, interval, refcheck)

        s.take_finish_event().synchronize()
        kv_cache.close()
        self.manager.clear_reusable_blocks()
        return time_taken

    def test_sol_mem_utilization(self):
        self.prepare(8 << 20, 8 << 20, 1 << 30, 36, 128, 1)
        # if we have n blocks, we need 8192*2*18*(1+5+n) bytes of memory. For the (1+5+n), 1 is for sink blocks, 5 is for SWA (window=128), n is for full attention.
        max_seq_len = 32 * 22  # 23 blocks will require more than 8MB memory
        seq_len = max_seq_len

        # create a request and suspend it. It shall not consume any GPU memory after suspend.
        req0 = self.new_request(0, None, 256, seq_len - 256)
        with TemporaryCudaStream([]) as s:
            stream = s.handle
            success = req0.kv_cache.resume(stream)
            assert success
            self.run_request(req0, 32, False)
        s.take_finish_event()
        req0.kv_cache.suspend()

        # run another request that will take all the GPU memory
        req1 = self.new_request(0, None, 256, seq_len - 256)
        with TemporaryCudaStream([]) as s:
            stream = s.handle
            success = req1.kv_cache.resume(stream)
            assert success
            self.run_request(req1, 1, True)
        s.take_finish_event()

        req1.kv_cache.close()
        req0.kv_cache.close()

        # run another longer request and expect OutOfPagesError
        # This also tests eviction to disk.
        self.assertRaises(OutOfPagesError,
                          lambda: self.run_naive(seq_len + 1, 1, False))

    def test_cache_reuse(self):
        self.prepare(8 << 20, 8 << 20, 1 << 30, 36, 128, 1)
        # if we have n blocks, we need 8192*2*18*(1+5+n) bytes of memory. For the (1+5+n), 1 is for sink blocks, 5 is for SWA (window=128), n is for full attention.
        max_seq_len = 32 * 22  # 23 blocks will require more than 8MB memory
        seq_len = max_seq_len

        req0 = self.new_request(0, None, 256, seq_len - 256)
        with TemporaryCudaStream([]) as s:
            stream = s.handle
            success = req0.kv_cache.resume(stream)
            assert success
            self.run_request(req0, 32, False)
        s.take_finish_event()
        req0.kv_cache.close()

        for root_block in self.manager._radix_tree.next.values():
            for block0 in root_block.next.values():
                for block in traverse_subtree(block0):
                    for page in block.storage:
                        if page is not None:
                            assert unwrap_weakref(
                                page).status == PageStatus.DROPPABLE

        prompt1 = req0.kv_cache._committed_tokens[:(seq_len // 2 - 7)]
        req1 = self.Request(1, self.manager.create_kv_cache(None, prompt1),
                            prompt1, seq_len - len(prompt1))
        assert req1.kv_cache.num_committed_tokens == len(prompt1)
        with TemporaryCudaStream([]) as s:
            stream = s.handle
            success = req1.kv_cache.resume(stream)
            assert success
            self.run_request(req1, 32, False)
        s.take_finish_event()
        req1.kv_cache.close()

        self.manager.clear_reusable_blocks()

    def test_naive(self):
        self.prepare(256 << 20, 256 << 20, 1 << 30, 36, 128, 48)
        self.run_naive(512, 1, True)

    @parameterized.expand([(2**i, False) for i in range(12)])
    # @parameterized.expand([(32, True)])
    def test_naive_perf(self, interval, profile: bool):
        self.skipTest("Skipping perf test")
        self.prepare(256 << 20, 256 << 20, 1 << 30, 36, 128, 48)
        self.run_naive(10240, interval, False)  # warm up for numba jit
        profiler = None
        if profile:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        time_taken = [
            self.run_naive(10240, interval, False)
            for _ in range(11 if profiler is None else 1)
        ]
        median_time_taken = median(time_taken)
        print(f"Median time taken: {median_time_taken}")
        if profiler is not None:
            profiler.disable()
            profiler.print_stats(sort='cumtime')
            profiler.dump_stats('profiler.prof')


if __name__ == "__main__":
    unittest.main()
