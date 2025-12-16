from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional, Tuple

import torch
from strenum import StrEnum

from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy
from tensorrt_llm.mapping import CpType
from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError

from .llm_request import LlmRequest, LlmRequestState

RequestList = list[LlmRequest]

SchedulerOutput = namedtuple("SchedulerOutput", [
    "context_requests", "generation_requests", "paused_requests",
    "fitting_disagg_gen_init_requests", "num_fitting_requests"
])


class ScheduledRequests:
    # to be aligned with ScheduledRequests in cpp/tensorrt_llm/batch_manager/common.h
    def __init__(self):
        self.context_requests: RequestList = []
        self.generation_requests: RequestList = []
        self.paused_requests: RequestList = []

    @property
    def is_generation_only(self) -> bool:
        return (not self.context_requests and all(
            len(req.draft_tokens) == 0 for req in self.generation_requests))

    @property
    def can_run_cuda_graph(self) -> bool:
        return (not self.context_requests)

    @property
    def batch_size(self) -> int:
        return len(self.context_requests) + len(self.generation_requests)

    def all_requests(self) -> list[LlmRequest]:
        return self.context_requests + self.generation_requests


class RequestScheduler(ABC):

    @abstractmethod
    def schedule_request(self, active_requests: RequestList,
                         inflight_request_ids: set[int]) -> SchedulerOutput:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :param inflight_request_ids: set of request ids that are inflight (of all micro batches)
        :return: SchedulerOutput
        """
        # to be aligned with RequestScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/requestScheduler.h
        raise NotImplementedError


class CapacityScheduler(ABC):

    @abstractmethod
    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :return: (scheduledRequests, pausedRequests)
        """
        # to be aligned with CapacityScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/capacityScheduler.h
        raise NotImplementedError


class BindCapacityScheduler(CapacityScheduler):

    def __init__(
        self,
        max_num_requests: int,
        kv_cache_manager,
        peft_cache_manager: tb_internal.batch_manager.PeftCacheManager | None,
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.
        GUARANTEED_NO_EVICT,
        two_step_lookahead: bool = False,
    ):
        super(BindCapacityScheduler, self).__init__()
        self.kv_cache_manager = kv_cache_manager
        self.peft_cache_manager = peft_cache_manager

        self.impl = tb_internal.algorithms.CapacityScheduler(
            max_num_requests=max_num_requests,
            capacity_scheduler_policy=scheduler_policy._to_pybind(),
            has_kv_cache_manager=kv_cache_manager is not None,
            two_step_lookahead=two_step_lookahead,
            no_schedule_until_state=LlmRequestState.CONTEXT_INIT,
            no_schedule_after_state=LlmRequestState.GENERATION_COMPLETE)

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        return self.impl(active_requests, self.kv_cache_manager,
                         self.peft_cache_manager)


class MaxUtilizationScheduler(CapacityScheduler):

    # only schedule requests has no_schedule_until_state <= state < no_schedule_after_state
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager):
        """
        Args:
            max_num_requests: Maximum number of concurrent requests
            kv_cache_manager: KV cache manager instance (KVCacheManagerV2)
        """
        super(MaxUtilizationScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:

        scheduled_requests = []

        for request in active_requests:
            req_state = request.state
            # if request cannot be scheduled yet or request should no longer be scheduled, skip
            if len(scheduled_requests) >= self.max_num_requests:
                break
            elif req_state.value < self.no_schedule_until_state.value or req_state.value >= self.no_schedule_after_state.value:
                continue
            scheduled_requests.append(request)

        return scheduled_requests, [], []

    def prepare_resources(
            self, context_requests: RequestList,
            generation_requests: RequestList
    ) -> tuple[RequestList, RequestList]:
        """
        Prepare resources for MAX_UTILIZATION scheduling policy.
        This method handles KV cache allocation with eviction support.

        Args:
            context_requests: List of context requests to prepare
            generation_requests: List of generation requests to prepare

        Returns:
            tuple: (updated_context_requests, updated_generation_requests)
        """
        from .resource_manager import request_context

        evicted_requests = []
        no_scheduled = []
        rm = self.kv_cache_manager
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests = list(context_requests)
        scheduled_batch.generation_requests = list(generation_requests)

        with request_context(rm.is_draft, scheduled_batch):
            context_batch = scheduled_batch.context_requests
            generation_batch = scheduled_batch.generation_requests

            new_generation_batch: RequestList = []

            for req in generation_batch:
                if req in evicted_requests:
                    continue
                kv_cache = rm.kv_cache_map[req.py_request_id]

                if not kv_cache.is_active:
                    result = kv_cache.resume(
                        torch.cuda.current_stream().cuda_stream)
                    if not result:
                        no_scheduled.append(req)
                        continue
                # Max Utilization Scheduler: Try to increase capacity for generation
                # Recursively try to evict requests until we have enough capacity
                max_eviction_attempts = len(generation_batch) - len(
                    evicted_requests)
                capacity_increased = False

                for _ in range(max_eviction_attempts):
                    try:
                        kv_cache.capacity += 1
                        new_generation_batch.append(req)
                        capacity_increased = True
                        break
                    except OutOfPagesError:
                        evicted = self._try_evict_requests_for_capacity(
                            scheduled_batch, req, kv_cache.capacity + 1,
                            kv_cache, new_generation_batch)
                        if evicted is None:
                            # No more requests to evict
                            break
                        if evicted in new_generation_batch:
                            new_generation_batch.remove(evicted)
                        evicted_requests.append(evicted)

                if not capacity_increased:
                    # Could not increase capacity even after evicting all possible requests
                    no_scheduled.append(req)
                    continue

            # allocate KV Cache
            new_context_batch: RequestList = []
            for req in context_batch:
                beam_width = req.sampling_config.beam_width
                if 'cp_type' in rm.mapping.cp_config and CpType.STAR == rm.mapping.cp_config[
                        'cp_type']:
                    raise RuntimeError(
                        "Star attention is not supported for kv cache manager v2"
                    )
                else:
                    kv_cache = None
                    if req.is_first_context_chunk and rm._kv_connector_should_add_sequence(
                            req):
                        if req.py_request_id in rm.kv_cache_map:
                            kv_cache = rm.kv_cache_map[req.py_request_id]
                        else:
                            # Last token cannot be recovered, so we don't include it in the input tokens to look up for the block that can be reused.
                            kv_cache = rm.impl.create_kv_cache(
                                req.lora_task_id,
                                req.get_tokens(0)[:-1]
                                if rm.enable_block_reuse else None)
                            assert beam_width == 1, "Currently, KVCacheManagerV2 only supports beam width 1"
                            assert req.py_request_id not in rm.kv_cache_map, f"req.py_request_id {req.py_request_id} already in kv_cache_map"
                            rm.kv_cache_map[req.py_request_id] = kv_cache
                        if not rm.enable_block_reuse:
                            assert kv_cache.num_committed_tokens == 0
                            kv_cache.stop_committing()
                        else:
                            req.context_current_position = kv_cache.num_committed_tokens
                            chunk_size = req.context_chunk_size
                            if req.context_current_position + req.context_chunk_size < req.prompt_len:
                                floored_end_position = (
                                    req.context_current_position +
                                    req.context_chunk_size
                                ) // rm.tokens_per_block * rm.tokens_per_block
                                chunk_size = floored_end_position - req.context_current_position

                            req.context_chunk_size = min(
                                chunk_size,
                                req.prompt_len - req.context_current_position)

                        success = kv_cache.resume(
                            torch.cuda.current_stream().cuda_stream)
                        if not success:
                            no_scheduled.append(req)
                            continue
                        try:
                            kv_cache.capacity = req.prompt_len
                            new_context_batch.append(req)
                        except OutOfPagesError:
                            no_scheduled.append(req)
                            kv_cache.suspend()
                            continue

                        if rm.kv_connector_manager is not None:
                            block_ids = rm.get_cache_indices(req)
                            rm.kv_connector_manager.update_state_after_alloc(
                                req, block_ids)
                    else:
                        assert req.py_request_id in rm.kv_cache_map, f"req.py_request_id {req.py_request_id} not in kv_cache_map"
                        kv_cache = rm.kv_cache_map[req.py_request_id]
                        assert kv_cache.status is _KVCache.Status.ACTIVE, f"kv_cache {req.py_request_id} is not active"
                        new_context_batch.append(req)

        # Update scheduled_batch with the filtered lists for kv_connector_manager
        scheduled_batch.context_requests = new_context_batch
        scheduled_batch.generation_requests = new_generation_batch

        if rm.kv_connector_manager is not None:
            rm.kv_connector_manager.build_scheduler_output(scheduled_batch, rm)

        return new_context_batch, new_generation_batch

    def _try_evict_requests_for_capacity(self,
                                         scheduled_batch: "ScheduledRequests",
                                         current_req: LlmRequest,
                                         needed_capacity: int, current_kv_cache,
                                         new_generation_batch) -> LlmRequest:
        """
        Try to evict requests to make room for capacity allocation.

        Based on TestBatching pattern:
        - Find requests that can be evicted (from the end - LIFO)
        - Suspend their kv_caches
        - Move them from scheduled to paused

        Args:
            scheduled_batch: Current scheduled batch
            current_req: Request that needs capacity
            needed_capacity: Required capacity
            current_kv_cache: KV cache of current request
            new_generation_batch: List of generation requests being built

        Returns:
            Evicted request or None
        """
        rm = self.kv_cache_manager
        evicted_request = None
        all_scheduled_requests = scheduled_batch.context_requests + scheduled_batch.generation_requests

        # Try to evict from the end (LIFO - Last In First Out)
        # Don't evict the current request itself
        for req in reversed(new_generation_batch):
            if req == current_req:
                continue

            req_id = req.py_request_id if hasattr(
                req, 'py_request_id') else req.request_id

            # Check if this request has a kv_cache
            if req_id not in rm.kv_cache_map:
                continue

            kv_cache = rm.kv_cache_map[req_id]

            if kv_cache.status is not _KVCache.Status.ACTIVE:
                continue

            # Only evict requests that are in generation or have started context processing
            can_evict = False
            if req.state == LlmRequestState.GENERATION_IN_PROGRESS:
                can_evict = True
            elif req.state == LlmRequestState.CONTEXT_INIT and \
                 hasattr(req, 'context_current_position') and \
                 req.context_current_position > 0:
                can_evict = True

            if not can_evict:
                continue

            # Suspend the kv_cache
            kv_cache.suspend()
            evicted_request = req
            break

        if evicted_request is None:
            for req in all_scheduled_requests:
                if req == current_req:
                    continue
                req_id = req.py_request_id if hasattr(
                    req, 'py_request_id') else req.request_id

                # Check if this request has a kv_cache
                if req_id not in rm.kv_cache_map:
                    continue

                kv_cache = rm.kv_cache_map[req_id]
                if kv_cache.status is not _KVCache.Status.ACTIVE:
                    continue
                can_evict = False
                if req.state == LlmRequestState.GENERATION_IN_PROGRESS:
                    can_evict = True
                elif req.state == LlmRequestState.CONTEXT_INIT and \
                    hasattr(req, 'context_current_position') and \
                    req.context_current_position > 0:
                    can_evict = True

                if not can_evict:
                    continue
                kv_cache.suspend()
                evicted_request = req
                break
        return evicted_request


class GuaranteedNoEvictScheduler(CapacityScheduler):
    # only schedule requests has no_schedule_until_state <= state < no_schedule_after_state
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager):
        super(GuaranteedNoEvictScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        scheduled_requests = []
        pending_requests = []
        reserved_blocks = 0
        max_blocks = self.kv_cache_manager.get_max_resource_count()
        for request in active_requests:
            req_state = request.state
            # if request cannot be scheduled yet or request should no longer be scheduled, skip
            if req_state.value < self.no_schedule_until_state.value or req_state.value >= self.no_schedule_after_state.value:
                continue

            if len(scheduled_requests
                   ) >= self.max_num_requests or reserved_blocks >= max_blocks:
                break
            elif req_state == LlmRequestState.GENERATION_IN_PROGRESS or req_state == LlmRequestState.GENERATION_TO_COMPLETE:
                scheduled_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
            else:
                pending_requests.append(request)

        avaiable_blocks = max_blocks - reserved_blocks
        for request in pending_requests:
            req_state = request.state
            if len(scheduled_requests) >= self.max_num_requests:
                break
            elif req_state == LlmRequestState.CONTEXT_INIT:
                needed_blocks = self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
                if needed_blocks <= avaiable_blocks:
                    scheduled_requests.append(request)
                    avaiable_blocks -= needed_blocks
                elif needed_blocks > avaiable_blocks:
                    # If one requests fails to be scheduled, break
                    break

        assert len(scheduled_requests) > 0, (
            "no pending request can get enough resource to complete, "
            "please increase KV cache pool size.")
        return scheduled_requests, [], []


class MicroBatchScheduler(ABC):

    @abstractmethod
    def schedule(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :param inflight_request_ids: set of request ids that are inflight (of all micro batches)
        :return: (contextRequests, generationRequests)
        """
        # to be aligned with MicroBatchScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/microBatchScheduler.h
        raise NotImplementedError


class BindMicroBatchScheduler(MicroBatchScheduler):

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: int = None,
        ctx_chunk_config: Optional[Tuple[StrEnum, int]] = None,
    ) -> None:
        super(BindMicroBatchScheduler, self).__init__()
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens

        ctx_chunk_config_cpp = None
        if ctx_chunk_config is not None:
            ctx_chunk_config_cpp = tb_internal.batch_manager.ContextChunkingConfig(
                ctx_chunk_config[0]._to_pybind(), ctx_chunk_config[1])

        self.impl = tb_internal.algorithms.MicroBatchScheduler(
            ctx_chunk_config_cpp, max_num_tokens)

    def schedule(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        return self.impl(active_requests, inflight_request_ids,
                         self.max_batch_size, self.max_num_tokens)


class SimpleScheduler(RequestScheduler):

    def __init__(self, capacity_scheduler: CapacityScheduler,
                 micro_batch_scheduler: MicroBatchScheduler):
        super(SimpleScheduler, self).__init__()
        self.capacity_scheduler = capacity_scheduler
        self.micro_batch_scheduler = micro_batch_scheduler

    def schedule_request(self, active_requests: RequestList,
                         inflight_request_ids: set[int]) -> SchedulerOutput:
        fitting_requests, fitting_disagg_gen_init_requests, paused_requests = self.capacity_scheduler.schedule_request(
            active_requests)

        context_requests, generation_requests = self.micro_batch_scheduler.schedule(
            fitting_requests, inflight_request_ids)
        # Convert from binding type RequestVector to list[LlmRequest],
        # so Python fields on LlmRequest won't be stripped away
        context_requests = list(context_requests)
        generation_requests = list(generation_requests)

        # For MAX_UTILIZATION scheduler, prepare resources as part of scheduling
        # and get the updated request lists back
        if isinstance(self.capacity_scheduler,
                      MaxUtilizationScheduler) and hasattr(
                          self.capacity_scheduler, 'prepare_resources'):
            context_requests, generation_requests = self.capacity_scheduler.prepare_resources(
                context_requests, generation_requests)

        return SchedulerOutput(context_requests, generation_requests,
                               list(paused_requests),
                               list(fitting_disagg_gen_init_requests),
                               len(fitting_requests))
