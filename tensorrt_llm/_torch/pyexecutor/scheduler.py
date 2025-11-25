from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional, Tuple

from strenum import StrEnum

from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

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
        print(
            f"GuaranteedNoEvictScheduler schedule_request, active_requests: {len(active_requests)}"
        )
        print(f"self.max_num_requests: {self.max_num_requests}")
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
        print(f"len(scheduled_requests): {len(scheduled_requests)}")
        return scheduled_requests, [], []


class MaxUtilizationScheduler(CapacityScheduler):
    """
    Simplified Python version of MaxUtilizationScheduler from C++.

    This scheduler maximizes KV cache utilization by:
    - Only allocating resources for the next step (not to completion)
    - Supporting pausing/evicting previously started requests when needed
    - Using greedy scheduling with backtracking

    Key differences from GuaranteedNoEvictScheduler:
    - Allocates resources for next step only (not to completion)
    - Can pause/evict previously started requests to make room for new ones
    - Returns paused_requests in the third element of the tuple

    Simplified compared to C++ version (not implemented):
    - No PEFT/LoRA cache management
    - No cross-KV cache support (encoder-decoder models)
    - No disaggregated generation init requests
    - No block reuse optimization (beneficialToSkip)
    - No two-step lookahead

    Note: Requires kv_cache_manager to implement:
    - get_max_resource_count(): returns total available blocks
    - get_needed_resource_to_completion(request): returns blocks needed for
      next iteration (vs get_needed_resource_to_completion for GuaranteedNoEvict)
    """
    # only schedule requests: no_schedule_until_state <= state < no_schedule_after_state
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager):
        super(MaxUtilizationScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager

    def _is_started_request(self, request: LlmRequest) -> bool:
        """Check if request has already started (in progress or chunked context)."""
        req_state = request.state
        # Check state bounds
        if (req_state.value < self.no_schedule_until_state.value
                or req_state.value >= self.no_schedule_after_state.value):
            return False
        # Started if in generation or if it's a chunked context (not first chunk)
        if req_state == LlmRequestState.GENERATION_IN_PROGRESS:
            return True
        # For context requests, check if it's not the first chunk
        # (simplified - assume first chunk if new)
        # In C++: (req->isContextInitState() && !req->isFirstContextChunk())
        return False

    def _can_schedule_request(self, request: LlmRequest, scheduled_count: int,
                              used_blocks: int, max_blocks: int) -> bool:
        """Try to schedule a request by checking resources for next step."""
        if scheduled_count >= self.max_num_requests:
            return False

        # Get resources needed for next step (not to completion)
        needed_blocks = self.kv_cache_manager.get_needed_resource_for_next_step(
            request)

        # Check if we have enough available blocks
        return (used_blocks + needed_blocks) <= max_blocks

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        """
        Schedule requests with max utilization policy.

        Returns:
            tuple: (scheduled_requests, [], paused_requests)
                   - scheduled_requests: requests that can be executed
                   - paused_requests: requests that were evicted to make room
        """
        print(f"MaxUtilizationScheduler schedule_request, "
              f"active_requests: {len(active_requests)}")
        print(f"self.max_num_requests: {self.max_num_requests}")

        scheduled_requests = []
        paused_requests = []
        used_blocks = 0
        max_blocks = self.kv_cache_manager.get_max_resource_count()

        # Track which requests we've processed
        active_window_end = len(active_requests)
        current_idx = 0

        if len(active_requests) > 1300:
            print(f"len(active_requests): {len(active_requests)}")

        while current_idx < active_window_end:
            request = active_requests[current_idx]
            req_state = request.state

            # Skip requests that cannot/should not be scheduled
            if (req_state.value < self.no_schedule_until_state.value
                    or req_state.value >= self.no_schedule_after_state.value):
                current_idx += 1
                continue

            # Try to schedule this request
            if self._can_schedule_request(request, len(scheduled_requests),
                                          used_blocks, max_blocks):
                # Successfully scheduled
                scheduled_requests.append(request)
                needed_blocks = \
                    self.kv_cache_manager.get_needed_resource_for_next_step(
                        request)
                used_blocks += needed_blocks
                # print(f"  Scheduled request (state={req_state}), "
                #       f"blocks used: {used_blocks}/{max_blocks}")
                current_idx += 1
            else:
                # Cannot schedule - try to evict a started request from the end
                print(f"  Cannot schedule request (state={req_state}), "
                      f"attempting eviction...")

                # Find the last started request before current position
                evicted = False
                for evict_idx in range(active_window_end - 1, current_idx - 1,
                                       -1):
                    if self._is_started_request(active_requests[evict_idx]):
                        # Found a request to evict
                        evicted_request = active_requests[evict_idx]
                        paused_requests.append(evicted_request)

                        # Remove from scheduled if it was already scheduled
                        if evicted_request in scheduled_requests:
                            scheduled_requests.remove(evicted_request)
                            evicted_blocks = \
                                self.kv_cache_manager.get_needed_resource_for_next_step(
                                    evicted_request)
                            used_blocks -= evicted_blocks

                        # Shrink the active window
                        active_window_end = evict_idx
                        print(f"    Evicted request at index {evict_idx}, "
                              f"new window: 0-{active_window_end}")
                        evicted = True
                        break

                if not evicted:
                    # No started request to evict, cannot make progress
                    print("  No requests to evict, stopping scheduling")
                    break
                # Don't increment current_idx - retry scheduling this request

        print(f"MaxUtilizationScheduler: scheduled={len(scheduled_requests)}, "
              f"paused={len(paused_requests)}")
        return scheduled_requests, [], paused_requests


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
        # print(f"fitting_requests: {len(fitting_requests)}")
        # # print(
        # #     f"fitting_disagg_gen_init_requests: {len(fitting_disagg_gen_init_requests)}"
        # # )
        # print(f"paused_requests: {len(paused_requests)}")
        context_requests, generation_requests = self.micro_batch_scheduler.schedule(
            fitting_requests, inflight_request_ids)
        # Convert from binding type RequestVector to list[LlmRequest],
        # so Python fields on LlmRequest won't be stripped away
        return SchedulerOutput(list(context_requests),
                               list(generation_requests), list(paused_requests),
                               list(fitting_disagg_gen_init_requests),
                               len(fitting_requests))
