from abc import ABC, abstractmethod
from collections import deque
from typing import List

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)


class BaseReplicaScheduler(ABC):
    def __init__(
            self,
            replica_config: ReplicaConfig,
            replica_scheduler_config: BaseReplicaSchedulerConfig,
            request_generator_config: BaseRequestGeneratorConfig,
            replica: Replica,
            num_stages: int,
            execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self.running_batches = []
        self._num_running_batches = 0
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._num_stages = num_stages

        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        self._preempted_requests = deque()
        self._request_queue = deque()

        memory_planner = MemoryPlanner(self._replica_config, replica)

        if not self._config.num_blocks:
            self._num_blocks = (
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        else:
            self._num_blocks = self._config.num_blocks
        self._max_batch_size = min(
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._num_allocated_blocks = 0
        self._allocation_map = {}

        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(num_stages)
        }


    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    @property
    def replica_stage_schedulers(self):
        return self._replica_stage_schedulers

    @property
    def num_tokens_to_prefill(self) -> int:
        return sum(request.num_prefill_tokens for request in self._request_queue)

    @property
    def num_tokens_to_decode(self) -> int:
        return sum(request.num_decode_tokens - request.num_processed_tokens
                   for request in self._preempted_requests + self._request_queue)

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_free_blocks(self) -> int:
        return self._num_blocks - self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> float:
        return (self._num_allocated_blocks * 100) / self._num_blocks

    def is_empty(self) -> bool:
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        if request._is_prefill_complete:
            return 1

        return request._num_prefill_tokens

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def add_preempted_request(self, request: Request) -> None:
        self._preempted_requests.append(request)

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:
        return self._num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:
        if num_blocks + self._num_allocated_blocks > self._num_blocks:
            return
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks


    def free(self, *request_ids: List[int]) -> None:
        for request_id in request_ids:
            if request_id not in self._allocation_map:
                continue
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks
        assert self._num_allocated_blocks >= 0

    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    def on_schedule(self) -> List[Batch]:
        scheduled_batches = []
        while self._num_running_batches < self._num_stages:
            batch = self._get_next_batch()
            if not batch:
                break
            scheduled_batches.append(batch)
            self._num_running_batches += 1
            self.running_batches.append(batch)
        return scheduled_batches

    def print_requests(self):
        for request in self._request_queue:
            print(f"request id: {request.id} and processed token: {request.num_processed_tokens}")
