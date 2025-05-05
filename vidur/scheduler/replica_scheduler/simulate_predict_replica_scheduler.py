import copy

from vidur.entities import Request, Batch
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
import heapq


class SimulatePredictReplicaScheduler:
    """
    Simulate the replica scheduler and predict the scheduling delay, request makespan, average batch size and
    average decoding latency
    Rely on actual replica scheduler to simulate the batch scheduling
    and use the execution time predictor to predict the execution time of each batch
    """

    def __init__(self, replica_scheduler: BaseReplicaScheduler,
                 request: Request,
                 execution_time_predictor: BaseExecutionTimePredictor,
                 use_estimated_execution_time=True,
                 copy_replica_scheduler=True,
                 start_time=0) -> None:
        self._replica_id = replica_scheduler.replica_id
        self._raw_replica_scheduler = replica_scheduler
        if copy_replica_scheduler:
            self._replica_scheduler = copy.deepcopy(replica_scheduler)
            self._target_request = copy.deepcopy(request)
            self._target_request._num_decode_tokens = request.num_predicted_decode_tokens
        else:
            self._replica_scheduler = replica_scheduler
            self._target_request = request
            self._target_request._num_decode_tokens = request.num_predicted_decode_tokens
        self._copy_needed = copy_replica_scheduler
        self._execution_time_predictor = execution_time_predictor
        self._all_request_batch_info = []
        self._scheduled_batch_heap = []
        self._scheduled_batch_id = 0
        self._estimate_execution_time = use_estimated_execution_time
        self._default_execution_time = 0.05
        self._start_time = start_time
        self._requests = {}

    def simulate(self):
        assert self._target_request is not None
        self._replica_scheduler.add_request(self._target_request)
        existing_batches = self._replica_scheduler.running_batches
        self._replica_scheduler.running_batches = []
        for batch in existing_batches:
            self.__push_batch(copy.copy(batch), self._start_time)
        new_batches = self._replica_scheduler.on_schedule()

        # so the initialized batch == the number of stages then only be pushed after pop so that the batch number
        # is limited by the number of stages
        for new_batch in new_batches:
            self.__push_batch(new_batch, self._start_time)
        while self._scheduled_batch_heap:
            (batch_id, batch_execution_time, schedule_time, batch, num_allocated_blocks) = self.__pop_batch()
            for request in batch.requests:
                self._requests[request.id] = request
            self._all_request_batch_info.append({
                "batch_id": batch_id,
                "batch_execution_time": batch_execution_time,
                "schedule_time": schedule_time,
                "batch_size": batch.size,
                "num_allocated_blocks": num_allocated_blocks
            })


    def __push_batch(self, batch: Batch, schedule_time: int):
        batch_execution_time = []
        for stage_id in self._replica_scheduler.replica_stage_schedulers.keys():
            execution_time = self.__get_execution_time(batch, stage_id)
            # if the stage is busy, wait for the current batch to complete.
            # TODO(wda): not sure if this will introduce a duplicated time so keep it as comments but be rechecked later
            # if replica_stage_scheduler.is_busy:
            #     replica_stage_scheduler = self._replica_scheduler.get_replica_stage_scheduler(stage_id)
            #     execution_time += replica_stage_scheduler.current_execution_time
            batch_execution_time.append(execution_time)
        batch_id = self._scheduled_batch_id
        self._scheduled_batch_id += 1
        completed_at = sum(batch_execution_time) + schedule_time
        batch_info = (completed_at, schedule_time, batch_id, batch, batch_execution_time)
        batch.on_schedule(schedule_time)
        heapq.heappush(self._scheduled_batch_heap, batch_info)

    def __pop_batch(self):
        (completed_at, schedule_time, batch_id, batch, batch_execution_time) = heapq.heappop(self._scheduled_batch_heap)
        batch.on_batch_end(completed_at)
        self._replica_scheduler.on_batch_end(batch)
        new_batches = self._replica_scheduler.on_schedule()
        num_allocated_blocks = self._replica_scheduler.num_allocated_blocks
        for new_batch in new_batches:
            self.__push_batch(new_batch, completed_at)
        return batch_id, batch_execution_time, schedule_time, batch, num_allocated_blocks

    def __get_execution_time(self, batch: Batch, stage_id: int):
        if self._estimate_execution_time:
            return self._execution_time_predictor.get_execution_time(batch, stage_id).total_time
        else:
            return self._default_execution_time

    @property
    def target_request_scheduled_at(self):
        return self._target_request.scheduled_at

    @property
    def target_request_completed_at(self):
        return self._target_request.completed_at

    @property
    def average_execution_time(self):
        return (sum([sum(info["batch_execution_time"]) for info in self._all_request_batch_info]) /
                len(self._all_request_batch_info))

    @property
    def average_latency(self):
        completed_requests = [request for request in self._requests.values() if request.completed]
        if not completed_requests:
            return 0
        return (sum([request.completed_at - request.scheduled_at for request in completed_requests]) /
                len(completed_requests))

    @property
    def average_stage_time(self):
        stage_times = []
        for info in self._all_request_batch_info:
            stage_times.extend(info["batch_execution_time"])
        return sum(stage_times) / len(stage_times)

    @property
    def average_batch_size(self):
        return (sum([info["batch_size"] for info in self._all_request_batch_info]) /
                len(self._all_request_batch_info))

    @property
    def max_batch_size(self):
        return max([info["batch_size"] for info in self._all_request_batch_info])

    @property
    def avg_block_size(self):
        return (sum([info["num_allocated_blocks"] for info in self._all_request_batch_info]) /
                len(self._all_request_batch_info))

    def get_execution_time(self, batch: Batch, stage_id: int):
        if self._estimate_execution_time:
            return self._execution_time_predictor.get_execution_time(batch, stage_id).total_time
        else:
            return self._default_execution_time
