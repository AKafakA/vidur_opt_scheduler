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
    def __init__(self, replica_scheduler: BaseReplicaScheduler, request: Request,
                 execution_time_predictor: BaseExecutionTimePredictor) -> None:
        self._replica_id = replica_scheduler.replica_id
        self._raw_replica_scheduler = replica_scheduler
        self._replica_scheduler = copy.deepcopy(replica_scheduler)
        self._target_request = copy.deepcopy(request)
        self._execution_time_predictor = execution_time_predictor
        self._target_request_batch_info = []
        self._scheduled_batch_heap = []
        self._scheduled_batch_id = 0

    def simulate(self):
        self._replica_scheduler.add_request(self._target_request)
        existing_batches = self._replica_scheduler.running_batches
        self._replica_scheduler.running_batches = []
        for batch in existing_batches:
            self.push_batch(copy.deepcopy(batch), 0)
        new_batches = self._replica_scheduler.on_schedule()
        for new_batch in new_batches:
            self.push_batch(new_batch, 0)
        while not self._target_request.completed and self._scheduled_batch_heap:
            (batch_id, batch_execution_time, schedule_time, batch) = self.pop_batch()
            if self._target_request.id in batch.request_ids:
                self._target_request_batch_info.append({
                    "batch_id": batch_id,
                    "batch_execution_time": batch_execution_time,
                    "schedule_time": schedule_time,
                    "batch_size": batch.size
                })

    def push_batch(self, batch: Batch, schedule_time: int):
        batch_execution_time = []
        for stage_id in self._replica_scheduler.replica_stage_schedulers.keys():
            replica_stage_scheduler = self._replica_scheduler.get_replica_stage_scheduler(stage_id)
            execution_time = (
                self._execution_time_predictor.get_execution_time(batch, stage_id)).total_time
            # if the stage is busy, wait for the current batch to complete
            if replica_stage_scheduler.is_busy:
                execution_time += replica_stage_scheduler.current_execution_time
            batch_execution_time.append(execution_time)
        batch_id = self._scheduled_batch_id
        self._scheduled_batch_id += 1
        completed_at = sum(batch_execution_time) + schedule_time
        batch_info = (completed_at, schedule_time, batch_id, batch, batch_execution_time)
        heapq.heappush(self._scheduled_batch_heap, batch_info)

    def pop_batch(self):
        (completed_at, schedule_time, batch_id, batch, batch_execution_time) = heapq.heappop(self._scheduled_batch_heap)
        batch.on_batch_end(completed_at)
        self._replica_scheduler.on_batch_end(batch)
        new_batches = self._replica_scheduler.on_schedule()
        for new_batch in new_batches:
            self.push_batch(new_batch, completed_at)
        return batch_id, batch_execution_time, schedule_time, batch

    @property
    def schedule_at(self):
        return min([info["schedule_time"] for info in self._target_request_batch_info])

    @property
    def completed_at(self):
        last_batch = sorted(self._target_request_batch_info, key=lambda x: x["schedule_time"])[-1]
        return last_batch["schedule_time"] + sum(last_batch["batch_execution_time"])

    @property
    def average_decode_time(self):
        return (sum([sum(info["batch_execution_time"]) for info in self._target_request_batch_info]) /
                len(self._target_request_batch_info))

    @property
    def average_stage_time(self):
        stage_times = []
        for info in self._target_request_batch_info:
            stage_times.extend(info["batch_execution_time"])
        return sum(stage_times) / len(stage_times)

    @property
    def average_batch_size(self):
        return (sum([info["batch_size"] for info in self._target_request_batch_info]) /
                len(self._target_request_batch_info))

    @property
    def min_batch_size(self):
        return min([info["batch_size"] for info in self._target_request_batch_info])

    @property
    def max_batch_size(self):
        return max([info["batch_size"] for info in self._target_request_batch_info])
