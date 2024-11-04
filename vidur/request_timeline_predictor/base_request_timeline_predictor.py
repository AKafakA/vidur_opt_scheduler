from vidur.config import BaseRequestTimelinePredictorConfig, ReplicaConfig, BaseReplicaSchedulerConfig
from vidur.entities import Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler


class BaseRequestTimelinePredictor:
    def __init__(self):
        self._execution_time_predictor = None

    def attach_execution_time_predictor(self, execution_time_predictor: BaseExecutionTimePredictor):
        self._execution_time_predictor = execution_time_predictor

    def predict_scheduling_delay(self, replica_scheduler: BaseReplicaScheduler, request: Request):
        raise NotImplementedError("predict method is not implemented")

    def predict_request_makespan(self, replica_scheduler: BaseReplicaScheduler, request: Request):
        raise NotImplementedError("predict method is not implemented")

    def predict_average_decoding_latency(self, replica_scheduler, request):
        raise NotImplementedError("predict method is not implemented")

    def predict_average_batch_size(self, replica_scheduler, request):
        raise NotImplementedError("predict method is not implemented")

    def predict_min_batch_size(self, replica_scheduler, request):
        raise NotImplementedError("predict method is not implemented")
