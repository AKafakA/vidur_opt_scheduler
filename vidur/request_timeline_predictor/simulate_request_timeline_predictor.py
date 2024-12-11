from vidur.entities import Request
from vidur.request_timeline_predictor.base_request_timeline_predictor import BaseRequestTimelinePredictor
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.scheduler.replica_scheduler.simulate_predict_replica_scheduler import SimulatePredictReplicaScheduler
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


class SimulateRequestTimelinePredictor(BaseRequestTimelinePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_estimated_time = True

    def disable_batch_time_estimation(self):
        self._use_estimated_time = False

    def predict_scheduling_delay(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self._use_estimated_time
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.schedule_at

    def predict_request_makespan(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self._use_estimated_time
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.completed_at

    def predict_average_batch_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=False
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_batch_size

    def predict_min_batch_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=False
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.min_batch_size

    def predict_average_decoding_latency(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self._use_estimated_time
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_decode_time


def get_target_metric_value(target_metric: TargetMetric,
                            replica_scheduler: BaseReplicaScheduler,
                            request: Request,
                            request_timeline_predictor: BaseRequestTimelinePredictor):
    if target_metric == TargetMetric.MIN_LATENCY:
        return request_timeline_predictor.predict_request_makespan(replica_scheduler, request)
    elif target_metric == TargetMetric.MIN_SCHEDULING_DELAY:
        return request_timeline_predictor.predict_scheduling_delay(replica_scheduler, request)
    elif target_metric == TargetMetric.MIN_DECODING_DELAY:
        return request_timeline_predictor.predict_average_decoding_latency(replica_scheduler, request)
    elif target_metric == TargetMetric.MAX_AVG_BATCH_SIZE:
        return request_timeline_predictor.predict_average_batch_size(replica_scheduler, request)
    elif target_metric == TargetMetric.MAX_MIN_BATCH_SIZE:
        return request_timeline_predictor.predict_average_batch_size(replica_scheduler, request)
    else:
        raise ValueError("Invalid target metric")
