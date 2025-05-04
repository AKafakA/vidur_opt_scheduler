from vidur.entities import Request
from vidur.request_timeline_predictor.base_request_timeline_predictor import BaseRequestTimelinePredictor
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.scheduler.replica_scheduler.simulate_predict_replica_scheduler import SimulatePredictReplicaScheduler
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


class SimulateRequestTimelinePredictor(BaseRequestTimelinePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_estimated_time = True
        self._copy_base_replica_scheduler = True

    def disable_batch_time_estimation(self):
        self._use_estimated_time = False

    def disable_copy_of_base_replica_scheduler(self):
        self._copy_base_replica_scheduler = False

    def predict_avg_block_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self._use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.avg_block_size

    def predict_request_scheduling_delay(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self._use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.target_request_scheduled_at

    def predict_request_makespan(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self._use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.target_request_completed_at

    def predict_average_latency(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self._use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_latency

    def predict_average_batch_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=False,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_batch_size

    def predict_average_execution_latency(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self._use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_execution_time
