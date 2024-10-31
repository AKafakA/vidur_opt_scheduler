from vidur.request_timeline_predictor.base_request_timeline_predictor import BaseRequestTimelinePredictor
from vidur.scheduler.replica_scheduler.simulate_predict_replica_scheduler import SimulatePredictReplicaScheduler


class SimulateRequestTimelinePredictor(BaseRequestTimelinePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_scheduling_delay(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor
        )
        return simulate_predict_replica_scheduler.schedule_at

    def predict_request_makespan(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor
        )
        return simulate_predict_replica_scheduler.completed_at

    def predict_average_batch_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor
        )
        return simulate_predict_replica_scheduler.average_batch_size

    def predict_average_decoding_latency(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor
        )
        return simulate_predict_replica_scheduler.average_decode_time

    def predict_min_batch_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor
        )
        return simulate_predict_replica_scheduler.min_batch_size
