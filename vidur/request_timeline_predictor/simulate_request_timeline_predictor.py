from vidur.request_timeline_predictor.base_request_timeline_predictor import BaseRequestTimelinePredictor
from vidur.scheduler.replica_scheduler.simulate_predict_replica_scheduler import SimulatePredictReplicaScheduler


class SimulateRequestTimelinePredictor(BaseRequestTimelinePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_estimated_time = True
        self._copy_base_replica_scheduler = True

    def disable_copy_of_base_replica_scheduler(self):
        self._copy_base_replica_scheduler = False

    def predict_avg_block_size(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self.use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.avg_block_size

    def predict_request_scheduling_delay(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self.use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.target_request_scheduled_at

    def predict_request_makespan(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self.use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.target_request_completed_at

    def predict_average_latency(self, replica_scheduler, request):
        simulate_predict_replica_scheduler = SimulatePredictReplicaScheduler(
            replica_scheduler=replica_scheduler,
            request=request,
            execution_time_predictor=self._execution_time_predictor,
            use_estimated_execution_time=self.use_estimated_time,
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
            use_estimated_execution_time=self.use_estimated_time,
            copy_replica_scheduler=self._copy_base_replica_scheduler,
        )
        simulate_predict_replica_scheduler.simulate()
        return simulate_predict_replica_scheduler.average_execution_time
