from typing import List, Tuple

from vidur.config import LengthAwareOptimalSchedulerConfig
from vidur.entities import Request
from vidur.request_timeline_predictor.base_request_timeline_predictor import BaseRequestTimelinePredictor
from vidur.request_timeline_predictor.request_timeline_predictor_registry import RequestTimelinePredictorRegistry
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


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


class LengthAwareOptimalScheduler(BaseGlobalScheduler):
    """
    Length-aware optimal scheduler to schedule requests based on the number of unprocessed tokens
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self._config.cluster_config.global_scheduler_config, LengthAwareOptimalSchedulerConfig):
            raise ValueError("Invalid global scheduler config type")
        self._target_metric = TargetMetric.from_str(self._config.cluster_config.global_scheduler_config.target_metric)
        self._request_timeline_predictor = RequestTimelinePredictorRegistry.get(
            self._config.cluster_config.global_scheduler_config.request_timeline_predictor_config.get_type()
        )
        self._request_timeline_predictor.attach_execution_time_predictor(self._execution_time_predictor)

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        while self._request_queue:
            request = self._request_queue.pop(0)
            latency_map = {
                replica_scheduler.replica_id: get_target_metric_value(self._target_metric,
                                                                      replica_scheduler, request,
                                                                      self._request_timeline_predictor)
                for replica_scheduler in self._replica_schedulers.values()
            }
            if self._target_metric.name.startswith("MAX"):
                replica_id = max(latency_map.items(), key=lambda x: x[1])[0]
            else:
                replica_id = min(latency_map.items(), key=lambda x: x[1])[0]
            request_mapping.append((replica_id, request))
        return request_mapping
