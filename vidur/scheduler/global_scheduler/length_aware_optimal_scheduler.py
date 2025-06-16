import copy
import random
from typing import List, Tuple

import numpy as np

from vidur.config import LengthAwareOptimalSchedulerConfig
from vidur.entities import Request
from vidur.request_timeline_predictor.request_timeline_predictor_registry import RequestTimelinePredictorRegistry
from vidur.request_timeline_predictor.base_request_timeline_predictor import get_target_metric_value
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


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
        self._length_prediction_error = self._config.cluster_config.global_scheduler_config.length_prediction_error
        self._metric_prediction_error = self._config.cluster_config.global_scheduler_config.metrics_prediction_error

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        while self._request_queue:
            original_request = self._request_queue.pop(0)
            if self._length_prediction_error > 0:
                noise = ((random.uniform(1 - self._length_prediction_error, 1 + self._length_prediction_error))
                         * original_request.num_decode_tokens)
                length_noise = original_request.num_decode_tokens + noise
                predicted_request = Request(
                    original_request.arrived_at,
                    original_request.num_prefill_tokens,
                    original_request.num_decode_tokens + int(length_noise),
                )
            else:
                predicted_request = copy.deepcopy(original_request)
            latency_map = {
                replica_scheduler.replica_id: get_target_metric_value(self._target_metric,
                                                                      replica_scheduler,
                                                                      predicted_request,
                                                                      self._request_timeline_predictor)
                for replica_scheduler in self._replica_schedulers.values()
            }
            if self._metric_prediction_error > 0:
                for replica_id, latency in latency_map.items():
                    noise = random.uniform(1 - self._metric_prediction_error, 1 + self._metric_prediction_error)
                    latency_map[replica_id] = latency * noise
            if self._target_metric.name.startswith("MAX"):
                replica_id = max(latency_map.items(), key=lambda x: x[1])[0]
            else:
                replica_id = min(latency_map.items(), key=lambda x: x[1])[0]
            request_mapping.append((replica_id, original_request))
        return request_mapping
