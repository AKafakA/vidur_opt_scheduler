import time

from vidur.config import DummyRequestGeneratorConfig, MetricsConfig, \
    SimulationRequestTimelinePredictorConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.prediction.predictor.predictor import Predictor
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import ReplicaSchedulerRegistry
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric


class SimulatePredictor(Predictor):
    """A raw implementation of the predictor class
      that extends the Simulator class which is used to predict the completion time of the request.
      It will always create a mirror of replica scheduler and make the prediction based on the target metric.
    """

    def __init__(self, config, port):
        super().__init__(config, port)
        self._generate_config = DummyRequestGeneratorConfig()
        self._metrics_config = MetricsConfig()
        self._simulation_config = SimulationRequestTimelinePredictorConfig()
        self._replica = Replica(config.replica_config, self._generate_config)
        self._execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.replica_config,
            replica_scheduler_config=config.replica_scheduler_config,
            metrics_config=self._metrics_config,
        )
        self._replica_scheduler = ReplicaSchedulerRegistry.get(
            config.replica_scheduler_config.get_type(),
            replica_config=config.replica_config,
            replica_scheduler_config=config.replica_scheduler_config,
            request_generator_config=self._generate_config,
            replica=self._replica,
            num_stages=self._replica.num_pipeline_stages,
            execution_time_predictor=self._execution_time_predictor,
        )
        self._request_queue = []
        self._target_metric = TargetMetric.from_str(config.target_metric)
        from vidur.request_timeline_predictor.simulate_request_timeline_predictor import \
            SimulateRequestTimelinePredictor
        self._request_timeline_predictor = SimulateRequestTimelinePredictor()
        self._request_timeline_predictor.attach_execution_time_predictor(self._execution_time_predictor)
        self._port = port
        self._request_decode_length_prediction_map = {}
        self._start_time = time.time()
        self._backend_url = f"http://localhost:{self._port}/schedule_trace"

    def predict(self, target_request: Request):
        self.reset()
        from vidur.request_timeline_predictor.simulate_request_timeline_predictor import get_target_metric_value
        metric = get_target_metric_value(self._target_metric, self._replica_scheduler, target_request,
                                         self._request_timeline_predictor)
        self._request_decode_length_prediction_map[target_request.id] = target_request.num_decode_tokens
        return metric

    def __generate_requests_from_backend(self, request_info: dict):
        request_id = request_info["request_id"]
        arrival_time = request_info["arrival_time"]
        context_length = request_info["seq_prompts_length"]
        decoded_length = request_info["seq_total_output_length"]
        decode_length = self._request_decode_length_prediction_map[request_id]
        request = Request(arrival_time, context_length, decode_length, decoded_length)
        return request

    def reset(self):
        from vidur.prediction.server_utils import get_http_request
        response = get_http_request(self._backend_url)
        serialized_response = response.json()
        for batch in serialized_response.keys():
            batch_request_information = serialized_response[batch]
            waiting_request_length = batch_request_information["waiting"]
            running_request_length = batch_request_information["running"]
            swap_request_length = batch_request_information["swap"]
            for requests_info in running_request_length:
                request = self.__generate_requests_from_backend(requests_info)
                self._replica_scheduler.add_request(request)
                self._replica_scheduler.allocate(request.id, requests_info["n_blocks"])

            for requests_info in swap_request_length:
                request = self.__generate_requests_from_backend(requests_info)
                self._replica_scheduler.add_preempted_request(request)

            for requests_info in waiting_request_length:
                request = self.__generate_requests_from_backend(requests_info)
                self._replica_scheduler.add_request(request)

