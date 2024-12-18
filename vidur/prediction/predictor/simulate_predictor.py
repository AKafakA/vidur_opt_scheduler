import random
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
        self._config = config
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
        if config.target_metric.upper() in TargetMetric.__members__.keys():
            self._target_metric = TargetMetric.from_str(config.target_metric)
            self._need_to_predict = True
        else:
            self._need_to_predict = False
        from vidur.request_timeline_predictor.simulate_request_timeline_predictor import \
            SimulateRequestTimelinePredictor
        self._request_timeline_predictor = SimulateRequestTimelinePredictor()
        self._request_timeline_predictor.attach_execution_time_predictor(self._execution_time_predictor)
        if config.disable_batch_time_estimation:
            self._request_timeline_predictor.disable_batch_time_estimation()
        self._port = port
        self._request_decode_length_prediction_map = {}
        self._start_time = time.time()
        self._backend_url = f"http://localhost:{self._port}/schedule_trace"
        self._current_gpu_blocks = 0
        self._num_requests = 0
        self._num_preempted = 0

    def predict(self, target_request: Request):
        self.reset()
        metrics = {}
        if self._need_to_predict:
            from vidur.request_timeline_predictor.base_request_timeline_predictor import get_target_metric_value
            metric = get_target_metric_value(self._target_metric, self._replica_scheduler, target_request,
                                             self._request_timeline_predictor)
            self._request_decode_length_prediction_map[target_request.id] = target_request.num_decode_tokens
            print(self._request_decode_length_prediction_map.keys())
            target_metric = metric
        elif self._config.target_metric == "min_gpu_blocks":
            target_metric = self._current_gpu_blocks
        elif self._config.target_metric == "min_requests":
            target_metric = self._num_requests
        elif self._config.target_metric == "random" or self._config.target_metric == "round_robin":
            target_metric = random.randint(0, 100)
        else:
            raise ValueError(f"Invalid metrics type: {self._config.target_metric}")
        metrics["target_metric"] = target_metric
        metrics["gpu_blocks"] = self._current_gpu_blocks
        metrics["num_requests"] = self._num_requests
        metrics["num_preempted"] = self._num_preempted
        return metrics

    def __generate_requests_from_backend(self, request_info: dict):
        request_id = int(request_info["request_id"])
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
        current_gpu_blocks = 0
        current_num_requests = 0
        current_num_preempted = 0
        for batch in serialized_response.keys():
            batch_request_information = serialized_response[batch]
            waiting_request_length = batch_request_information["waiting"]
            running_request_length = batch_request_information["running"]
            swap_request_length = batch_request_information["swap"]
            if self._need_to_predict:
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
            current_gpu_blocks += batch_request_information["free_gpu_blocks"]
            current_num_requests += len(running_request_length) + len(swap_request_length) + len(waiting_request_length)
            current_num_preempted += len(swap_request_length)
        self._current_gpu_blocks = current_gpu_blocks
        self._num_requests = current_num_requests
        self._num_preempted = current_num_preempted
