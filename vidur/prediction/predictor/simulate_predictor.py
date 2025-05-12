import json
import random
import time
from math import ceil
import itertools
import logging
import aiohttp

from vidur.config import DummyRequestGeneratorConfig, MetricsConfig, \
    SimulationRequestTimelinePredictorConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.prediction.predictor.predictor import Predictor
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import ReplicaSchedulerRegistry
from vidur.types import ReplicaSchedulerType
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric

logging.basicConfig(level=logging.INFO,
                    filemode='a+',
                    filename='predictor_benchmark.log')
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60 * 10)


class SimulatePredictor(Predictor):
    """A raw implementation of the predictor class
      that extends the Simulator class which is used to predict the completion time of the request.
      It will always create a mirror of replica scheduler and make the prediction based on the target metric.
    """

    def __init__(self, config, port):
        super().__init__(config, port)
        self._logger = logging.getLogger(__name__)
        self._config = config
        self._generate_config = DummyRequestGeneratorConfig()
        self._metrics_config = MetricsConfig(
            write_metrics=False,
            create_output_dir=False
        )
        self._simulation_config = SimulationRequestTimelinePredictorConfig()
        self._replica = Replica(config.replica_config, self._generate_config)
        self._execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.replica_config,
            replica_scheduler_config=config.replica_scheduler_config,
            metrics_config=self._metrics_config,
        )
        self._enable_chunked_prefill = config.replica_scheduler_config.get_type() == ReplicaSchedulerType.SARATHI
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
        self._request_timeline_predictor.disable_copy_of_base_replica_scheduler()
        self._request_timeline_predictor.use_estimated_time = config.enable_batch_time_estimation
        self._request_timeline_predictor.threshold_batch_size_for_time_estimation = \
            config.threshold_batch_size_for_time_estimation
        self._port = port
        self._start_time = time.time()
        self._backend_url = f"http://localhost:{self._port}/schedule_trace"

    async def predict(self, target_request: Request):
        start_time = time.time()
        (replica_scheduler, current_gpu_blocks, current_num_requests, current_num_running_request,
         current_num_waiting_request, current_num_preempted) = await self.get_replica_scheduler(target_request.id)
        metrics = {}
        time_to_get_replica_scheduler = (time.time() - start_time) * 1000
        # replica_scheduler.print_requests()
        if self._need_to_predict:
            from vidur.request_timeline_predictor.base_request_timeline_predictor import get_target_metric_value
            metric = get_target_metric_value(self._target_metric, replica_scheduler, target_request,
                                             self._request_timeline_predictor)
            target_metric = metric
            # self._logger.info(f"Predicted metric: {metric} for request: {str(target_request.id)}")
        elif self._config.target_metric == "min_current_gpu_blocks":
            target_metric = current_gpu_blocks
        elif self._config.target_metric == "min_current_requests":
            target_metric = current_num_requests
        elif self._config.target_metric == "random" or self._config.target_metric == "round_robin":
            target_metric = random.randint(0, 100)
        elif self._config.target_metric == "min_infass_load":
            target_metric = (current_num_requests / current_gpu_blocks) * (-1)
        else:
            raise ValueError(f"Invalid metrics type: {self._config.target_metric}")
        metrics["target_metric"] = target_metric
        metrics["gpu_blocks"] = current_gpu_blocks
        metrics["num_requests"] = current_num_requests
        metrics["num_preempted"] = current_num_preempted
        metrics["time_to_predict_in_ms"] = (time.time() - start_time) * 1000
        metrics["time_to_get_replica_scheduler_in_ms"] = time_to_get_replica_scheduler
        return metrics

    def __generate_requests_from_backend(self, request_info: dict, source: str) -> Request:
        request_id = int(request_info["request_id"])
        arrival_time = request_info["arrival_time"]
        context_length = request_info["seq_prompts_length"]
        total_length = request_info["seq_total_output_length"]
        prefilled_length = request_info["seq_computed_length"]
        is_prefill = request_info["is_prefill"]
        total_decode_length = request_info["seq_expected_decoded_length"]
        if self._enable_chunked_prefill:
            if is_prefill:
                # total length = sequence.prompts_length + sequence.decoded_length
                # with chunked prefilled, running consisted ongoing prefilling request, decoding request w/o preemption
                # if it is prefilling and not preempted, the total length = sequence.prompts_length
                # else if preempted, the total length = prompts_length + decoded_length before preemption
                # and the decoded tokens can also be recomputed concurrently as prompted request
                context_length = total_length
                processed_length = prefilled_length
                is_prefill_complete = False
            else:
                # if it is decoded, context length = sequence.prompts_length and processed length = decoded_length
                is_prefill_complete = True
                processed_length = total_length
            request = Request(arrival_time, context_length, total_decode_length, processed_length)
            if source == 'running':
                request._is_prefill_complete = is_prefill_complete
        else:
            # if not chunked prefill, the ongoing request is always prefilled and preempted will be appended to waiting
            # queue
            request = Request(arrival_time, context_length, total_decode_length, total_length)
            if source == 'running':
                request._is_prefill_complete = True
        request.source = source
        request.set_id(request_id)
        return request

    async def get_replica_scheduler(self, request_id):
        start_time = time.time()
        print(f"Connecting to backend at {self._backend_url} at {start_time}"
              f" for request, {request_id}")
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            print(f"Connected to backend at {self._backend_url} at {time.time() - start_time} for request, "
                  f"{request_id}")
            async with session.get(self._backend_url) as response:
                connect_time = (time.time() - start_time) * 1000
                print(f"Time taken to connect to backend: {connect_time} ms at {time.time() - start_time} "
                      f"for request, {request_id}")
                response_data = await response.json()
                return self.get_replica_scheduler_with_backend_response(response_data)

    def get_replica_scheduler_with_backend_response(self, response):
        current_gpu_blocks = 0
        current_num_requests = 0
        current_num_preempted = 0
        current_num_running_request = 0
        current_num_waiting_request = 0

        if self._need_to_predict:
            replica_scheduler = ReplicaSchedulerRegistry.get(
                self._config.replica_scheduler_config.get_type(),
                replica_config=self._config.replica_config,
                replica_scheduler_config=self._config.replica_scheduler_config,
                request_generator_config=self._generate_config,
                replica=self._replica,
                num_stages=self._replica.num_pipeline_stages,
                execution_time_predictor=self._execution_time_predictor,
            )
        else:
            replica_scheduler = None

        for batch in response.keys():
            batch_request_information = response[batch]
            waiting_request_length = batch_request_information["waiting"]
            running_request_length = batch_request_information["running"]
            swap_request_length = batch_request_information["swap"]
            if self._need_to_predict:
                for requests_info in running_request_length:
                    request = self.__generate_requests_from_backend(requests_info, 'running')
                    if request.num_processed_tokens == request.total_tokens:
                        continue
                    if self._enable_chunked_prefill:
                        allocated_tokens = max(request.num_processed_tokens, request.num_prefill_tokens)
                    else:
                        allocated_tokens = request.num_processed_tokens
                    num_required_blocks = ceil(
                        allocated_tokens / self._config.replica_scheduler_config.block_size
                    )
                    replica_scheduler.allocate(request.id, num_required_blocks)
                    request.loading_tokens = request.num_processed_tokens
                    replica_scheduler.add_preempted_request(request)

                preempted_request = []
                waiting_request = []

                for requests_info in itertools.chain(waiting_request_length, swap_request_length):
                    request = self.__generate_requests_from_backend(requests_info, 'waiting')
                    if request.num_processed_tokens == request.total_tokens:
                        continue
                    if request.num_processed_tokens > 0:
                        request.restart()
                        preempted_request.append(request)
                    else:
                        waiting_request.append(request)

                for request in itertools.chain(preempted_request, waiting_request):
                    replica_scheduler.add_request(request)

            current_gpu_blocks += batch_request_information["free_gpu_blocks"]
            current_num_requests += len(running_request_length) + len(swap_request_length) + len(
                waiting_request_length)
            current_num_preempted += batch_request_information["num_preempted"]
            current_num_running_request += len(running_request_length)
            current_num_waiting_request += len(waiting_request_length)
        return (replica_scheduler, current_gpu_blocks, current_num_requests, current_num_running_request,
                current_num_waiting_request, current_num_preempted)
