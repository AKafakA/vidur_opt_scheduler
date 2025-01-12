import argparse
import asyncio
import json
import random
import ssl
import time
from argparse import Namespace
from typing import Any, Optional, List

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vidur.prediction.global_scheduler.instance import Instance
from vidur.prediction.server_utils import serve_http
import resource
import logging
import threading

lock = asyncio.Lock()
TIMEOUT_KEEP_ALIVE = 10  # seconds.
app = FastAPI()
instances = []
num_requests = 0
n = 0
m = 0
start_time = 0
metrics_type = "random"
logging.basicConfig(level=logging.INFO,
                    filemode='a+',
                    filename='experiment_output/logs/predictor_output.log')
logger = logging.getLogger(__name__)


@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    """Generate completion for the request with profiling.
    This API will 1) calling the predictor to predict the completion time of the request,
    2) select the host based on target metrics and call its vllm generate API to generate the completion.
    3) return the completion to the client with profiling
    """
    assert len(instances) > 0
    request_start_time = time.time()
    request_dict = await request.json()
    request_id = request_dict["request_id"]
    prompt = request_dict.pop("prompt")
    num_context_tokens = request_dict.pop("prompt_len")
    num_decode_tokens = request_dict.pop("expected_response_len")
    arrived_at = time.time() - start_time
    _ = request_dict.pop("stream", False)
    # async with lock:
    #     global num_requests
    #     num_requests += 1
    #     request_id = num_requests
    predict_tasks = []

    for instance in instances:
        predict_tasks.append(instance.query_predictor(
            request_id, num_context_tokens, num_decode_tokens, arrived_at))
    try:
        predict_results = await asyncio.gather(*predict_tasks)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse({"error": "Prediction failed"}, status_code=500)

    if (metrics_type.startswith("min") or metrics_type.startswith("max")) and "current" not in metrics_type:
        predict_results = random.sample(predict_results, min(n, len(predict_results)))

    target_metrics = [x['target_metric'] for x in predict_results]
    time_in_predictions = [(x["time_to_predict"], x["time_to_probe"]) for x in predict_results]
    assert len(target_metrics) == len(predict_results)
    if metrics_type.startswith("min") or metrics_type.startswith("max"):
        # if current in metrics means all node need to be queried and select the one with min/max
        # as just report the current value without prediction is cheap and sample no need to be limited
        if metrics_type.startswith("min"):
            target_metric = min(target_metrics)
        else:
            target_metric = max(target_metrics)
        candidates_indexes = [i for i, value in enumerate(target_metrics) if value == target_metric]
        # if args.debugging_logs:
        #     free_gpus = [predict_results[i]['gpu_blocks'] for i in candidates_indexes]
        #     max_gpu = max(free_gpus)
        #     candidates_indexes = [i for i in candidates_indexes if predict_results[i]['gpu_blocks'] == max_gpu]
        metric_selected_index = random.choice(candidates_indexes)
        selected_instance_id = (predict_results[metric_selected_index])['instance_id']
        selected_index = [i for i, instance in enumerate(instances) if selected_instance_id == instance._instance_id][0]
    elif metrics_type == "random":
        selected_index = random.randint(0, len(instances) - 1)
    elif metrics_type == "round_robin":
        selected_index = int(request_id) % len(instances)
    elif metrics_type == "request_per_seconds":
        min_request_per_second = min(instance.total_request for instance in instances)
        selected_index = random.choice([i for i in range(len(instances)) if instances[i].total_request
                                        == min_request_per_second])
    else:
        raise ValueError(f"Invalid metrics type: {metrics_type}")

    selected_instance = instances[selected_index]
    try:
        time_to_query = time.time()
        response = await selected_instance.query_backend(prompt, num_decode_tokens, request_id)
        time_for_inference = (time.time() - time_to_query) * 1000
    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse({"error": "Prediction failed"}, status_code=500)
    if args.debugging_logs:
        print(f"Selected instance: {selected_instance._instance_id} for request {request_id} "
              f"with metrics type: {metrics_type} and predict results: {predict_results}")
    response['sampled_avg_gpu_blocks'] = np.mean([x['gpu_blocks'] for x in predict_results])
    response['sampled_var_gpu_blocks'] = np.var([x['gpu_blocks'] for x in predict_results])
    response['sampled_avg_n_request'] = np.mean([x['num_requests'] for x in predict_results])
    response['sampled_var_n_request'] = np.var([x['num_requests'] for x in predict_results])
    response['num_preempted'] = sum([x['num_preempted'] for x in predict_results])
    bottleneck_host = max(time_in_predictions, key=lambda x: x[0])
    response['time_on_backend'] = time_for_inference + bottleneck_host[1]
    response['time_on_probe'] = bottleneck_host[0] - bottleneck_host[1]
    return JSONResponse(response)


def build_app(args: Namespace) -> FastAPI:
    global app, n, m
    n = args.num_query_predictor
    m = args.num_required_predictor
    app.root_path = args.root_path
    return app


async def init_app(
        args: Namespace,
        instances_list: Optional[List[Instance]] = None,
) -> FastAPI:
    app = build_app(args)
    global instances, start_time, metrics_type
    config_path = args.config_path

    instance_dict = json.load(open(config_path))
    if instances_list is not None:
        instances.extend(instances_list)
    else:
        for key, value in instance_dict.items():
            instance = Instance(key, value["ip_address"], value["predictor_port"], value["backend_port"])
            instances.append(instance)
    start_time = time.time()
    metrics_type = args.metrics_type
    return app


async def run_server(args: Namespace,
                     instances_list: Optional[List[Instance]] = None,
                     **uvicorn_kwargs: Any) -> None:
    app = await init_app(args, instances_list)
    assert len(instances) > 0

    if args.debugging_logs:
        logger.setLevel(logging.DEBUG)

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        workers=args.workers,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--config_path", type=str, default="vidur/prediction/config/test_host_configs.json")
    parser.add_argument("--metrics_type", type=str, default="min_latency")
    parser.add_argument("-n", "--num_query_predictor", type=int, default=1)
    parser.add_argument("-m", "--num_required_predictor", type=int, default=1)
    parser.add_argument("--debugging_logs", type=bool, default=False)
    args = parser.parse_args()
    # in case the limited by the number of files
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    asyncio.run(run_server(args))
