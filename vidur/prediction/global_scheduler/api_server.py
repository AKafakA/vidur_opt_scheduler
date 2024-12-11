import argparse
import asyncio
import json
import random
import ssl
import time
from argparse import Namespace
from typing import Any, Optional, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vidur.prediction.global_scheduler.instance import Instance
from vidur.prediction.server_utils import serve_http
import resource

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
instances = []
num_requests = 0
n = 0
m = 0
start_time = 0
metrics_type = "min_latency"


@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    """Generate completion for the request with profilling.
    This API will 1) calling the predictor to predict the completion time of the request,
    2) select the host based on target metrics and call its vllm generate API to generate the completion.
    3) return the completion to the client with profilling
    """
    assert len(instances) > 0
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    num_context_tokens = request_dict.pop("prompt_len")
    num_decode_tokens = request_dict.pop("expected_response_len")
    arrived_at = time.time() - start_time
    _ = request_dict.pop("stream", False)
    global num_requests
    request_id = num_requests
    num_requests += 1
    predict_tasks = []

    for instance in random.sample(instances, n):
        predict_tasks.append(instance.query_predictor(
            request_id, num_context_tokens, num_decode_tokens, arrived_at))

    predict_results = await asyncio.gather(*predict_tasks)
    if metrics_type.startswith("min"):
        selected_instances = instances[predict_results.index(min(predict_results))]
    elif metrics_type.startswith("max") or metrics_type == "random":
        selected_instances = instances[predict_results.index(max(predict_results))]
    elif metrics_type == "round_robin":
        selected_instances = instances[num_requests % len(instances)]
    else:
        raise ValueError(f"Invalid metrics type: {metrics_type}")
    response = await selected_instances.query_backend(prompt, num_decode_tokens)
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

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8200)
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
    parser.add_argument("--num_query_predictor", type=int, default=1)
    parser.add_argument("--num_required_predictor", type=int, default=1)
    args = parser.parse_args()
    # in case the limited by the number of files
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    asyncio.run(run_server(args))
