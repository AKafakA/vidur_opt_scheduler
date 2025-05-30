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

profiling_sampling_rate = 0.001
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
instances = []
num_requests = 0
num_probed_instance = 0
num_min_probed = 0
start_time = 0
metrics_type = "random"
logging.basicConfig(level=logging.INFO,
                    filemode='a+',
                    filename='experiment_output/logs/predictor_output.log')
logger = logging.getLogger(__name__)

correct_flags = []
average_gaps = []
random_assigned = []

sampled_mean_error_ratios = []
sampled_predict_accuracies = []


def print_instance_errors():
    errors = []
    error_ratios = []
    real_response_serving_time = []
    global instances
    for instance in instances:
        errors.extend(instance.predicted_error)
        error_ratios.extend(instance.predicted_error_ratio)
        real_response_serving_time.extend(instance.serving_time)
    print(real_response_serving_time)
    total_s = 0.0
    total_p = 0.0
    count = 0.0
    for serving, predicted in real_response_serving_time:
        total_s = total_s + serving
        total_p = total_p + predicted
        count = count + 1

    predict_accuracy = (1.0 * sum(correct_flags)) / len(correct_flags)

    if count > 0 and error_ratios and correct_flags:    #
        print(f"average serving {total_s / count} and average predicted {total_p / count}")
        print(f"Mean of Prediction error ratio {np.mean(error_ratios)}")
        print(f"P50 of Prediction error ratio {np.percentile(error_ratios, 50)}")
        print(f"Predict accuracy: {predict_accuracy} for compare")
        return predict_accuracy, np.mean(error_ratios)
    else:
        print("No serving time or error ratios collected.")
        return 0.0, 0.0


@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    """Generate completion for the request with profiling.
    This API will 1) calling the predictor to predict the completion time of the request,
    2) select the host based on target metrics and call its vllm generate API to generate the completion.
    3) return the completion to the client with profiling
    """
    request_dict = await request.json()
    request_id = request_dict["request_id"]
    prompt = request_dict.pop("prompt")
    num_context_tokens = request_dict.pop("prompt_len")
    predicted_num_decode_tokens = request_dict.pop("predicted_response_len")
    max_response_len = request_dict.pop("max_response_len")
    arrived_at = time.time() - start_time
    _ = request_dict.pop("stream", False)
    predict_tasks = []
    # if num_sampled_requests > 0:
    #     is_sampled_for_compare = request_id % num_sampled_requests == 0
    # else:
    #     is_sampled_for_compare = False
    global profiling_sampling_rate
    random_flag = random.uniform(0, 1)
    is_sampled_for_compare = random_flag <= profiling_sampling_rate
    random_selected_instances = random.sample(instances, min(num_probed_instance, len(instances)))
    for instance in random_selected_instances:
        predict_tasks.append(instance.query_predictor(
            request_id, num_context_tokens, predicted_num_decode_tokens, arrived_at))
    try:
        predict_results = await asyncio.gather(*predict_tasks)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse({"error": "Prediction failed"}, status_code=500)

    predict_results = [x for x in predict_results if x['gpu_blocks'] > 0]
    single_metric = {'sampled_avg_gpu_blocks': np.mean([x['gpu_blocks'] for x in predict_results]),
                     'sampled_var_gpu_blocks': np.var([x['gpu_blocks'] for x in predict_results]),
                     'sampled_avg_n_request': np.mean([x['num_requests'] for x in predict_results]),
                     'sampled_var_n_request': np.var([x['num_requests'] for x in predict_results]),
                     'num_preempted': sum([x['num_preempted'] for x in predict_results])}

    if len(predict_results) == 0:
        selected_index = random.randint(0, len(instances) - 1)
        selected_instance = instances[selected_index]
        try:
            response = await selected_instance.query_backend(prompt, max_response_len, request_id,
                                                             predicted_num_decode_tokens)
            global random_assigned
            random_assigned.append((request_id, selected_instance._instance_id))
        except Exception as e:
            print(f"Error during prediction: {e}")
            return JSONResponse({"error": "Prediction failed"}, status_code=500)
        print(f"Randomly assigned request {request_id} to instance {selected_instance._instance_id}, "
              f"current number of randomly assigned requests: {len(random_assigned)} at time {time.time() - start_time}")
        return JSONResponse(response)

    target_metrics = [x['target_metric'] for x in predict_results]
    assert len(target_metrics) == len(predict_results)

    if is_sampled_for_compare:
        predicted_sampled_results = [(result['instance_id'], result['target_metric']) for result in predict_results]
        # only one profile sampling is allowed and will block the other normal request
        sampled_instanced = [x['instance_id'] for x in predict_results]
        responses = await asyncio.gather(*[instance.query_backend(
            prompt, max_response_len, request_id, predicted_num_decode_tokens)
            for instance in instances if instance._instance_id in sampled_instanced])

        serving_times = [(response['instance_id'], response["serving_time"]) for response in responses]

        instance_id_with_least_serving_time = min(serving_times, key=lambda x: x[1])[0]
        instance_id_with_least_predicted_time = min(predicted_sampled_results, key=lambda x: x[1])[0]
        min_predicted_time = min(predicted_sampled_results, key=lambda x: x[1])[1]

        sorted_instances_serving_time = sorted(serving_times, key=lambda x: x[1])

        gap_between_average_serving_time_and_least_predicted_time = \
            np.mean([x[1] for x in sorted_instances_serving_time]) - min_predicted_time

        gap_between_min_serving_time_and_least_predicted_time = \
            min([x[1] for x in sorted_instances_serving_time]) - min_predicted_time

        gap_between_random_serving_time_and_least_predicted_time = \
            random.choice([x[1] for x in sorted_instances_serving_time]) - min_predicted_time

        gap_info = {
            "average": gap_between_average_serving_time_and_least_predicted_time,
            "min": gap_between_min_serving_time_and_least_predicted_time,
            "random": gap_between_random_serving_time_and_least_predicted_time
        }

        if instance_id_with_least_serving_time == instance_id_with_least_predicted_time:
            correct_flags.append(True)
        else:
            correct_flags.append(False)
        average_gaps.append(gap_info)
        response = random.choice(responses)
        global sampled_mean_error_ratios, sampled_predict_accuracies
        sampled_predict_accuracy, sampled_error_ratio = print_instance_errors()
        sampled_mean_error_ratios.append(sampled_error_ratio)
        sampled_predict_accuracies.append(sampled_predict_accuracy)
        response["sampled_mean_error_ratio"] = sampled_error_ratio
        response["sampled_predict_accuracy"] = sampled_predict_accuracy

        response["sampled_serving_latencies"] = [serving_times[i][1] for i in range(len(serving_times))]
        response["min_predicted_latency"] = min_predicted_time
    else:
        if metrics_type.startswith("min") or metrics_type.startswith("max"):
            # if current in metrics means all node need to be queried and select the one with min/max
            # as just report the current value without prediction is cheap and sample no need to be limited
            if metrics_type.startswith("min"):
                target_metric = min(target_metrics)
            else:
                target_metric = max(target_metrics)
            candidates_indexes = [i for i, value in enumerate(target_metrics) if value == target_metric]
            metric_selected_index = random.choice(candidates_indexes)
            selected_instance_id = (predict_results[metric_selected_index])['instance_id']
            selected_index = \
                [i for i, instance in enumerate(instances) if selected_instance_id == instance._instance_id][0]
        elif metrics_type == "random":
            selected_index = random.randint(0, len(instances) - 1)
        elif metrics_type == "round_robin":
            selected_index = int(request_id) % len(instances)
        elif metrics_type == "request_per_seconds":
            instance_qpm = [(instance.get_current_qpm(), instance._instance_id) for instance in instances]
            min_qpm = min(instance_qpm, key=lambda x: x[0])[0]
            selected_instance_id = random.choice([x[1] for x in instance_qpm if x[0] == min_qpm])
            selected_index = [i for i in range(len(instances)) if instances[i]._instance_id == selected_instance_id][0]
        else:
            raise ValueError(f"Invalid metrics type: {metrics_type}")
        selected_instance = instances[selected_index]
        try:
            response = await selected_instance.query_backend(prompt, max_response_len, request_id,
                                                             predicted_num_decode_tokens)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return JSONResponse({"error": "Prediction failed"}, status_code=500)
    for key, value in single_metric.items():
        response[key] = value
    return JSONResponse(response)


def build_app(args: Namespace) -> FastAPI:
    global app, num_probed_instance, num_min_probed, profiling_sampling_rate
    num_min_probed = args.num_required_predictor
    num_probed_instance = args.num_query_predictor
    profiling_sampling_rate = args.profiling_sampling_rate
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
            ports = []
            if args.num_predictor_ports > 0:
                for i in range(min(args.num_predictor_ports, len(value["predictor_ports"]))):
                    ports.append(value["predictor_ports"][i])
            print(f"instance {key} with ip {value['ip_address']} and predictor port {ports}")
            instance = Instance(key, value["ip_address"], ports, value["backend_port"],
                                query_predictor_timeout=args.predictor_timeout,
                                query_backend_timeout=args.backend_timeout)
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
    parser.add_argument("--debugging_logs", type=bool, default=True)
    parser.add_argument("--profiling_sampling_rate", type=float, default=0.001)
    parser.add_argument("--num_predictor_ports", type=int, default=-1)
    parser.add_argument("--predictor_timeout", type=int, default=10)
    parser.add_argument("--backend_timeout", type=int, default=1800)
    args = parser.parse_args()
    logger.info("Starting server with args: %s", str(args))
    # in case the limited by the number of files
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    asyncio.run(run_server(args))
