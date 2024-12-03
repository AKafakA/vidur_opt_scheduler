import argparse
import json
import time
from typing import List

import requests

from vidur.config import FixedRequestLengthGeneratorConfig, PoissonRequestIntervalGeneratorConfig, \
    SyntheticRequestGeneratorConfig
from vidur.request_generator.synthetic_request_generator import SyntheticRequestGenerator


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["metric"]
    return output

def post_request(api_url,
                 request_id: int, num_context_tokens: int, num_decode_tokens: int,
                 arrived_at: float,
                 stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "id": request_id,
        "arrival_time": arrived_at,
        "num_context_tokens": num_context_tokens,
        "num_decode_tokens": num_decode_tokens,
    }
    print(requests)
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--num_request", type=int, default=10)
    parser.add_argument("--qps", type=int, default=1)

    args = parser.parse_args()
    predict_api_url = f"http://{args.host}:{args.port}/predict"
    update_api_url = f"http://{args.host}:{args.port}/update"

    # can be replaced by other length config such as uniform, zipfian, trace. Check the config.py for more details
    length_generator_config = FixedRequestLengthGeneratorConfig(10, 10)
    request_interval_config = PoissonRequestIntervalGeneratorConfig(args.qps)

    request_generator_config = SyntheticRequestGeneratorConfig(num_requests=args.num_request,
                                                               length_generator_config=length_generator_config,
                                                               interval_generator_config=request_interval_config)

    request_generator = SyntheticRequestGenerator(request_generator_config)
    generate_requests = request_generator.generate_requests()
    for request in generate_requests:
        res = post_request(predict_api_url,
                           request_id=request.id,
                           num_context_tokens=request.num_prefill_tokens,
                           num_decode_tokens=request.num_decode_tokens, arrived_at=request.arrived_at)
        output = get_response(res)
        print(output)

        time.sleep(1 / args.qps)
        post_request(update_api_url,
                     request_id=request.id,
                     num_context_tokens=request.num_prefill_tokens,
                     num_decode_tokens=request.num_decode_tokens, arrived_at=request.arrived_at)

        # send request to the prediction API
        # send request to the update API
        # sleep for the interval time
