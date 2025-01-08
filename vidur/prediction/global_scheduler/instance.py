from datetime import datetime

import aiohttp

from vidur.prediction.server_utils import post_predicting_request, get_predicting_response

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60 * 10)


class Instance:
    def __init__(self, instance_id,
                 ip_address,
                 predictor_port,
                 backend_port):
        self._instance_id = instance_id
        self._predictor_port = predictor_port
        self._backend_port = backend_port
        self._predictor_url = f"http://{ip_address}:{predictor_port}/predict"
        self._backend_url = f"http://{ip_address}:{backend_port}/generate_benchmark"
        self.ip_address = ip_address
        self.latest_response_time = None
        self.total_request = 0
        self.total_tokens = 0

    def __str__(self):
        return (f"Instance {self._instance_id} with predictor port {self._predictor_port} "
                f"and backend port {self._backend_port}")

    async def query_predictor(self, request_id: int,
                              num_context_tokens: int,
                              num_decode_tokens: int,
                              arrived_at: float):
        predict_parameters = {
            "id": request_id,
            "arrival_time": arrived_at,
            "num_context_tokens": num_context_tokens,
            "num_decode_tokens": num_decode_tokens,
        }
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(self._predictor_url, json=predict_parameters) as response:
                response_dict = await response.json()
                response_dict['instance_id'] = self._instance_id
                return response_dict

    async def query_backend(self, prompt: str, expected_response_len: int, request_id: int):
        self.total_request += 1
        start_time = datetime.now()
        output_len = expected_response_len
        request_dict = {
            "prompt": prompt,
            "n": 1,
            "best_of": 1,
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": max(output_len, 1),
            "ignore_eos": True,
            "stream": False,
            "request_id": str(request_id)
        }
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(self._backend_url, json=request_dict) as response:
                response_dict = await response.json()
        response_time = datetime.now() - start_time
        self.latest_response_time = response_time
        self.total_tokens += len(response_dict['per_token_latency'])
        return response_dict
