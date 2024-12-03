from typing import Optional, Dict

import psutil
from vidur.entities import Request as VidurRequest, Request
from vidur.prediction.predictor.predictor_config import PredictorConfig
from vidur.prediction.predictor.dummy_predictor import DummyPredictor
from vidur.prediction.predictor.simulate_predictor import SimulatePredictor


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def convert_request(request: Dict) -> VidurRequest:
    request_id = request["id"]
    arrival_time = request["arrival_time"]
    num_context_tokens = request["num_context_tokens"]
    num_decode_tokens = request["num_decode_tokens"]
    vidur_request = VidurRequest(arrival_time, num_context_tokens, num_decode_tokens)
    vidur_request.set_id(request_id)
    return vidur_request


def get_predictor(type_str: str, predictor_config: PredictorConfig, instance_port: int == -1):
    if type_str == "dummy":
        return DummyPredictor(predictor_config, instance_port)
    elif type_str == "simulate":
        return SimulatePredictor(predictor_config, instance_port)



