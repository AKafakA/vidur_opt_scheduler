from typing import Optional, Dict

import psutil
from vidur.config import SimulationConfig
from vidur.entities import Request as VidurRequest
from vidur.prediction.predictor.predictor import Predictor


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def convert_request(request: Dict) -> VidurRequest:
    pass


def get_predictor(type_str: str, simulation_config: SimulationConfig, instance_port: int == -1) -> Predictor:
    if type_str == "dummy":
        return Predictor(simulation_config, instance_port)

