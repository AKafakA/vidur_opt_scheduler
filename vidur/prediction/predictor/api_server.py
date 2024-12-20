import argparse
import asyncio
import json
import logging
import ssl
from argparse import Namespace
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vidur.prediction.predictor.predictor_config import PredictorConfig
from vidur.prediction.predictor.predictor import Predictor
from vidur.prediction.server_utils import convert_request, get_predictor, serve_http
import resource

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/predict")
async def predict(request: Request) -> Response:
    """Predict completion for the request. """
    assert predictor is not None
    request_dict = await request.json()
    vidur_request = convert_request(request_dict)
    metric = predictor.predict(vidur_request)
    logging.debug("Predicted metric: %s for request: %s", metric, str(vidur_request.id))
    return JSONResponse(metric)


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def init_app(
        args: Namespace,
        instance_predictor: Optional[Predictor] = None,
) -> FastAPI:
    app = build_app(args)
    instance_port = args.instance_port
    global predictor
    config_path = args.config_path
    config_dict = json.load(open(config_path))
    config: PredictorConfig = PredictorConfig.create_from_dict(config_dict)
    config.target_metric = args.metric_type
    config.disable_batch_time_estimation = args.disable_time_estimation
    predictor = (instance_predictor if instance_predictor is not None else
                 get_predictor(args.predictor_type, config, instance_port))
    return app


async def run_server(args: Namespace,
                     instance_predictor: Optional[Predictor] = None,
                     **uvicorn_kwargs: Any) -> None:
    app = await init_app(args, instance_predictor)
    assert predictor is not None

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
    parser.add_argument("--port", type=int, default=8100)
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
    parser.add_argument("--instance-port", type=int, default=8000)
    parser.add_argument("--config_path", type=str, default= "vidur/prediction/config/test_config.json")
    parser.add_argument("--predictor_type", type=str, default="simulate")
    parser.add_argument("--metric_type", type=str, default="random")
    parser.add_argument("--disable_time_estimation", type=bool, default=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    asyncio.run(run_server(args))
