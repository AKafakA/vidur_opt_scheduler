"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import argparse
import asyncio
import json
import logging
import signal
import ssl
from argparse import Namespace
from typing import Any, Optional
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vidur.prediction.predictor.predictor_config import PredictorConfig
from vidur.prediction.predictor.predictor import Predictor
from vidur.prediction.server_utils import convert_request, find_process_using_port, get_predictor
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric

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
    ret = {"metric": metric}
    print(ret)
    logging.debug("Predicted metric: %s for request: %s", metric, str(vidur_request.id))
    return JSONResponse(ret)


@app.post("/update")
async def update(request: Request) -> Response:
    """Schedule the request. """
    assert predictor is not None
    request_dict = await request.json()
    vidur_request = convert_request(request_dict)
    predictor.update(vidur_request)
    logging.debug("Scheduled request: %s", str(vidur_request.id))
    return Response(status_code=200)


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
    predictor = (instance_predictor if instance_predictor is not None else
                 get_predictor(args.predictor_type, config, instance_port))
    return app


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logging.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logging.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


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
    args = parser.parse_args()
    asyncio.run(run_server(args))
