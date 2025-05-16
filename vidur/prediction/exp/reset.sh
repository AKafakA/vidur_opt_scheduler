parallel-ssh -i -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && rm -rf experiment_output/logs/* && mkdir -p experiment_output/logs"

parallel-ssh -h vidur/prediction/config/hosts "pkill -f vllm.entrypoints.api_server"
parallel-ssh -h vidur/prediction/config/hosts "pkill -f predictor"
parallel-ssh -h vidur/prediction/config/hosts "pkill -f multiprocessing.spawn"