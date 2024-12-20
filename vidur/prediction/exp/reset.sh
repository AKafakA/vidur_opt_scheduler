parallel-ssh -h vidur/prediction/config/hosts "pkill -f vllm"
parallel-ssh -h vidur/prediction/config/hosts "pkill -f predictor"
parallel-ssh -h vidur/prediction/config/hosts "pkill -f global_scheduler"