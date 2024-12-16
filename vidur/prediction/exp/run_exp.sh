parallel-ssh -h vidur/prediction/config/hosts "pkill -f vllm"
parallel-ssh -h vidur/prediction/config/hosts "export HF_TOKEN=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 && cd vllm && nohup python -m vllm.entrypoints.api_server --model=google/gemma-2b 1>service.out  2>/dev/null &"

parallel-ssh -h vidur/prediction/config/hosts "pkill -f vidur"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && export PYTHONPATH=. && nohup python vidur/prediction/predictor/api_server.py 1>service.out  2>/dev/null &"
