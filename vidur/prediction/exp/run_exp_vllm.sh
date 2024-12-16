BATCH_CAP=128
MODEL="meta-llama/Llama-2-7b-hf"

parallel-ssh -h vidur/prediction/config/hosts "pkill -f vllm"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "export HF_TOKEN=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 && cd vllm && nohup python -m vllm.entrypoints.api_server --model=$MODEL --max_num_seq $BATCH_CAP 1>service.out  2>/dev/null &"
