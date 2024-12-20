BATCH_CAP=128
MODEL="meta-llama/Llama-2-7b-hf"
#MODEL="google/gemma-2b"

parallel-ssh -i -t 0 -h vidur/prediction/config/hosts "export HF_TOKEN=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 && cd vllm && nohup python -m vllm.entrypoints.api_server --model=$MODEL --max_num_seq $BATCH_CAP > vllm_output.log 2>&1 &"
