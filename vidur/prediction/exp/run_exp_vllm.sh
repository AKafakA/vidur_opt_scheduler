BATCH_CAP=$1
MODEL=$2
UPDATE_CODE=$3
#MODEL="google/gemma-2b"

if [ "$UPDATE_CODE" = "true" ]; then
    parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vllm && git reset --hard HEAD~1 && git pull"
fi
parallel-ssh -i -t 0 -h vidur/prediction/config/hosts "export HF_TOKEN=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 && cd vllm && nohup python -m vllm.entrypoints.api_server --model=$MODEL --max_num_seq $BATCH_CAP > ~/vidur_opt_scheduler/experiment_output/logs/vllm.log 2>&1 &"
