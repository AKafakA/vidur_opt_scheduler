BATCH_CAP=$1
MODEL=$2
UPDATE_CODE=$3
VLLM_VERSION=$4
MAX_MODEL_LENGTH=$5
ENABLE_CHUNKED_PREFILL=$6
NUM_WORKERS=$7
MAX_NUM_BATCHED_TOKEN=$8

if [ "$UPDATE_CODE" = "true" ]; then
    parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vllm && sudo chown -R asdwb .git/ && git reset --hard HEAD~1 && git pull"
fi
parallel-ssh -t 0 -h vidur/prediction/config/hosts "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && export HF_TOKEN=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 && cd vllm && mkdir -p  ~/vidur_opt_scheduler/experiment_output/logs && export VLLM_USE_V1=${VLLM_VERSION} && nohup python -m vllm.entrypoints.api_server --workers $NUM_WORKERS --model=$MODEL --max_num_seq $BATCH_CAP  --trust-remote-code --max_model_len $MAX_MODEL_LENGTH --enable_chunked_prefill $ENABLE_CHUNKED_PREFILL --max-num-batched-tokens $MAX_NUM_BATCHED_TOKEN > ~/vidur_opt_scheduler/experiment_output/logs/vllm.log 2>&1 &"
