BATCH_CAP=$1
MODEL=$2
UPDATE_CODE=$3
#MODEL="google/gemma-2b"

if [ "$UPDATE_CODE" = "true" ]; then
    parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vllm && git reset --hard HEAD~1 && git pull"
fi
parallel-ssh -t 0 -h vidur/prediction/config/hosts "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && export HF_TOKEN=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 && cd vllm && mkdir -p  ~/vidur_opt_scheduler/experiment_output/logs && nohup python -m vllm.entrypoints.api_server --model=$MODEL --max_num_seq $BATCH_CAP --dtype=half > ~/vidur_opt_scheduler/experiment_output/logs/vllm.log 2>&1 &"
