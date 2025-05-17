START_INDEX=0
BATCH_CAP=48
TARGET_HOST='asdwb@d7525-10s10309.wisc.cloudlab.us'
PREDICTOR_WORKERS=8
GLOBAL_SCHEDULER_WORKERS=2
BACKEND_WORKERS=4
MAX_MODEL_LENGTH=4096
CHUNK_SIZE=512
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="single_predictor_evaluation"
USE_PROCESS_FOR_FRONTEND=true
UPDATE_VIDUR_CODE=true
UPDATE_VLLM_CODE=true
RUN_EXP=true
RESTART_VLLM=true

# Config for end to end experiment
#SCHEDULER_NAME="min_new_request_latency random round_robin min_infass_load request_per_seconds min_infass_load"
#MODEL="meta-llama/Llama-2-7b-hf Qwen/Qwen-7B"
#DATASET_NAMES="sharegpt lmsys"
#QPS="24"
#N_SELECTED="12 2"
#PROFILING_SAMPLE_RATE="0.0 0.1"
#USE_FOR_PROFILING_ONLY="false"
#NUM_REQUEST=10000
ENABLE_CHUNKED_PREFILL="true"

# Config for single predictor experiment
SCHEDULER_NAME="min_new_request_latency"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
QPS="24"
N_SELECTED="12"
PROFILING_SAMPLE_RATE="0.1"
USE_FOR_PROFILING_ONLY="true"
NUM_REQUEST=5


for model in $MODEL; do
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen-7B" ]; then
    MODEL_TYPE="qwen"
  fi
  for dataset_name in $DATASET_NAMES; do
    if [ "$dataset_name" = "sharegpt" ]; then
      DATASET_PATH="~/data/sharegpt"
      DATASET_TYPE="sharegpt"
    elif [ "$dataset_name" = "lmsys" ]; then
      DATASET_PATH="~/data/lmsys"
      DATASET_TYPE="lmsys"
    fi
    for scheduler in $SCHEDULER_NAME; do
      for enable_chunked_prefill in $ENABLE_CHUNKED_PREFILL; do
        for qps in $QPS; do
          echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name and model: $model with qps: $qps and chunked prefill: $enable_chunked_prefill and predictor workers: $PREDICTOR_WORKERS"
          sh vidur/prediction/exp/experiment.sh $scheduler $NUM_REQUEST $RESTART_VLLM  $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX $model $MODEL_TYPE $MAX_MODEL_LENGTH $TARGET_HOST $enable_chunked_prefill $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $CHUNK_SIZE $qps $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $N_SELECTED $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND $UPDATE_VIDUR_CODE $UPDATE_VLLM_CODE $RUN_EXP
        done
      done
    done
  done
done

#mkdir -p ~/vidur_opt_scheduler/single_node_experiment_output/
#scp -r $TARGET_HOST:~/vidur_opt_scheduler/experiment_output/* ~/vidur_opt_scheduler/single_node_experiment_output/.