SCHEDULER_NAME="min_new_request_latency"
BATCH_CAP=48
#MODEL="Qwen/Qwen-7B"
MODEL="meta-llama/Llama-2-7b-hf"
MAX_MODEL_LENGTH=4096
DATASET_NAMES="sharegpt"
NUM_REQUEST=10
START_INDEX=0
TARGET_HOST='asdwb@d7525-10s10309.wisc.cloudlab.us'
ENABLE_CHUNKED_PREFILL="true"

PREDICTOR_WORKERS=4
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=4
CHUNK_SIZE=512
QPS="24"
BRANCH_NAME="single_predictor_evaluation"
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION=128
PREDICTOR_PORTS="8100 8300"

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
          echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name and model: $model with qps: $qps and chunked prefill: $enable_chunked_prefill"
          sh vidur/prediction/exp/experiment.sh $scheduler $NUM_REQUEST true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX $model $MODEL_TYPE $MAX_MODEL_LENGTH $TARGET_HOST $enable_chunked_prefill $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $CHUNK_SIZE $qps $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_PORTS
        done
      done
    done
  done
done

#mkdir -p ~/vidur_opt_scheduler/single_node_experiment_output/
#scp -r $TARGET_HOST:~/vidur_opt_scheduler/experiment_output/* ~/vidur_opt_scheduler/single_node_experiment_output/.