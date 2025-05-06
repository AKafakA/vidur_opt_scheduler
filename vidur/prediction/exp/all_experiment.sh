SCHEDULER_NAME="min_new_request_latency"
BATCH_CAP=48
#MODEL="Qwen/Qwen-7B"
MODEL="meta-llama/Llama-2-7b-hf"
MAX_MODEL_LENGTH=4096
DATASET_NAMES="sharegpt"
NUM_REQUEST=10
START_INDEX=0
TARGET_HOST='asdwb@d7525-10s10325.wisc.cloudlab.us'
ENABLE_CHUNKED_PREFILL=true

PREDICTOR_WORKERS=4
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=4

CHUNK_SIZE=512

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
      echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name and model: $model"
      sh vidur/prediction/exp/experiment.sh $scheduler $NUM_REQUEST true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX $model $MODEL_TYPE $MAX_MODEL_LENGTH $TARGET_HOST $ENABLE_CHUNKED_PREFILL $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $CHUNK_SIZE
    done
  done
done