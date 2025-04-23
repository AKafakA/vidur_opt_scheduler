SCHEDULER_NAME="random"
BATCH_CAP=48
MODEL="meta-llama/Llama-2-7b-hf Qwen/Qwen-7B"
DATASET_NAMES="sharegpt lmsys"
N=50
START_INDEX=0

for model in $MODEL; do
  for dataset_name in $DATASET_NAMES; do
    if [ "$dataset_name" = "sharegpt" ]; then
      DATASET_PATH="~/data/sharegpt"
      DATASET_TYPE="sharegpt"
    elif [ "$dataset_name" = "lmsys" ]; then
      DATASET_PATH="~/data/lmsys"
      DATASET_TYPE="lmsys"
    fi

    for scheduler in $SCHEDULER_NAME; do
      echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name"
      sh vidur/prediction/exp/experiment.sh $scheduler $N true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX $model
    done
  done
done