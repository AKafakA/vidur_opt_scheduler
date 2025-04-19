SCHEDULER_NAME="random"
BATCH_CAP=48

DATASET_NAMES="sharegpt lmsys arxiv new_sharegpt"
DATASET_NAMES="sharegpt"

for dataset_name in $DATASET_NAMES; do
  if [ "$dataset_name" = "sharegpt" ]; then
    DATASET_PATH="~/data/sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=0
    N="50"
  elif [ "$dataset_name" = "lmsys" ]; then
    DATASET_PATH="~/data/lmsys"
    DATASET_TYPE="lmsys"
    START_INDEX=0
    N="100"
  elif [ "$dataset_name" = "arxiv" ]; then
    DATASET_PATH="~/data/arxiv"
    DATASET_TYPE="arxiv"
    START_INDEX=0
    N="50"
  elif [ "$dataset_name" = "new_sharegpt" ]; then
    DATASET_PATH="~/data/new_sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=0
    N="100"
  fi

  for scheduler in $SCHEDULER_NAME; do
    echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name"
    sh vidur/prediction/exp/experiment.sh $scheduler $N true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX
  done
done

for dataset_name in $DATASET_NAMES; do
  if [ "$dataset_name" = "sharegpt" ]; then
    DATASET_PATH="~/data/sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=51
    N="50"
  elif [ "$dataset_name" = "lmsys" ]; then
    DATASET_PATH="~/data/lmsys"
    DATASET_TYPE="lmsys"
    START_INDEX=101
    N="100"
  elif [ "$dataset_name" = "arxiv" ]; then
    DATASET_PATH="~/data/arxiv"
    DATASET_TYPE="arxiv"
    START_INDEX=51
    N="50"
  elif [ "$dataset_name" = "new_sharegpt" ]; then
    DATASET_PATH="~/data/new_sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=101
    N="100"
  fi

  for scheduler in $SCHEDULER_NAME; do
    echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name"
    sh vidur/prediction/exp/experiment.sh $scheduler $N true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX
  done
done
