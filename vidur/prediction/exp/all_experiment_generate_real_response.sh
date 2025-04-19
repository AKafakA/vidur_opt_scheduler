SCHEDULER_NAME="random"
BATCH_CAP=48

DATASET_NAMES="sharegpt lmsys arxiv new_sharegpt"

for dataset_name in $DATASET_NAMES; do
  if [ "$dataset_name" = "sharegpt" ]; then
    DATASET_PATH="~/data/sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=0
    N="30000"
  elif [ "$dataset_name" = "lmsys" ]; then
    DATASET_PATH="~/data/lmsys"
    DATASET_TYPE="lmsys"
    START_INDEX=0
    N="30000"
  elif [ "$dataset_name" = "arxiv" ]; then
    DATASET_PATH="~/data/arxiv"
    DATASET_TYPE="arxiv"
    START_INDEX=0
    N="30000"
  elif [ "$dataset_name" = "new_sharegpt" ]; then
    DATASET_PATH="~/data/new_sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=0
    N="30000"
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
    START_INDEX=30001
    N="20000"
  elif [ "$dataset_name" = "lmsys" ]; then
    DATASET_PATH="~/data/lmsys"
    DATASET_TYPE="lmsys"
    START_INDEX=30001
    N="30000"
  elif [ "$dataset_name" = "arxiv" ]; then
    DATASET_PATH="~/data/arxiv"
    DATASET_TYPE="arxiv"
    START_INDEX=30001
    N="20000"
  elif [ "$dataset_name" = "new_sharegpt" ]; then
    DATASET_PATH="~/data/new_sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=30001
    N="30000"
  fi
  for scheduler in $SCHEDULER_NAME; do
    echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name"
    sh vidur/prediction/exp/experiment.sh $scheduler $N true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX
  done
done

DATASET_NAMES="lmsys new_sharegpt"
for dataset_name in $DATASET_NAMES; do
  if [ "$dataset_name" = "lmsys" ]; then
    DATASET_PATH="~/data/lmsys"
    DATASET_TYPE="lmsys"
    START_INDEX=50001
    N="30000"
  elif [ "$dataset_name" = "new_sharegpt" ]; then
    DATASET_PATH="~/data/new_sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=50001
    N="30000"
  fi
  for scheduler in $SCHEDULER_NAME; do
    echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name"
    sh vidur/prediction/exp/experiment.sh $scheduler $N true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX
  done
done

for dataset_name in $DATASET_NAMES; do
  if [ "$dataset_name" = "lmsys" ]; then
    DATASET_PATH="~/data/lmsys"
    DATASET_TYPE="lmsys"
    START_INDEX=80001
    N="20000"
  elif [ "$dataset_name" = "new_sharegpt" ]; then
    DATASET_PATH="~/data/new_sharegpt"
    DATASET_TYPE="sharegpt"
    START_INDEX=80001
    N="20000"
  fi

  for scheduler in $SCHEDULER_NAME; do
    echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name"
    sh vidur/prediction/exp/experiment.sh $scheduler $N true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE true false $START_INDEX
  done
done