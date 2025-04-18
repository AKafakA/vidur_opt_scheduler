SCHEDULER_NAME="random"
BATCH_CAP=48

#DATASET_NAMES="sharegpt lmsys arxiv new_sharegpt"
DATASET_NAMES="arxiv"

for dataset_name in $DATASET_NAMES; do
  if [ "$dataset_name" = "sharegpt" ]; then
    DATASET_PATH="~/data/sharegpt/$dataset_name.jsonl"
    DATASET_TYPE="sharegpt"
    N="50"
  elif [ "$dataset_name" = "lmsys" ]; then
    DATASET_PATH="~/data/lmsys/mlsys_1.parquet"
    DATASET_TYPE="lmsys"
    N="100"
  elif [ "$dataset_name" = "arxiv" ]; then
    DATASET_PATH="~/data/arxiv/arxiv_1.parquet"
    DATASET_TYPE="arxiv"
    N="50"
  elif [ "$dataset_name" = "new_sharegpt" ]; then
    DATASET_PATH="~/data/new_sharegpt/sharegpt_1.json;~/data/new_sharegpt/sharegpt_2.json"
    DATASET_TYPE="sharegpt"
    N="100"
  fi
  for scheduler in $SCHEDULER_NAME; do
    echo "Running experiment for scheduler: $scheduler with dataset: $dataset_name"

    sh vidur/prediction/exp/experiment.sh $scheduler $N true $BATCH_CAP $dataset_name $DATASET_PATH $DATASET_TYPE
  done
done
