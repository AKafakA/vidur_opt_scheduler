SCHEDULER_NAME="min_latency"
BATCH_CAP=48

for scheduler in $SCHEDULER_NAME; do
  echo "Running experiment for scheduler: $scheduler"
  sh vidur/prediction/exp/experiment.sh $scheduler true true $BATCH_CAP
done
