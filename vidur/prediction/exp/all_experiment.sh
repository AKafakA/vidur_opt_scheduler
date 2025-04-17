SCHEDULER_NAME="random"
BATCH_CAP=48

for scheduler in $SCHEDULER_NAME; do
  echo "Running experiment for scheduler: $scheduler"
  sh vidur/prediction/exp/experiment.sh $scheduler false true $BATCH_CAP
done
