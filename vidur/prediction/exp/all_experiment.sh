SCHEDULER_NAME="min_scheduling_delay round_robin min_infass_load min_latency random request_per_seconds"
BATCH_CAP=48

for scheduler in $SCHEDULER_NAME; do
  echo "Running experiment for scheduler: $scheduler"
  sh vidur/prediction/exp/experiment.sh $scheduler false true $BATCH_CAP
done
