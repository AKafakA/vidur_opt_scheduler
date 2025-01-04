SCHEDULER_NAME = "min_latency min_current_gpu_blocks min_current_requests min_gpu_blocks"
BATCH_CAP = 48

for scheduler in $SCHEDULER_NAME; do
  echo "Running experiment for scheduler: $scheduler"
  sh vidur/prediction/exp/experiment.sh $scheduler false true $BATCH_CAP
done
