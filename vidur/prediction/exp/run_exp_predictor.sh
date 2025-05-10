CONFIG_PATH=$1
METRIC_TYPE=$2
ENABLE_TIME_ESTIMATION=$3
UPDATE_CODE=$4
BATCH_CAP=$5
ENABLE_CHUNKED_PREFILL=$6
NUM_WORKERS=$7
BRANCH_NAME=$8
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION=$9

if [ "$UPDATE_CODE" = "true" ]; then
    parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && git checkout $BRANCH_NAME && git pull"
    parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && git reset --hard HEAD~1 && git pull"
fi

APPEND_CHUNKED_PREFILL=""
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    APPEND_CHUNKED_PREFILL="--enable_chunked_prefill"
fi

parallel-ssh -i -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && export PYTHONPATH=. && nohup python vidur/prediction/predictor/api_server.py --config_path $CONFIG_PATH --metric_type $METRIC_TYPE --enable_time_estimation $ENABLE_TIME_ESTIMATION --batch_size_cap $BATCH_CAP --workers $NUM_WORKERS $APPEND_CHUNKED_PREFILL --threshold_batch_size_for_time_estimation $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION > experiment_output/logs/predictor.log 2>&1 &"
sleep 60