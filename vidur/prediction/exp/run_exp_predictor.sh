CONFIG_PATH=$1
METRIC_TYPE=$2
DISABLE_TIME_ESTIMATION=$3
UPDATE_CODE=$4
BATCH_CAP=$5

if [ "$UPDATE_CODE" = "true" ]; then
    parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && git reset --hard HEAD~1 && git pull"
fi
parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && export PYTHONPATH=. && nohup python vidur/prediction/predictor/api_server.py --config_path $CONFIG_PATH --metric_type $METRIC_TYPE --disable_time_estimation $DISABLE_TIME_ESTIMATION --batch_size_cap $BATCH_CAP > experiment_output/logs/predictor.log 2>&1 &"
