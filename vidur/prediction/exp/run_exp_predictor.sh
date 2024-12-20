CONFIG_PATH="vidur/prediction/config/test_config.json"
METRIC_TYPE="min_latency"
DISABLE_TIME_ESTIMATION="false"

#parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && git reset --hard HEAD~1 && git pull"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && export PYTHONPATH=. && nohup python vidur/prediction/predictor/api_server.py --config_path $CONFIG_PATH --metric_type $METRIC_TYPE --disable_time_estimation $DISABLE_TIME_ESTIMATION > experiment_output/logs/vllm_output.log 2>&1 &"
