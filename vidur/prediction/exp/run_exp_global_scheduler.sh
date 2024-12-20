HOST=$1
M=$2
N=$3
METRICS_TYPE=$4
CONFIG_PATH=$5

parallel-ssh -t 0 --host $HOST "pkill -f global_scheduler"
parallel-ssh -t 0 --host $HOST "cd vidur_opt_scheduler && export PYTHONPATH=. && nohup python vidur/prediction/global_scheduler/api_server.py --config_path $CONFIG_PATH --metrics_type $METRICS_TYPE --num_query_predictor $M --num_required_predictor $N > experiment_output/logs/global_scheduler.log 2>&1 &"