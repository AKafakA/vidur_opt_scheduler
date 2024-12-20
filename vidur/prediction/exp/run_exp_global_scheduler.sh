HOST='asdwb@d7525-10s10327.wisc.cloudlab.us'
M=12
N=12
METRICS_TYPE='min_latency'
CONFIG_PATH='vidur/prediction/config/host_configs.json'

parallel-ssh -t 0 --host $HOST "pkill -f global_scheduler"
parallel-ssh -t 0 --host $HOST "rm -rf vidur_opt_scheduler/*.log  && rm -rf vidur_opt_scheduler/*.png"
parallel-ssh -t 0 --host $HOST "cd vidur_opt_scheduler && export PYTHONPATH=. && nohup python vidur/prediction/global_scheduler/api_server.py --config_path $CONFIG_PATH --metrics_type $METRICS_TYPE --num_query_predictor $M --num_required_predictor $N > experiment_output/logs/global_scheduler_output.log 2>&1 &"