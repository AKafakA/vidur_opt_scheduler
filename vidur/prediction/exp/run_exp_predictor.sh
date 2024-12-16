CONFIG_PATH="vidur/prediction/config/test_config.json"

parallel-ssh -h vidur/prediction/config/hosts "pkill -f vidur"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && export PYTHONPATH=. && nohup python vidur/prediction/predictor/api_server.py ----config_path $CONFIG_PATH &"