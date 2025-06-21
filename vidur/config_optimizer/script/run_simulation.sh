filename=$1
export PYTHONPATH=.
python vidur/config_optimizer/config_explorer/main.py --config-path ${filename} --num-threads 10 --output-dir ./simulation_output &