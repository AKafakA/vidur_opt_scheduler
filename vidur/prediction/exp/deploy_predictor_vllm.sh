#!/bin/bash

case "$1" in
    -d|--daemon)
        $0 < /dev/null &> /dev/null & disown
        exit 0
        ;;
    *)
        ;;
esac

rm -rf experiment_output
mkdir -p experiment_output/logs
sh vidur/prediction/exp/reset.sh
nohup sh vidur/prediction/exp/run_exp_vllm.sh > /dev/null 2>&1 &
nohup sh vidur/prediction/exp/run_exp_predictor.sh > /dev/null 2>&1 &