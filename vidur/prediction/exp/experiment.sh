#!/bin/bash

METRIC_TYPE="min_latency"

BATCH_CAP=128
MODEL="meta-llama/Llama-2-7b-hf"

TARGET_HOST='asdwb@d7525-10s10327.wisc.cloudlab.us'
M=12
N=12
HOST_CONFIG_PATH='vidur/prediction/config/host_configs.json'
PREDICTOR_CONFIG_PATH="vidur/prediction/config/test_config.json"
DISABLE_TIME_ESTIMATION=false
UPDATE_VIDUR_CODE=true
UPDATE_VLLM_CODE=false

case "$1" in
    -d|--daemon)
        $0 < /dev/null &> /dev/null & disown
        exit 0
        ;;
    *)
        ;;
esac


#rm -rf experiment_output
#mkdir -p experiment_output/logs
sh vidur/prediction/exp/reset.sh
nohup sh vidur/prediction/exp/run_exp_vllm.sh $BATCH_CAP $MODEL $UPDATE_UPDATE_VLLM_CODE > /dev/null 2>&1 &
nohup sh vidur/prediction/exp/run_exp_predictor.sh $PREDICTOR_CONFIG_PATH $METRIC_TYPE $DISABLE_TIME_ESTIMATION $UPDATE_VIDUR_CODE> /dev/null 2>&1 &
nohup sh vidur/prediction/exp/run_exp_global_scheduler.sh $TARGET_HOST $M $N $METRIC_TYPE $HOST_CONFIG_PATH> /dev/null 2>&1 &