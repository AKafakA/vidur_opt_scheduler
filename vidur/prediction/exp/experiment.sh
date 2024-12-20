#!/bin/bash

METRIC_TYPE="min_latency"
BATCH_CAP=48
MODEL="meta-llama/Llama-2-7b-hf"

TARGET_HOST='asdwb@d7525-10s10327.wisc.cloudlab.us'
M=12
N=12
HOST_CONFIG_PATH='vidur/prediction/config/host_configs.json'
PREDICTOR_CONFIG_PATH="vidur/prediction/config/test_config.json"
DISABLE_TIME_ESTIMATION=false
UPDATE_VIDUR_CODE=true
UPDATE_VLLM_CODE=false
DATASET_PATH="~/sharegpt_gpt4_with_real_response.jsonl"
DATASET_TYPE="sharegpt"

RESTART_VLLM=false
RUN_EXP=true

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

if [ "$RESTART_VLLM" = "true" ]; then
  sh vidur/prediction/exp/reset.sh
  nohup sh vidur/prediction/exp/run_exp_vllm.sh $BATCH_CAP $MODEL $UPDATE_VLLM_CODE > /dev/null 2>&1 &
  nohup sh vidur/prediction/exp/run_exp_predictor.sh $PREDICTOR_CONFIG_PATH $METRIC_TYPE $DISABLE_TIME_ESTIMATION $UPDATE_VIDUR_CODE $BATCH_CAP> /dev/null 2>&1 &
fi

if [ "$RUN_EXP" = "true" ]; then
  QPS="6.0 8.0 10.0 12.0 24.0 36.0"
  NUM_QUERIES="2000 6000"
  METRIC_TYPES="min_latency random round_robin"
  for qps in $QPS; do
      for num_queries in $NUM_QUERIES; do
        for metric_type in $METRIC_TYPES; do
         if [ "$metric_type" = "min_latency" ]; then
              N="6 8 10 12 24 36"
          else
              N="6"
          fi
          for n in $N; do
                  echo "Running experiment with qps: $qps, num_queries: $num_queries, n: $n, metric_type: $metric_type"
                  nohup sh vidur/prediction/exp/run_exp_global_scheduler.sh $TARGET_HOST $n $n $metric_type $HOST_CONFIG_PATH > /dev/null 2>&1 &
                  LOG_FILENAME="${qps}_${num_queries}_${n}"
                  LOG_FILES_DIR="experiment_output/$metric_type/logs/"
                  parallel-ssh --host $TARGET_HOST "mkdir -p $LOG_FILES_DIR"
                  parallel-ssh --host $TARGET_HOST "cd vidur_opt_scheduler && export PYTHONPATH=. && python vidur/prediction/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $num_queries --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename $LOG_FILENAME --output_dir $metric_type --tag_dataset_with_real_response false --enable_csv_files false > ${LOG_FILES_DIR}${LOG_FILENAME}.log"
              done
          done
      done
  done
fi

