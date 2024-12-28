#!/bin/bash

METRIC_TYPE="min_latency"
BATCH_CAP=48
MODEL="meta-llama/Llama-2-7b-hf"
HOST_CONFIG_PATH='vidur/prediction/config/host_configs.json'
PREDICTOR_CONFIG_PATH="vidur/prediction/config/test_config.json"
DISABLE_TIME_ESTIMATION=false

TARGET_HOST='asdwb@d7525-10s10323.wisc.cloudlab.us'

#DATASET_NAME = "sharegpt_gpt4"
DATASET_NAME="sharegpt_V3_format"
DATASET_PATH="~/$DATASET_NAME.jsonl"
DATASET_TYPE="sharegpt"

GENERATE_NEW_DATA=true
DOWNLOAD_DATASET=true
UPDATE_VIDUR_CODE=false
UPDATE_VLLM_CODE=false
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
  parallel-ssh --host $TARGET_HOST "cd vidur_opt_scheduler && rm experiment_output/logs/*"
  sh vidur/prediction/exp/reset.sh
  nohup sh vidur/prediction/exp/run_exp_vllm.sh $BATCH_CAP $MODEL $UPDATE_VLLM_CODE > /dev/null 2>&1 &
  nohup sh vidur/prediction/exp/run_exp_predictor.sh $PREDICTOR_CONFIG_PATH $METRIC_TYPE $DISABLE_TIME_ESTIMATION $UPDATE_VIDUR_CODE $BATCH_CAP> /dev/null 2>&1 &
  sleep 60
fi

if [ "$RUN_EXP" = "true" ]; then
  QPS="2.0"
  NUM_QUERIES="58"
  METRIC_TYPES="random"
  if [ "$DOWNLOAD_DATASET" = "true" ]; then
    parallel-ssh -t 0 --host $TARGET_HOST "wget https://huggingface.co/datasets/shibing624/sharegpt_gpt4/resolve/main/$DATASET_NAME.jsonl"
  fi
  for qps in $QPS; do
      for num_queries in $NUM_QUERIES; do
        for metric_type in $METRIC_TYPES; do
         if [ "$metric_type" = "min_latency" ]; then
              N="6 8 10 12 24 36"
          else
              N="2"
          fi
          for n in $N; do
                  echo "Running experiment with qps: $qps, num_queries: $num_queries, n: $n, metric_type: $metric_type"
                  nohup sh vidur/prediction/exp/run_exp_global_scheduler.sh $TARGET_HOST $n $n $metric_type $HOST_CONFIG_PATH > /dev/null 2>&1 &
                  LOG_FILENAME="benchmark.log"
                  OUTPUT_DIR="${metric_type}/qps_${qps}_num_queries_${num_queries}_n_${n}"
                  parallel-ssh -i -t 0 --host $TARGET_HOST "cd vidur_opt_scheduler && export PYTHONPATH=. && python vidur/prediction/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $num_queries --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename $LOG_FILENAME --output_dir $OUTPUT_DIR --tag_dataset_with_real_response $GENERATE_NEW_DATA --enable_csv_files $GENERATE_NEW_DATA"
                  sleep 10
              done
          done
      done
  done
fi

