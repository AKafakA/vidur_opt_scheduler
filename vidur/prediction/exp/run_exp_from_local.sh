SCHEDULER_METRIC_TYPE=$1
MODEL="meta-llama/Llama-2-7b-hf"
HOST_CONFIG_PATH='vidur/prediction/config/host_configs.json'
PREDICTOR_CONFIG_PATH="vidur/prediction/config/test_config.json"
DISABLE_TIME_ESTIMATION=false

#DATASET_NAME = "sharegpt_gpt4"
DATASET_NAME="sharegpt-val-10k-predicted"
DATASET_PATH="~/$DATASET_NAME.json"
DATASET_TYPE="sharegpt"

GENERATE_NEW_DATA=false
DOWNLOAD_DATASET=$2
BATCH_CAP=$4
UPDATE_VIDUR_CODE=false
UPDATE_VLLM_CODE=false
RUN_EXP=false

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

if [ "$RUN_EXP" = "true" ]; then
  cd ~/vidur_opt_scheduler
  export PYTHONPATH=.
  QPS="36"
  NUM_QUERIES="10000"
  if [ "$SCHEDULER_METRIC_TYPE" = "min_latency" ]; then
    METRIC_TYPES="round_robin min_latency"
  else
    METRIC_TYPES=$SCHEDULER_METRIC_TYPE
  fi
  for qps in $QPS; do
      for num_queries in $NUM_QUERIES; do
        for metric_type in $METRIC_TYPES; do
         if [ "$metric_type" = "min_latency" ] || [ "$metric_type" = "min_gpu_blocks" ]
         then
              N="2"
          else
              N="12"
          fi
          for n in $N; do
                  echo "Running experiment with qps: $qps, num_queries: $num_queries, n: $n, metric_type: $metric_type"
                  pkill -f global_scheduler
                  nohup python vidur/prediction/global_scheduler/api_server.py --config_path $CONFIG_PATH --metrics_type $METRICS_TYPE --num_query_predictor $M --num_required_predictor $N > experiment_output/logs/global_scheduler.log 2>&1 &
                  LOG_FILENAME="benchmark.log"
                  OUTPUT_DIR="${metric_type}/qps_${qps}_num_queries_${num_queries}_n_${n}"
                  python vidur/prediction/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $num_queries --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename $LOG_FILENAME --output_dir $OUTPUT_DIR --tag_dataset_with_real_response $GENERATE_NEW_DATA --enable_csv_files $GENERATE_NEW_DATA
              done
          done
      done
  done

  #  test if using the estimated length
    if [ "$SCHEDULER_METRIC_TYPE" = "min_latency" ] || [ "$SCHEDULER_METRIC_TYPE" = "min_gpu_blocks" ]; then
      QPS="10 12 16 20 24 36"
      N="2"
      for qps in $QPS; do
        for num_queries in $NUM_QUERIES; do
          for n in $N; do
            for metric_type in $SCHEDULER_METRIC_TYPE; do
                pkill -f global_scheduler
                echo "Running experiment with qps: $qps, num_queries: $num_queries, n: $n, metric_type: $metric_type with estimated length"
                nohup python vidur/prediction/global_scheduler/api_server.py --config_path $CONFIG_PATH --metrics_type $METRICS_TYPE --num_query_predictor $M --num_required_predictor $N > experiment_output/logs/global_scheduler.log 2>&1 &
                LOG_FILENAME="benchmark.log"
                OUTPUT_DIR="${metric_type}*/qps_${qps}_num_queries_${num_queries}_n_${n}_estimated_length"
                python vidur/prediction/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $num_queries --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename $LOG_FILENAME --output_dir $OUTPUT_DIR --tag_dataset_with_real_response $GENERATE_NEW_DATA --enable_csv_files $GENERATE_NEW_DATA --use_estimated_response_lens true
                sleep 60
              done
          done
        done
      done
    fi
fi



