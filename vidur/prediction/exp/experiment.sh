SCHEDULER_METRIC_TYPE=$1
ENABLE_TIME_ESTIMATION=true

NUM_DATA=$2
RESTART_VLLM=$3
BATCH_CAP=$4


DATASET_NAME=$5
DATASET_PATH=$6
DATASET_TYPE=$7
GENERATE_NEW_DATA=$8
KEEP_ALL_METRICS=$9
START_INDEX=${10}
MODEL=${11}
MODEL_TYPE=${12}
MAX_MODEL_LENGTH=${13}
TARGET_HOST=${14}
HOST_CONFIG_PATH='vidur/prediction/config/host_configs.json'
PREDICTOR_CONFIG_PATH="vidur/prediction/config/${MODEL_TYPE}_config.json"
ENABLE_CHUNKED_PREFILL=${15}

PREDICTOR_WORKERS=${16}
GLOBAL_SCHEDULER_WORKERS=${17}
BACKEND_WORKERS=${18}
CHUNK_SIZE=${19}
QPS=${20}
BRANCH_NAME=${21}
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION=${22}
N_SELECTED=${23}
PROFILING_SAMPLE_RATE=${24}
TIMEOUT_IN_SECONDS=${25}
USE_FOR_PROFILING_ONLY=${26}
PREDICTOR_TIMEOUT_IN_SECONDS=${27}
USE_PROCESS_FOR_FRONTEND=${28}

UPDATE_VIDUR_CODE=${29}
UPDATE_VLLM_CODE=${30}
RUN_EXP=${31}

if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
  MAX_NUM_BATCHED_TOKEN=$CHUNK_SIZE
else
  MAX_NUM_BATCHED_TOKEN=$MAX_MODEL_LENGTH
fi

# Current the v1 version of vllm is supported yet
VLLM_VERSION=0
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
  sleep 10
  nohup sh vidur/prediction/exp/run_exp_vllm.sh $BATCH_CAP $MODEL $UPDATE_VLLM_CODE $VLLM_VERSION $MAX_MODEL_LENGTH $ENABLE_CHUNKED_PREFILL $BACKEND_WORKERS $MAX_NUM_BATCHED_TOKEN > /dev/null 2>&1 &
  if [ "$UPDATE_VIDUR_CODE" = "true" ]; then
    parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && git checkout $BRANCH_NAME && git pull"
    parallel-ssh -t 0 -h vidur/prediction/config/hosts "cd vidur_opt_scheduler && git reset --hard HEAD~20 && git pull"
  fi
  script_base="vidur/prediction/exp/run_exp_predictor"
  suffix_range=$(seq 1 7)
  for suffix in $suffix_range; do
    nohup sh "${script_base}_${suffix}.sh" $PREDICTOR_CONFIG_PATH $SCHEDULER_METRIC_TYPE $ENABLE_TIME_ESTIMATION $BATCH_CAP $ENABLE_CHUNKED_PREFILL $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
  done
  sleep 10
  suffix_range=$(seq 8 $PREDICTOR_WORKERS)
  for suffix in $suffix_range; do
    nohup sh "${script_base}_${suffix}.sh" $PREDICTOR_CONFIG_PATH $SCHEDULER_METRIC_TYPE $ENABLE_TIME_ESTIMATION $BATCH_CAP $ENABLE_CHUNKED_PREFILL $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
  done
  sleep 60
fi

if [ "$RUN_EXP" = "true" ]; then
  NUM_QUERIES=$NUM_DATA
  # Still use random for global scheduler but use min_latency for predictor
  METRIC_TYPES=$SCHEDULER_METRIC_TYPE
  if [ "$USE_FOR_PROFILING_ONLY" = "true" ]; then
    METRIC_TYPES="random"
  else
    METRIC_TYPES=$SCHEDULER_METRIC_TYPE
  fi
  for qps in $QPS; do
      for num_queries in $NUM_QUERIES; do
        for metric_type in $METRIC_TYPES; do
          if [ "$metric_type" = "min_new_request_latency" ]; then
            N=$N_SELECTED
            USE_ESTIMATION_LEN="true false"
          else
            N="12"
            USE_ESTIMATION_LEN="false"
          fi
          for n in $N; do
              for use_estimation_len in $USE_ESTIMATION_LEN; do
                  nohup sh vidur/prediction/exp/run_exp_global_scheduler.sh $TARGET_HOST $n $n $metric_type $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
                  LOG_FILENAME="benchmark.log"
                  OUTPUT_DIR="${DATASET_TYPE}/${metric_type}/qps_${qps}_num_queries_${num_queries}_n_${n}_chunked_${ENABLE_CHUNKED_PREFILL}_predictor_${PREDICTOR_WORKERS}_global_${GLOBAL_SCHEDULER_WORKERS}_len_estimated_${use_estimation_len}_estimation_cut_${BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION}"
                  sleep 10
                  if [ "$use_estimation_len" = "true" ]; then
                    parallel-ssh -i -t 0 --host $TARGET_HOST "cd vidur_opt_scheduler && export PYTHONPATH=. && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && python vidur/prediction/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $num_queries --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename $LOG_FILENAME --output_dir $OUTPUT_DIR --tag_dataset_with_real_response $GENERATE_NEW_DATA --enable_csv_files false --keep_all_metrics $KEEP_ALL_METRICS --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens"
                  else
                    parallel-ssh -i -t 0 --host $TARGET_HOST "cd vidur_opt_scheduler && export PYTHONPATH=. && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && python vidur/prediction/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $num_queries --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename $LOG_FILENAME --output_dir $OUTPUT_DIR --tag_dataset_with_real_response $GENERATE_NEW_DATA --enable_csv_files false --keep_all_metrics $KEEP_ALL_METRICS --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS"
                  fi

                  sleep 10
                  parallel-ssh --host $TARGET_HOST "cd vidur_opt_scheduler && mkdir experiment_output/$OUTPUT_DIR/running_logs"
                  parallel-ssh --host $TARGET_HOST "cd vidur_opt_scheduler && mv experiment_output/logs/* experiment_output/$OUTPUT_DIR/running_logs/."
                  done
              done
          done
      done
  done
fi



