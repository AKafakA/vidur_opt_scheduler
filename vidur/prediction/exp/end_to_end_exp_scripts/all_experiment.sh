START_INDEX=0
BATCH_CAP=48
TARGET_HOST='asdwb@d7525-10s10309.wisc.cloudlab.us'
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
CHUNK_SIZE=512
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="single_predictor_evaluation"
USE_PROCESS_FOR_FRONTEND=true
UPDATE_VIDUR_CODE=false
UPDATE_VLLM_CODE=false
RUN_EXP=true
RESTART_VLLM=true

# Config for end to end experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
SCHEDULER_NAME="min_new_request_latency min_lunmnix_load min_infass_load round_robin request_per_seconds random"
#QPS="20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
N_SELECTED="12"
OUTPUT_DIR_PREFIX="main"


for model in $MODEL; do
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen2-7B" ]; then
    MODEL_TYPE="qwen"
  fi
  for dataset_name in $DATASET_NAMES; do
    for scheduler in $SCHEDULER_NAME; do
      if [ "$scheduler" = "min_new_request_latency" ]; then
        USE_LENGTH_ESTIMATION="true false"
      else
        USE_LENGTH_ESTIMATION="false"
      fi
      for enable_chunked_prefill in $ENABLE_CHUNKED_PREFILL; do
        for use_estimation_len in $USE_LENGTH_ESTIMATION; do
          if [ "$use_estimation_len" = "true" ]; then
            # only block for min_new_request_latency scheduler use this flag
            QPS="31.1 31.2 31.3 31.4 31.5 31.6 31.7 31.8 31.9 32.0"
          elif [ "$scheduler" = "min_new_request_latency" ]; then
            QPS="32.1 32.2 32.3 32.4 32.5 32.6 32.7 32.8 32.9"
          elif [ "$scheduler" = "min_lunmnix_load" ]; then
            QPS="30.1 30.2 30.3 30.4 30.5 30.6 30.7 30.8 30.9"
          elif [ "$scheduler" = "round_robin" ]; then
            QPS="22.1 22.2 22.3 22.4 22.5 22.6 22.7 22.8 22.9"
          elif [ "$scheduler" = "min_infass_load" ]; then
            QPS="28.1 28.2 28.3 28.4 28.5 28.6 28.7 28.8 28.9"
          elif [ "$scheduler" = "request_per_seconds" ]; then
            QPS="21.1 21.2 21.3 21.4 21.5 21.6 21.7 21.8 21.9"
          elif [ "$scheduler" = "random" ]; then
            QPS="20.1 20.2 20.3 20.4 20.5 20.6 20.7 20.8 20.9"
          fi
          for batch_size_cut in $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION; do
            for n_selected in $N_SELECTED; do
              for qps in $QPS; do
                dataset_path="~/vidur_opt_scheduler/data/trace_data/$dataset_name/generate/$MODEL_TYPE"
                echo "Running experiment with scheduler: $scheduler, model: $model, dataset: $dataset_name, qps: $qps, batch_size_cut: $batch_size_cut enable_chunked_prefill: $enable_chunked_prefill use_for_profiling_only: $USE_FOR_PROFILING_ONLY predictor timeout: $PREDICTOR_TIMEOUT_IN_SECONDS"
                sh vidur/prediction/exp/experiment.sh $scheduler $NUM_REQUEST $RESTART_VLLM  $BATCH_CAP $dataset_name $dataset_path $dataset_name true $KEEP_ALL_METRICS $START_INDEX $model $MODEL_TYPE $MAX_MODEL_LENGTH $TARGET_HOST $enable_chunked_prefill $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $CHUNK_SIZE $qps $BRANCH_NAME $batch_size_cut $n_selected $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND $UPDATE_VIDUR_CODE $UPDATE_VLLM_CODE $RUN_EXP $use_estimation_len $OUTPUT_DIR_PREFIX
              done
            done
          done
        done
      done
    done
  done
done

#mkdir -p ~/vidur_opt_scheduler/single_node_experiment_output/
#scp -r $TARGET_HOST:~/vidur_opt_scheduler/experiment_output/* ~/vidur_opt_scheduler/single_node_experiment_output/.
