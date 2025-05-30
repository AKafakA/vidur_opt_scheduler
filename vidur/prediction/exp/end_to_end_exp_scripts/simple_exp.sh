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
UPDATE_VIDUR_CODE=true
UPDATE_VLLM_CODE=false
RUN_EXP=true
RESTART_VLLM=true

ENABLE_CHUNKED_PREFILL="true"

MODEL="meta-llama/Llama-2-7b-hf"
SCHEDULER_NAME="min_new_request_latency"
#QPS="20 24 28 32 36"
QPS="20"
PROFILING_SAMPLE_RATE=0.00
USE_FOR_PROFILING_ONLY=False
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
N_SELECTED="12"
OUTPUT_DIR_PREFIX="simple"
DATASET_NAMES="sharegpt"


for model in $MODEL; do
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
  elif [ "$model" = "Qwen/Qwen-7B" ]; then
    MODEL_TYPE="qwen"
  fi
  for dataset_name in $DATASET_NAMES; do
    for scheduler in $SCHEDULER_NAME; do
      if [ "$scheduler" = "min_new_request_latency" ]; then
        USE_LENGTH_ESTIMATION="true"
      else
        USE_LENGTH_ESTIMATION="false"
      fi
      for enable_chunked_prefill in $ENABLE_CHUNKED_PREFILL; do
        if [ "$enable_chunked_prefill" = "true" ]; then
          USE_LENGTH_ESTIMATION="false"
        else
          USE_LENGTH_ESTIMATION="false"
        fi
        for use_estimation_len in $USE_LENGTH_ESTIMATION; do
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
