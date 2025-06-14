START_INDEX=0
TARGET_HOST='asdwb@d7525-10s10309.wisc.cloudlab.us'
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="single_predictor_evaluation"
USE_PROCESS_FOR_FRONTEND=true
UPDATE_VIDUR_CODE=true
UPDATE_VLLM_CODE=false
RUN_EXP=true
RESTART_VLLM=true

# Config for end to end experiment
ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_NAMES="sharegpt"
SCHEDULER_NAME="min_new_request_latency"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
N_SELECTED="12"
OUTPUT_DIR_PREFIX="config_search"
BATCH_CAP="48 24"

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
          for batch_size_cut in $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION; do
            for n_selected in $N_SELECTED; do
              for batch_cap in $BATCH_CAP; do
                if [ "$batch_cap" = "24" ]; then
                  OUTPUT_DIR_PREFIX="config_search/batch"
                  CHUNK_SIZE="512"
                  # Test for block scheduler from 20 to 36
                  if [ "$scheduler" = "min_new_request_latency" ]; then
                    # Test for block scheduler from 20 to 36
                    if [ "$use_estimation_len" = "false" ]; then
                      QPS="20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36"
                    else
                      # Test for block* scheduler
                      QPS=""
                    fi
                  else
                    # Test for lumnix scheduler
                    QPS=""
                  fi
                else
                  CHUNK_SIZE="2048"
                  OUTPUT_DIR_PREFIX="config_search/chunkSize"
                  if [ "$scheduler" = "min_new_request_latency" ]; then
                    if [ "$use_estimation_len" = "false" ]; then
                      # Test for block scheduler from 20 to 36
                      QPS="20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36"
                    else
                      # Test for block* scheduler
                      QPS="30.00 30.10 30.20 30.30 30.4 30.5 30.6 30.7 30.8 30.9"
                    fi
                  else
                    QPS=""
                  fi
                fi
                for chunk in $CHUNK_SIZE; do
                  for qps in $QPS; do
                    dataset_path="~/vidur_opt_scheduler/data/trace_data/$dataset_name/generate/$MODEL_TYPE"
                    echo "Running experiment with scheduler: $scheduler, model: $model, dataset: $dataset_name, qps: $qps, enable_chunked_prefill: $enable_chunked_prefill batch_size: $batch_cap, chunk_size $chunk"
                    sh vidur/prediction/exp/experiment.sh $scheduler $NUM_REQUEST $RESTART_VLLM  $batch_cap $dataset_name $dataset_path $dataset_name true $KEEP_ALL_METRICS $START_INDEX $model $MODEL_TYPE $MAX_MODEL_LENGTH $TARGET_HOST $enable_chunked_prefill $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $chunk $qps $BRANCH_NAME $batch_size_cut $n_selected $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND $UPDATE_VIDUR_CODE $UPDATE_VLLM_CODE $RUN_EXP $use_estimation_len $OUTPUT_DIR_PREFIX
                  done
                done
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
