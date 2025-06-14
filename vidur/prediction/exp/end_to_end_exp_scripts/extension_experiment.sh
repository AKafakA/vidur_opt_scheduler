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
UPDATE_VLLM_CODE=true
RUN_EXP=true
RESTART_VLLM=true

ENABLE_CHUNKED_PREFILL="true"

MODEL="meta-llama/Llama-2-7b-hf Qwen/Qwen2-7B"
SCHEDULER_NAME="min_new_request_latency min_lunmnix_load"
#QPS="30 24 18"
PROFILING_SAMPLE_RATE=0.0
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
N_SELECTED="12"
OUTPUT_DIR_PREFIX="extension"


for model in $MODEL; do
  if [ "$model" = "meta-llama/Llama-2-7b-hf" ]; then
    MODEL_TYPE="llama"
    DATASET_NAMES="burstgpt"
  elif [ "$model" = "Qwen/Qwen2-7B" ]; then
    MODEL_TYPE="qwen"
    DATASET_NAMES="sharegpt"
    echo "Running warmup script for ${model} model to download the model weights and cache them"
    sh vidur/prediction/exp/end_to_end_exp_scripts/warmup.sh ${model} > /dev/null 2>&1
    echo "Warmup for ${model} model completed"
  fi
  for dataset_name in $DATASET_NAMES; do
    for scheduler in $SCHEDULER_NAME; do
      #  use for qwen-2-7b and sharegpt
      #  test block from 55 to 80 and block*/lunmnix_load from 71 to 80
      for enable_chunked_prefill in $ENABLE_CHUNKED_PREFILL; do
        for use_estimation_len in $USE_LENGTH_ESTIMATION; do
          if [ "$model" = "Qwen/Qwen2-7B" ]; then
            if [ "$scheduler" = "min_new_request_latency" ]; then
              if [ "$use_estimation_len" = "true" ]; then
                # test block* from 55 to 80
                QPS="71 72 73 74 75 76 77 78 79 80"
              else
                # test block from 55 to 80
                QPS="55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80"
              fi
            else
              # test lumnix_load from 71 to 80
              QPS="71 72 73 74 75 76 77 78 79 80"
            fi
          else
            # use for llama-2-7b and burstgpt
            # test block*/lunmnix_load from 48 to 64 excluding tested qps as block is unavailable without training data
            if [ "$scheduler" = "min_new_request_latency" ]; then
              if [ "$use_estimation_len" = "true" ]; then
                # block* is unavailable without training data
                QPS=""
              else
                # test block from 48 to 64
                QPS="48 49 50 51 52 53 54 55 61 62 63 64"
              fi
            else
              # test lumnix_load from 48 to 64
              QPS="48 49 50 51 52 53 54 55 61 62 63 64"
            fi
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
