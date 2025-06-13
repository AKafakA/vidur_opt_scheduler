# Plot main results
TTFT_SLO=3
#!/bin/bash
export PYTHONPATH=.
python3 experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/sharegpt \
    --output-dir experiment_output/results/main \
    --ttft-p99-slo $TTFT_SLO \
    --max-qps 36 \
    --min-qps 20 \
    --num-of-cdf-figures 5 \
    --zoomed True

python3 experiments_analysis/prediction_plot.py \
    --experiments-dir experiment_output/data/prediction/sharegpt \
    --output-dir experiment_output/results/prediction \

#python3 experiments_analysis/experiment_plot.py \
#    --experiments-dir experiment_output/config_search/sharegpt \
#    --output-dir experiment_output/results/main \
#    --ttft-p99-slo $TTFT_SLO \
#    --max-qps 36 \
#    --min-qps 20 \
#    --num-of-cdf-figures 5 \
#    --zoomed True

python3 experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/extension/burstgpt \
    --output-dir experiment_output/results/burstgpt \
    --ttft-p99-slo $TTFT_SLO \
    --max-qps 60 \
    --min-qps 55 \
    --num-of-cdf-figures 5 \
    --zoomed True

python3 experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/extension/sharegpt \
    --output-dir experiment_output/results/qwen \
    --ttft-p99-slo $TTFT_SLO \
    --max-qps 70 \
    --min-qps 50 \
    --num-of-cdf-figures 5 \
    --zoomed True

git checkout -b
