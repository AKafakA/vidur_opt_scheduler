import argparse
import os
import re
from operator import index

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.pyplot as plt
import numpy as np
import shutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from experiments_analysis.experiment_plot import plot_linear_for_multiple_qps

experiment_name_replacement = {"min latency": "Block", "min infass load": "INFaaS++",
                               "request per seconds": "Instance-QPM"}
scheduler_name_ordered = ['Round Robin', 'random', 'INFaaS++', 'Instance-QPM', 'Block*', 'Block']


def directory_name_parser(directory_name):
    directory_name = directory_name.split("_")
    qps = directory_name[1]
    n = directory_name[6]
    chunked = directory_name[8]
    return qps, n, chunked


def extract_prediction_errors(experiment):
    average_prediction_errors_ratio = experiment['prediction_errors']
    compare_error_rate = experiment['compare_error_rate']
    return average_prediction_errors_ratio, compare_error_rate


def plot_line_scatter_mixed(ax, scatter_y, linea_y):
    """
    Plot a line and scatter mixed plot on the given axes.
    :param ax: The axes to plot on.
    :param scatter_y: The y-values for the scatter plot.
    :param linea_y: The y-values for the line plot.
    """
    assert len(scatter_y) == len(linea_y), "Scatter and line data must have the same length"
    # scatter is a 2 dimensional np array, x is the index of first column
    # ax.scatter(x, scatter_y, color='blue', label='Scatter Data', s=10)
    # ax.plot(x, linea_y, color='red', label='Line Data')
    x = np.arange(len(scatter_y))
    ax.scatter(x, scatter_y, color='blue', label='Scatter Data', s=10)
    ax.plot(x, np.mean(scatter_y), color='red', label='Line Data', s=10)
    ax.plot(x, linea_y, color='green', label='Line Data', s=10)
    ax.plot(x, np.min(scatter_y), color='black', label='Min Data', s=10)
    ax.plot(x, np.max(scatter_y), color='orange', label='Max Data', s=10)


def plot_linea_scatter_for_multiple_qps(axes, linear_data, scatter_data,
                                        title_fontsize=10):
    """
    Plot multiple line and scatter mixed plots for different QPS values.
    """
    assert len(scatter_data) == len(linear_data), "Scatter and line data must have the same length"
    for qps in linear_data.keys():
        ax = axes.get(qps)
        qps_linear_data = linear_data[qps]
        qps_scatter_data = scatter_data[qps]
        plot_line_scatter_mixed(ax, qps_scatter_data, qps_linear_data)
        ax.set_title(f"{qps} QPS", fontsize=title_fontsize)


def plot_per_qps(experiments_set, output_dir, min_qps=24, max_qps=32):
    qps_output_dir = output_dir + "/qps"
    if os.path.exists(qps_output_dir):
        shutil.rmtree(qps_output_dir)
    os.makedirs(qps_output_dir)

    # avg_free_gpu = {}
    # var_free_gpu_per_node = {}
    # num_total_preemption = {}

    qps_set = sorted(set([record["qps"] for record in experiments_set]))
    if min_qps > 0:
        qps_set = [qps for qps in qps_set if min_qps <= qps <= max_qps]

    sorted_keys = []
    map_from_name_exp = {}
    prediction_overhead = {}
    prediction_overhead_ratio = {}
    prediction_errors = {}
    compare_errors = {}
    prediction_errors_rate = {}
    sampled_predict_latency = {}
    sampled_serving_latencies = {}
    for qps in qps_set:
        prediction_overhead_per_qps = {}
        prediction_overhead_ratio_per_qps = {}
        prediction_errors_per_qps = {}
        prediction_errors_rate_per_qps = {}
        compare_errors_per_qps = {}
        sampled_predict_latency_per_qps = {}
        sampled_serving_latencies_per_qps = {}
        prediction_overhead[qps] = prediction_overhead_per_qps
        prediction_overhead_ratio[qps] = prediction_overhead_ratio_per_qps
        prediction_errors[qps] = prediction_errors_per_qps
        prediction_errors_rate[qps] = prediction_errors_rate_per_qps
        compare_errors[qps] = compare_errors_per_qps
        sampled_predict_latency[qps] = sampled_predict_latency_per_qps
        sampled_serving_latencies[qps] = sampled_serving_latencies_per_qps

        qps_experiments = [record for record in experiments_set if record["qps"] == qps]
        for experiment in qps_experiments:
            if experiment['chunked'] == 'true':
                experiment_name = "Chunked Prefilled"
            else:
                experiment_name = "Prefilled Prioritized"
            for key in experiment_name_replacement.keys():
                if key in experiment_name:
                    experiment_name = experiment_name.replace(key, experiment_name_replacement[key])
            map_from_name_exp[experiment_name] = experiment
        ordered_key = []
        if len(sorted_keys) == 0:
            sorted_keys = sorted(map_from_name_exp.keys())
            for key in scheduler_name_ordered:
                if key in sorted_keys:
                    sorted_keys.remove(key)
                    ordered_key.append(key)
            sorted_keys = sorted_keys + ordered_key
        for index_name in sorted_keys:
            experiments = map_from_name_exp[index_name]
            if index_name not in prediction_overhead:
                current_prediction_overhead = experiments['prediction_overhead']
                prediction_overhead_per_qps[index_name] = current_prediction_overhead
                end_to_end_latencies = experiments['request_latencies']
                prediction_overhead_ratio_per_qps[index_name] = [(100.0 * overhead / latency)
                                                                 for overhead, latency
                                                                 in
                                                                 zip(current_prediction_overhead, end_to_end_latencies)
                                                                 if latency > 0]
                prediction_errors_per_qps[index_name] = experiments['prediction_errors']
                compare_errors_per_qps[index_name] = experiments['compare_error_rate']
                if index_name == "Chunked Prefilled":
                    sampled_predict_latency_per_qps[index_name] = experiments['min_predicted_latency']
                    sampled_serving_latencies_per_qps[index_name] = experiments['serving_latencies']

    fig, axs = plt.subplots(2, len(qps_set))
    axs_for_prediction_overhead_ratio = {}
    axs_for_prediction_errors_rate = {}

    for i, qps_value in enumerate(qps_set):
        # axs_for_prediction_overhead[qps_value] = axs[0][i]
        axs_for_prediction_errors_rate[qps_value] = axs[0][i]
        axs_for_prediction_overhead_ratio[qps_value] = axs[1][i]

    # plot_linear_for_multiple_qps(axs_for_prediction_overhead, prediction_overhead, "Overhead (ms)",
    #                              sigma=10,
    #                              enable_legend_at_middle=True, legend_anchor=(1.1, 1.25),
    #                              title_fontsize=10)
    plot_linear_for_multiple_qps(axs_for_prediction_errors_rate, prediction_errors,
                                 "Error Rate (%)", sigma=1,
                                 enable_legend_at_middle=False,
                                 title_fontsize=10)

    plot_linea_scatter_for_multiple_qps(axs_for_prediction_overhead_ratio,
                                        sampled_predict_latency,
                                        sampled_serving_latencies)

    fig.tight_layout()
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(hspace=0.2, wspace=0.35)
    plt.savefig(qps_output_dir + "/all_qps.png", bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='Plot the results of the experiments')
    parser.add_argument("--experiments-dir", type=str,
                        default="experiments_analysis/single_node_experiment_output/sharegpt")
    parser.add_argument("--output-dir", type=str, default="./experiments_analysis/single_node_exp_plots")
    parser.add_argument("--plot-per-qps", type=bool, default=True)
    # parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    data_dir = os.getcwd() + "/" + args.experiments_dir

    experiments_set = []
    for scheduler_name in os.listdir(data_dir):
        scheduler_dir = data_dir + "/" + scheduler_name
        if scheduler_name == 'logs':
            continue
        for directory in os.listdir(scheduler_dir):
            record = {"scheduler_name": "block"}
            experiments_set.append(record)
            qps, n, chunked = directory_name_parser(directory)
            record["qps"] = float(qps)
            record["n"] = int(n)
            record["chunked"] = chunked
            for experiments_trace in os.listdir(scheduler_dir + "/" + directory):
                if experiments_trace.endswith("npz"):
                    b = np.load(scheduler_dir + "/" + directory + "/" + experiments_trace)
                    record['prediction_overhead'] = b['scheduling_overhead']
                    record['request_latencies'] = b['request_latencies']
                    record['prediction_errors'] = b['sampled_mean_error_ratios']
                    record['compare_error_rate'] = b['sampled_predict_accuracies']
                    record['serving_latencies'] = b['sampled_serving_latencies']
                    record['min_predicted_latency'] = b['sampled_predict_latency']

    plot_per_qps(experiments_set, args.output_dir)

    # if args.plot_per_scheduler:
    #     plot_per_scheduler(experiments_set, args.output_dir)


if __name__ == "__main__":
    main()
