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

from experiments_analysis.experiment_plot import plot_linear_for_multiple_qps, directory_name_parser

experiment_name_replacement = {"min latency": "Block"}
scheduler_name_ordered = ['Block']


def directory_name_parser_for_auto_provision(directory_name):
    directory_name = directory_name.split("_")
    qps = directory_name[1]
    n = directory_name[6]
    enable_preemptive_provision = directory_name[24] == "true"
    waiting_time_slo = int(directory_name[19])
    enable_auto_scaling = waiting_time_slo > 0
    return qps, n, enable_preemptive_provision, enable_auto_scaling, waiting_time_slo


def plot_dual_timeline_data(experiments_set, qps):
    """
    Plot two sets of data on the same timeline with different y-axes.
    """
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_axes([0.1,0.1, 0.8,0.8])
    ax2 = fig.add_axes([0.1,0.1, 0.8,0.2], facecolor=(0, 0, 0, 0))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_coords(1.05, 0.0)
    ax2.yaxis.set_label_position("right")
    ax1.set_xlim([0, len(experiments_set[0]['request_latencies'])])
    ax2.set_xlim([0, len(experiments_set[0]['request_latencies'])])
    ax2.spines['top'].set_visible(False)

    for exp in experiments_set:
        enable_preemptive_provision = exp['enable_preemptive_provision']
        enable_auto_scaling = exp['enable_auto_scaling']
        if enable_preemptive_provision:
            label1 = "Preemptive Provisioning"
            color1 = "blue"
        elif enable_auto_scaling:
            label1 = "relief scaling"
            color1 = "orange"
        else:
            label1 = "Full Provisioned"
            color1 = "red"
        request_latencies = exp['request_latencies']
        available_instances = exp['available_instances']
        x = np.arange(len(request_latencies))
        request_latencies = gaussian_filter1d(request_latencies, sigma=20)
        ax1.plot(x, request_latencies, label=label1, color=color1, linewidth=2)
        ax1.set_xlabel("Query ID")
        ax1.set_ylabel("Request Latencies (s)")
        ax2.plot(x, available_instances, label=label1, color=color1, ls='--', linewidth=2)
        ax2.set_ylabel("Available Instances")
    fig.tight_layout()
    ax1.legend(loc='best')
    return fig


def plot_per_qps(experiments_set, output_dir, selected_qps):
    selected_experiments = [experiment for experiment in experiments_set if experiment['qps'] == selected_qps]
    if not selected_experiments:
        print(f"No experiments found for QPS: {selected_qps}")
        return
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    fig = plot_dual_timeline_data(selected_experiments, selected_qps)
    fig.savefig(os.path.join(output_dir, f"qps_{selected_qps}_dual_timeline.png"))


def main():
    parser = argparse.ArgumentParser(description='Plot the results of the experiments')
    parser.add_argument("--experiments-dir", type=str,
                        default="experiments_analysis/auto_provision_experiment_output/sharegpt")
    parser.add_argument("--output-dir", type=str, default="./experiments_analysis/auto_provision_plots")
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
            qps, n, enable_preemptive_provision, enable_auto_scaling, waiting_time_slo \
                = directory_name_parser_for_auto_provision(directory)
            record["qps"] = float(qps)
            record["n"] = int(n)
            record["enable_preemptive_provision"] = enable_preemptive_provision
            record["enable_auto_scaling"] = enable_auto_scaling
            record["waiting_time_slo"] = waiting_time_slo
            for experiments_trace in os.listdir(scheduler_dir + "/" + directory):
                if experiments_trace.endswith("npz"):
                    b = np.load(scheduler_dir + "/" + directory + "/" + experiments_trace)
                    record['request_latencies'] = b['request_latencies'] / 1000.0  # Convert to seconds
                    record['available_instances'] = b['available_instances']
                    record['waiting_latencies'] = b['waiting_latencies'] / 1000.0  # Convert to seconds

    for qps in set([experiment['qps'] for experiment in experiments_set]):
        plot_per_qps(experiments_set, args.output_dir, qps)

    # if args.plot_per_scheduler:
    #     plot_per_scheduler(experiments_set, args.output_dir)


if __name__ == "__main__":
    main()
