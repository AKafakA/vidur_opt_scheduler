import argparse
import math
import os
import random
import re
from itertools import islice, cycle
from operator import index

from shapely.geometry import LineString

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.pyplot as plt
import numpy as np
import shutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d

experiment_name_replacement = {"min new request latency": "Block", "min infass load": "INFaaS++",
                               "request per seconds": "Min QPM",
                               "min lunmnix load": "Lumnix-"}
scheduler_name_ordered = ['random', 'Round Robin', 'INFaaS++', 'Min QPM', 'Block*', 'Block', "Lumnix-"]
scheduler_to_color = {
    'random': 'olive',
    'round robin': 'green',
    'INFaaS++': 'orange',
    'Min QPM': 'purple',
    'Block*': 'black',
    'Block': 'red',
    'Lumnix-': 'skyblue',
}

ttft_slo = 10  # default value for ttft p99 slo


def directory_name_parser(directory_name):
    directory_name = directory_name.split("_")
    qps = float(directory_name[1])
    n = directory_name[6]
    use_length_estimation = True if directory_name[15] == "true" else False
    return qps, n, use_length_estimation


def extract_data_from_log_file(log_file):
    pattern = r"""
        dur_s\s(?P<dur_s>\d+\.\d+)\s tokens_per_s\s(?P<tokens_per_s>\d+\.\d+)\s qps\s(?P<qps>\d+\.\d+)\s
    """
    with open(log_file, "r") as f:
        log = f.read()
        match = re.search(pattern, log, re.VERBOSE)
        return match.groupdict()


def plot_linear_for_multiple_qps(axes, data, metric_name, sigma=-1,
                                 enable_legend_at_middle=False,
                                 x_label="Query ID",
                                 legend_anchor=(2.0, 1.315),
                                 title_fontsize=13,
                                 enable_y_labels=True,
                                 enable_x_label_at_middle=False,
                                 enable_x_label_at_left_corner=False,
                                 x_label_coords=(-0.285, -0.105),
                                 enable_title_labels=False,
                                 max_num_samples=-1):
    i = 0
    enable_label = True
    for qps in data.keys():
        if qps not in axes.keys():
            continue
        ax = axes.get(qps)
        qps_data = data[qps]
        for key, value in qps_data.items():
            # smooth by guassian 1d
            if 0 < max_num_samples < len(value):
                value = value[:max_num_samples]
            if sigma > 0:
                value = gaussian_filter1d(value, sigma)
            if key in scheduler_to_color:
                ax.plot(value, label=key, color=scheduler_to_color[key])
            else:
                ax.plot(value, label=key)

        if enable_x_label_at_middle and i == len(axes.keys()) // 2:
            assert enable_x_label_at_left_corner is False, "Cannot enable both x_label at middle and left corner"
            ax.set_xlabel(x_label, fontsize=title_fontsize, loc='center')
        elif enable_x_label_at_left_corner and i == 0:
            ax.set_xlabel(x_label, fontsize=title_fontsize, loc='left')
            ax.xaxis.set_label_coords(*x_label_coords)
        else:
            ax.set_xlabel("")

        if enable_label and enable_y_labels:
            ax.set_ylabel(f"{metric_name}", fontsize=title_fontsize)
            enable_label = False

        if enable_title_labels:
            ax.set_title(f"QPS={qps}", fontsize=title_fontsize * 1.1)

        if enable_legend_at_middle and i == len(axes.keys()) // 2:
            ax.legend(fancybox=False, shadow=False, ncol=len(qps_data.keys()), fontsize=title_fontsize,
                      loc='upper right', bbox_to_anchor=legend_anchor)
        i += 1


def plot_bar_chart(ax, dataframe, metric_name, column_name, show_value_on_top=True):
    print(dataframe)
    dataframe = dataframe.sort_values(by=column_name)
    for scheduler, row in dataframe.iterrows():
        capacity = row['capacity']
        ax.bar(scheduler, capacity, label=scheduler, color=scheduler_to_color[scheduler])

    ax.set_ylabel(metric_name)
    x_ticks = ax.get_xticks()
    ax.set_ylim(min(dataframe[column_name] * 0.9), max(dataframe[column_name]) * 1.05)
    ax.set_xticks(x_ticks)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_size(8)

    if show_value_on_top:
        rects = ax.patches
        for rect in rects:
            height = rect.get_height()
            ax.text(
                (rect.get_x() + rect.get_width() / 2) - 0.4, height + 0.2, height
            )


def plot_linear_with_different_qps(ax,
                                   data_map,
                                   metric_name,
                                   keep_legend=False,
                                   keep_x_ticks_labels=False,
                                   x_dim="QPS",
                                   anchor=(2.0, 1.215),
                                   slo=-1,
                                   legend_fontsize=12):
    scheduler_list = data_map.keys()
    xdim_per_scheduler = [data_map[scheduler].keys() for scheduler in scheduler_list]
    flattened_x_dims = [float(item) for sublist in xdim_per_scheduler for item in sublist]
    capacity_indexes = []
    ax.set_xlim(min(flattened_x_dims), max(flattened_x_dims))
    min_qps = min(flattened_x_dims)
    max_qps = max(flattened_x_dims)
    for i, scheduler in enumerate(scheduler_list):
        y_values = list(data_map[scheduler].values())
        x_values = [float(qps) for qps in data_map[scheduler].keys()]
        ax.plot(x_values, y_values, label=scheduler, color=scheduler_to_color[scheduler])
        if slo > 0:
            first_line = LineString(np.column_stack((np.array(x_values), np.array(y_values, dtype=float))))
            second_line = LineString(np.column_stack((np.array(x_values),
                                                      np.array([slo] * len(y_values), dtype=float))))
            intersection = first_line.intersection(second_line)
            if intersection.geom_type == 'Point':
                capacity_indexes.append((scheduler, intersection.x))
            elif intersection.geom_type == 'MultiPoint':
                # print(intersection.geoms[-1].x)
                capacity_indexes.append((scheduler, intersection.geoms[-1].x))

    if slo > 0 and len(capacity_indexes) > 0:
        axins = inset_axes(ax, width="60%", height="30%", loc='upper left')
        selected_schedulers = ["Lumnix-", "Block", "Block*"]
        max_intersection = max([capacity_x for scheduler, capacity_x in capacity_indexes
                                if scheduler in selected_schedulers], default=0)
        min_intersection = min([capacity_x for scheduler, capacity_x in capacity_indexes
                                if scheduler in selected_schedulers], default=0)
        for i, scheduler in enumerate(selected_schedulers):
            y_values = list(data_map[scheduler].values())
            x_values = [float(qps) for qps in data_map[scheduler].keys()]
            axins.plot(x_values, y_values, label=scheduler, color=scheduler_to_color[scheduler])

        axins.set_xlim(math.floor(min_intersection) - 0.1,
                       math.ceil(max_intersection) + 0.1)  # Adjust limits as needed
        axins.set_ylim(ttft_slo - 2, ttft_slo + 2)  # Adjust limits as needed
        axins.plot([math.floor(min_intersection) - 0.2, math.ceil(max_intersection) + 0.5], [slo, slo], linewidth=1.1,
                   ls='--', color='blue')
        # adjusted_capacity_indexes = [capacity_x[1] for capacity_x in capacity_indexes]
        # axins.scatter(adjusted_capacity_indexes, [slo] * len(capacity_indexes), marker='8', linewidths=2)
        axins.scatter([capacity_x[1] for capacity_x in capacity_indexes], [slo] * len(capacity_indexes), marker='x',
                      linewidths=1.1)

        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="none", ls='--', linewidth=0.1)
        axins.get_xaxis().set_visible(False)
        axins.set_xticks([])
        axins.set_yticks([])

        ax.plot([min_qps, max_qps], [ttft_slo, ttft_slo], linewidth=1.5, ls='--', color='blue')
        ax.text(max_qps - 2, ttft_slo + 2.8, f"SLO", fontsize=12)

    ax.set_ylabel(metric_name)
    if keep_x_ticks_labels:
        ax.set_xlabel(x_dim, fontsize=12, loc='center')
    if keep_legend:
        ax.legend(fancybox=False, shadow=False, ncol=len(scheduler_list), fontsize=legend_fontsize,
                  loc='upper right', bbox_to_anchor=anchor)
    if slo > 0 and len(capacity_indexes) > 0:
        capacity_maps = {"capacity": {
            scheduler_name: round(capacity, 1) for scheduler_name, capacity in capacity_indexes}
        }
        capacity_df = pd.DataFrame(capacity_maps)
        return capacity_df
    return None


def plot_single_cdf(ax, data, qps, metric_name, x_dim_appendix="", y_dim_appendix="", zoom_out=False,
                    max_x_range_for_zoom=50000, enable_legend=False, enable_title_label=False,
                    enable_x_label=False, enable_y_label=False,
                    bbox_to_anchor=(3.0, 1.315),
                    title_fontsize=14):
    if zoom_out:
        fig, ax = plt.subplots(1, 1)
        axins = inset_axes(plt.gca(), width="60%", height="60%", loc='lower right')
        for key, value in data.items():
            ax.ecdf(value, label=key)
            axins.ecdf(value, label=key)
        axins.set_xlim(0, max_x_range_for_zoom)  # Adjust limits as needed
        axins.set_ylim(0.6, 1)  # Adjust limits as needed
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", ls='--')
        ax.legend(fancybox=True, shadow=True, ncol=1, fontsize=8,
                  loc='upper right', bbox_to_anchor=(1.1, 1.015))
        ax.set_xlabel(metric_name.lower() + x_dim_appendix)
        ax.set_ylabel("CDF")
        ax.set_title(metric_name + " CDF" + y_dim_appendix)
        axins.set_xticks([])
        axins.set_yticks([])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    else:
        for key, value in data.items():
            ax.ecdf(value, label=key, color=scheduler_to_color[key])
        # plt.xlabel(metric_name.lower() + x_dim_appendix)
        # plt.ylabel("CDF")
        # plt.legend(fancybox=True, shadow=True, loc='best')
        # plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        # if enable_label:
        #     ax.set_xlabel(x_dim_appendix, fontsize=12)
        #     ax.xaxis.set_label_coords(-0.125, -0.025)
        #     ax.set_ylabel(f"{metric_name} \n QPS={qps}", fontsize=12)
        # else:
        #     ax.set_ylabel("QPS " + str(qps), fontsize=12)
        if enable_x_label:
            ax.set_xlabel(x_dim_appendix, fontsize=title_fontsize, loc='center')
        if enable_y_label:
            ax.set_ylabel(metric_name, fontsize=title_fontsize)
        if enable_title_label:
            ax.set_title("QPS " + str(qps), fontsize=11)
        if enable_legend:
            ax.legend(fancybox=False, shadow=False, ncol=len(data.keys()), fontsize=title_fontsize,
                      loc='upper right', bbox_to_anchor=bbox_to_anchor)

    # plt.savefig(f"{output_dir_per_qps}/{metric_name}_cdf.png")


def plot_latency_cdf_per_qps(axes, data, output_dir, metric_name, x_dim_appendix="", zoom_out=False,
                             max_x_range_for_zoom=50000, enable_legend_at_middle=False,
                             enable_x_at_middle=False, enable_y_label=True, enable_title_label=False,
                             bbox_to_anchor=(3.0, 1.315)):
    output_dir = output_dir + "/cdf_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    enable_label = True
    mid_point = len(axes) // 2
    i = 0
    for qps in data.keys():
        output_dir_per_qps = output_dir + f"/{qps}"
        if not os.path.exists(output_dir_per_qps):
            os.makedirs(output_dir_per_qps)
        if qps not in axes.keys():
            continue
        ax = axes.get(qps)
        enable_legend = enable_legend_at_middle and i == mid_point
        enable_x_label = enable_x_at_middle and i == mid_point
        enable_y_label = i == 0 and enable_y_label
        plot_single_cdf(ax, data[qps], qps, metric_name, x_dim_appendix,
                        f" under QPS {qps}", zoom_out=zoom_out, max_x_range_for_zoom=max_x_range_for_zoom,
                        enable_legend=enable_legend, enable_y_label=enable_y_label, enable_x_label=enable_x_label,
                        enable_title_label=enable_title_label, bbox_to_anchor=bbox_to_anchor)
        i += 1
        if enable_label:
            enable_label = False


def plot_per_qps(experiments_set, output_dir, min_qps=20, max_qps=36, num_selected_qps_per_figures=4):
    qps_output_dir = output_dir + "/qps"
    if os.path.exists(qps_output_dir):
        shutil.rmtree(qps_output_dir)
    os.makedirs(qps_output_dir)

    token_throughput = {}
    requests_throughput = {}
    average_ttft = {}
    average_tbt = {}
    p99_ttft = {}
    p99_tbt = {}
    average_e2e = {}
    p99_e2e = {}

    ttft_cdfs = {}
    tbt_cdfs = {}
    e2e_cdfs = {}

    avg_free_gpu = {}
    var_free_gpu_per_node = {}
    num_total_preemption = {}
    scheduling_overhead_ratio = {}
    scheduling_overhead = {}

    qps_set = sorted(set([record["qps"] for record in experiments_set]))
    if min_qps > 0:
        qps_set = [qps for qps in qps_set if min_qps <= qps <= max_qps]
    sorted_keys = []
    for qps in qps_set:
        ttft_cdf_per_qps = {}
        tbt_cdfs_per_qps = {}
        e2e_cdfs_per_qps = {}
        ttft_cdfs[qps] = ttft_cdf_per_qps
        tbt_cdfs[qps] = tbt_cdfs_per_qps
        e2e_cdfs[qps] = e2e_cdfs_per_qps
        qps_experiments = [record for record in experiments_set if record["qps"] == qps]
        map_from_name_exp = {}

        avg_free_gpu_per_qps = {}
        # var_free_gpu_per_node = {}
        # num_total_preemption = {}
        var_free_gpu_per_node_per_qps = {}
        num_total_preemption_per_qps = {}
        scheduling_overhead_ratio_per_qps = {}
        scheduling_overhead_per_qps = {}

        avg_free_gpu[qps] = avg_free_gpu_per_qps
        var_free_gpu_per_node[qps] = var_free_gpu_per_node_per_qps
        num_total_preemption[qps] = num_total_preemption_per_qps
        scheduling_overhead_ratio[qps] = scheduling_overhead_ratio_per_qps
        scheduling_overhead[qps] = scheduling_overhead_per_qps

        for experiment in qps_experiments:
            experiment_name = f"{experiment['scheduler_name']}".replace("_", " ")
            for key in experiment_name_replacement.keys():
                if key in experiment_name:
                    experiment_name = experiment_name.replace(key, experiment_name_replacement[key])
                if experiment_name == "Block" and experiment["use_length_estimation"]:
                    experiment_name += "*"
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
            if index_name not in map_from_name_exp:
                continue
            experiments = map_from_name_exp[index_name]

            if index_name not in token_throughput:
                token_throughput[index_name] = {}
                requests_throughput[index_name] = {}
                average_ttft[index_name] = {}
                average_tbt[index_name] = {}
                p99_ttft[index_name] = {}
                p99_tbt[index_name] = {}
                average_e2e[index_name] = {}
                p99_e2e[index_name] = {}

            token_throughput[index_name][qps] = float(experiments['token_throughput'])
            requests_throughput[index_name][qps] = float(experiments['request_throughput'])
            average_ttft[index_name][qps] = int(np.mean(experiments['ttft'])) * 1.0 / 1000
            average_tbt[index_name][qps] = np.mean(experiments['tbt'])
            p99_ttft[index_name][qps] = int(np.percentile(experiments['ttft'], 99)) * 1.0 / 1000
            p99_tbt[index_name][qps] = np.percentile(experiments['tbt'], 99)
            average_e2e[index_name][qps] = int(np.mean(experiments['e2e']) * 1.0) / 1000
            p99_e2e[index_name][qps] = int(np.percentile(experiments['e2e'], 99)) * 1.0 / 1000

            ttft_cdf_per_qps[index_name] = experiments['ttft'] * 1.0 / 1000
            tbt_cdfs_per_qps[index_name] = experiments['tbt'] * 1.0 / 1000
            e2e_cdfs_per_qps[index_name] = experiments['e2e'] * 1.0 / 1000
            avg_free_gpu_per_qps[index_name] = experiments['avg_gpu_blocks']
            var_free_gpu_per_node_per_qps[index_name] = experiments['var_gpu_blocks']
            num_preempted_list = (experiments['num_preempted'] - experiments['num_preempted'][0]).tolist()
            num_preempted = np.asarray([max(0, preempted) for preempted in num_preempted_list])
            num_total_preemption_per_qps[index_name] = num_preempted

            current_prediction_overhead = experiments['scheduling_overhead']
            end_to_end_latencies = experiments['e2e']
            scheduling_overhead_ratio_per_qps[index_name] = [(100.0 * overhead / latency)
                                                             for overhead, latency
                                                             in
                                                             zip(current_prediction_overhead, end_to_end_latencies)
                                                             if latency > 0]
            scheduling_overhead_per_qps[index_name] = current_prediction_overhead

    index_names = sorted_keys

    fig, axs = plt.subplots(2, 3)

    plot_linear_with_different_qps(axs[0, 0], average_e2e, "Average Request Latency (s)")
    plot_linear_with_different_qps(axs[1, 0], p99_e2e, "Request Latency P99 (s)")

    plot_linear_with_different_qps(axs[0, 1], average_ttft, "Average TTFT (s)", keep_legend=True,
                                   keep_x_ticks_labels=True, x_dim="QPS", anchor=(2.0, 1.28))

    capacity_df = plot_linear_with_different_qps(axs[1, 1], p99_ttft, "TTFT P99 (s)",
                                                 x_dim="QPS", slo=ttft_slo)

    plot_linear_with_different_qps(axs[0, 2], requests_throughput, "Request Throughput (r/s)")

    plot_bar_chart(axs[1, 2], capacity_df, "Max QPS before Hitting SLO", column_name='capacity')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.25)
    fig.set_size_inches(14, 6)
    fig.savefig(f"{qps_output_dir}/qps.png", bbox_inches='tight')

    selected_qps_per_figures = [qps for qps in qps_set if qps % num_selected_qps_per_figures == 0]

    fig, axs = plt.subplots(2, len(selected_qps_per_figures))
    axes_dict_for_ttft = {}
    axes_dict_for_e2e = {}
    i = 0
    for qps in selected_qps_per_figures:
        axes_dict_for_ttft[qps] = axs[0, i]
        axes_dict_for_e2e[qps] = axs[1, i]
        i += 1
    plot_latency_cdf_per_qps(axes_dict_for_ttft, ttft_cdfs, qps_output_dir, "TTFT", "Time(s)",
                             enable_legend_at_middle=True, enable_title_label=True,
                             bbox_to_anchor=(3.0, 1.355))
    # plot_latency_cdf_per_qps(tbt_cdfs, qps_output_dir, "TBT", " (ms)", max_x_range_for_zoom=100000)
    plot_latency_cdf_per_qps(axes_dict_for_e2e,
                             e2e_cdfs, qps_output_dir, "Request Latency", "Time(s)",
                             max_x_range_for_zoom=100000, enable_x_at_middle=True,
                             bbox_to_anchor=(2.0, 1.255))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    fig.set_size_inches(18, 6)
    fig.savefig(f"{qps_output_dir}/cdf.png", bbox_inches='tight')

    # plot_linear(avg_free_gpu, "Average Free GPU Blocks", qps_output_dir, sigma=10)
    # plot_linear(var_free_gpu_per_node, "Free GPU Blocks Var", qps_output_dir, sigma=20,
    #             adjust_legend=True)
    # plot_linear(num_total_preemption, "Number of new Preemption", qps_output_dir, sigma=10)

    fig, axs = plt.subplots(3, len(selected_qps_per_figures))
    axs_for_avg_free_gpu = {}
    axs_for_var_free_gpu = {}
    axs_for_num_preemption = {}
    i = 0
    for qps in selected_qps_per_figures:
        axs_for_avg_free_gpu[qps] = axs[0, i]
        axs_for_var_free_gpu[qps] = axs[1, i]
        axs_for_num_preemption[qps] = axs[2, i]
        i += 1
    plot_linear_for_multiple_qps(axs_for_avg_free_gpu, avg_free_gpu, "Free GPU Blocks Mean", sigma=20,
                                 enable_legend_at_middle=True, enable_title_labels=True,
                                 legend_anchor=(3.0, 1.455))
    plot_linear_for_multiple_qps(axs_for_var_free_gpu, var_free_gpu_per_node, "Free GPU Blocks Var",
                                 sigma=20)
    plot_linear_for_multiple_qps(axs_for_num_preemption, num_total_preemption, "Total Preemption Count",
                                 sigma=20, enable_x_label_at_middle=True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    fig.set_size_inches(16, 8)
    fig.savefig(f"{qps_output_dir}/linear.png", bbox_inches='tight')

    fig, axs = plt.subplots(2, len(selected_qps_per_figures))
    i = 0
    axs_for_scheduling_overhead_ratio = {}
    axs_for_scheduling_overhead = {}
    for qps in selected_qps_per_figures:
        axs_for_scheduling_overhead_ratio[qps] = axs[0, i]
        axs_for_scheduling_overhead[qps] = axs[1, i]
        i += 1
    plot_linear_for_multiple_qps(axs_for_scheduling_overhead_ratio, scheduling_overhead_ratio,
                                 "Overhead Ratio (%)", sigma=80,
                                 enable_legend_at_middle=True, x_label="Query ID",
                                 legend_anchor=(2.5, 1.40), title_fontsize=12, enable_title_labels=True)
    plot_linear_for_multiple_qps(axs_for_scheduling_overhead, scheduling_overhead,
                                 "Overhead Latency (ms)", sigma=80,
                                 enable_legend_at_middle=False, x_label="Query ID",
                                 legend_anchor=(1.1, 1.35), title_fontsize=12, enable_x_label_at_middle=True)
    fig.tight_layout()
    fig.set_size_inches(18, 6)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.savefig(f"{qps_output_dir}/overhead.png", bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='Plot the results of the experiments')
    parser.add_argument("--experiments-dir", type=str, default="experiments_analysis/experiment_output/sharegpt")
    parser.add_argument("--output-dir", type=str, default="./experiments_analysis/exp_plots")
    parser.add_argument("--plot-per-qps", type=bool, default=True)
    parser.add_argument("--plot-per-scheduler", type=bool, default=True)
    parser.add_argument("--ttft-p99-slo", type=float, default=3)
    # parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    data_dir = os.getcwd() + "/" + args.experiments_dir

    global ttft_slo
    ttft_slo = args.ttft_p99_slo

    experiments_set = []
    for scheduler_name in os.listdir(data_dir):
        scheduler_dir = data_dir + "/" + scheduler_name
        if scheduler_name == 'logs':
            continue
        for root, dirs, files in os.walk(scheduler_dir):
            for directory in dirs:
                if directory == 'running_logs':
                    continue
                record = {"scheduler_name": scheduler_name}
                experiments_set.append(record)
                qps, n, use_length_est = directory_name_parser(directory)
                record["qps"] = float(qps)
                record["n"] = int(n)
                record["use_length_estimation"] = use_length_est
                for experiments_trace in os.listdir(scheduler_dir + "/" + directory):
                    if experiments_trace.endswith("logs.txt"):
                        metrics = extract_data_from_log_file(scheduler_dir + "/" + directory + "/" + experiments_trace)
                        record["token_throughput"] = metrics['tokens_per_s']
                        record["request_throughput"] = metrics['qps']
                    if experiments_trace.endswith("npz"):
                        b = np.load(scheduler_dir + "/" + directory + "/" + experiments_trace)
                        record['ttft'] = b['prefill_token_latencies']
                        record['tbt'] = b['decode_sum_latencies']
                        record['e2e'] = b['request_latencies']
                        record['avg_gpu_blocks'] = b['avg_gpu_blocks']
                        record['var_gpu_blocks'] = b['var_gpu_blocks']
                        record['num_preempted'] = b['num_preempted']
                        record['scheduling_overhead'] = b['scheduling_overhead']
    if args.plot_per_qps:
        plot_per_qps(experiments_set, args.output_dir)

    # if args.plot_per_scheduler:
    #     plot_per_scheduler(experiments_set, args.output_dir)


if __name__ == "__main__":
    main()
