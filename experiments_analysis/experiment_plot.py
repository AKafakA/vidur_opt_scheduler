import argparse
import os
import re
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.pyplot as plt
import numpy as np
import shutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d

experiment_name_replacement = {"min latency": "block"}
scheduler_name_ordered = ['random','block*', 'block']


def directory_name_parser(directory_name):
    directory_name = directory_name.split("_")
    qps = directory_name[1]
    n = directory_name[-1]
    return qps, n


def extract_data_from_log_file(log_file):
    pattern = r"""
        dur_s\s(?P<dur_s>\d+\.\d+)\s tokens_per_s\s(?P<tokens_per_s>\d+\.\d+)\s qps\s(?P<qps>\d+\.\d+)\s
    """
    with open(log_file, "r") as f:
        log = f.read()
        match = re.search(pattern, log, re.VERBOSE)
        return match.groupdict()


def plot_linear(data, metric_name, output_dir, y_dim_appendix="Per Node", sigma=-1, title_appendix=""):
    plt.figure()
    output_dir = output_dir + "/linear_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for key, value in data.items():
        # smooth by guassian 1d
        if sigma > 0:
            value = gaussian_filter1d(value, sigma)
        plt.plot(value, label=key)

    plt.xlabel("Request ID")
    plt.ylabel(metric_name + " " + y_dim_appendix)
    plt.title(metric_name + title_appendix)
    plt.legend(fancybox=True, shadow=True)
    plt.savefig(f"{output_dir}/{metric_name}_linear.png")


def plot_bar_chart(dataframe, index_names, output_dir, metric_name, x_dim="QPS", stack_data=False, plot_kind='bar',
                   xt_rotation='horizontal', legend_title='', zoom_out=False):
    plt.figure()
    output_dir = output_dir + "/bar_charts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if zoom_out:
        fig, ax = plt.subplots(1, 1)
        axins = inset_axes(plt.gca(), width="40%", height="30%", loc='upper left')
        dataframe.plot(x=x_dim, y=list(index_names), kind=plot_kind, stacked=stack_data, ax=ax)
        dataframe.plot(x=x_dim, y=list(index_names), kind=plot_kind, stacked=stack_data, ax=axins)
        max_value = sorted(dataframe[4:5].values.tolist()[0][1:])[-2]
        axins.set_xlim(2.5, 4.5)  # Adjust limits as needed
        axins.set_ylim(0, max_value)  # Adjust limits as needed
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", ls= '--')
        ax.legend(fancybox=True, shadow=True, ncol=1, fontsize=8,
                   loc='upper right', bbox_to_anchor=(1.1, 1.015))
        axins.get_legend().remove()
        axins.get_xaxis().set_visible(False)
        ax.set_xlabel(x_dim)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name + " Per " + x_dim)
        axins.set_xticks([])
        axins.set_yticks([])
        ax.tick_params("x", rotation=0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    else:
        dataframe.plot(x=x_dim, y=list(index_names), kind=plot_kind, stacked=stack_data)
        plt.xlabel(x_dim)
        plt.ylabel(metric_name)
        plt.title(metric_name + " Per " + x_dim)
        plt.xticks(rotation=xt_rotation)
        if legend_title:
            plt.legend(fancybox=True, shadow=True, ncol=1, fontsize=8, title=legend_title, title_fontsize='small',
                       loc='upper right', bbox_to_anchor=(1.1, 1.015))
        else:
            plt.legend(ncol=3, fontsize=8, loc='best', fancybox=True, shadow=True)
        plt.tight_layout()
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.savefig(f"{output_dir}/{metric_name}_bar_chart.png")

def plot_single_cdf(data, output_dir_per_qps, metric_name, x_dim_appendix="", y_dim_appendix="", zoom_out=False,
                    max_x_range_for_zoom=50000):
    plt.figure()
    if zoom_out:
        fig, ax = plt.subplots(1, 1)
        axins = inset_axes(plt.gca(), width="60%", height="60%", loc='lower right')
        for key, value in data.items():
            ax.ecdf(value, label=key)
            axins.ecdf(value, label=key)
        axins.set_xlim(0, max_x_range_for_zoom)  # Adjust limits as needed
        axins.set_ylim(0.6, 1)  # Adjust limits as needed
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", ls= '--')
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
             plt.ecdf(value, label=key)
        plt.xlabel(metric_name.lower() + x_dim_appendix)
        plt.ylabel("CDF")
        plt.legend(fancybox=True, shadow=True, loc='best')
        plt.title(metric_name + " CDF" + y_dim_appendix)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.savefig(f"{output_dir_per_qps}/{metric_name}_cdf.png")

def plot_latency_cdf_per_qps(data, output_dir, metric_name, x_dim_appendix="", zoom_out=False, max_x_range_for_zoom=50000):
    output_dir = output_dir + "/cdf_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for qps in data.keys():
        output_dir_per_qps =  output_dir + f"/{qps}"
        if not os.path.exists(output_dir_per_qps ):
            os.makedirs(output_dir_per_qps)
        plot_single_cdf(data[qps], output_dir_per_qps, metric_name, x_dim_appendix,
                        f" under QPS {qps}", zoom_out=zoom_out,max_x_range_for_zoom=max_x_range_for_zoom)


def plot_per_scheduler(experiments_set, output_dir, scheduler_excluded="round_robin"):
    if scheduler_excluded is None:
        scheduler_excluded = ['round_robin']
    exp_output_dir = output_dir + "/scheduler"
    if os.path.exists(exp_output_dir):
        shutil.rmtree(exp_output_dir)
    os.makedirs(exp_output_dir)
    experiments_data = {}
    qps_set = sorted(set([record["qps"] for record in experiments_set]))
    for record in experiments_set:
        if record['scheduler_name'] == scheduler_excluded:
            continue
        experiment_name = f"{record['scheduler_name']}".replace("_", " ")
        # if experiments_name.startswith("min") or experiments_name.startswith("max"):
        #     experiments_name = experiments_name + f"_{record['n']}"
        for key in experiment_name_replacement.keys():
            if key in experiment_name:
                experiment_name = experiment_name.replace(key, experiment_name_replacement[key])
        if experiment_name not in experiments_data:
            experiments_data[experiment_name] = [record]
        else:
            experiments_data[experiment_name].append(record)

    token_throughput = []
    requests_throughput = []
    average_ttft = []
    average_tbt = []
    p99_ttft = []
    p99_tbt = []
    average_e2e = []
    p99_e2e = []

    for experiment_name, records in experiments_data.items():
        output_dir = exp_output_dir
        token_throughput_data = [f"{experiment_name}"]
        requests_throughput_data = [f"{experiment_name}"]
        average_ttft_data = [f"{experiment_name}"]
        average_tbt_data = [f"{experiment_name}"]
        p99_ttft_data = [f"{experiment_name}"]
        p99_tbt_data = [f"{experiment_name}"]
        average_e2e_data = [f"{experiment_name}"]
        p99_e2e_data = [f"{experiment_name}"]

        for qps in qps_set:
            for record in records:
                if record["qps"] == qps:
                    token_throughput_data.append(float(record['token_throughput']))
                    requests_throughput_data.append(float(record['request_throughput']))
                    average_ttft_data.append(np.mean(record['ttft']))
                    average_tbt_data.append(np.mean(record['tbt']))
                    p99_ttft_data.append(np.percentile(record['ttft'], 99))
                    p99_tbt_data.append(np.percentile(record['tbt'], 99))
                    average_e2e_data.append(np.mean(record['e2e']))
                    p99_e2e_data.append(np.percentile(record['e2e'], 99))
        token_throughput.append(token_throughput_data)
        requests_throughput.append(requests_throughput_data)
        average_ttft.append(average_ttft_data)
        average_tbt.append(average_tbt_data)
        p99_ttft.append(p99_ttft_data)
        p99_tbt.append(p99_tbt_data)
        average_e2e.append(average_e2e_data)
        p99_e2e.append(p99_e2e_data)

    token_s_df = pd.DataFrame(token_throughput, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(token_s_df, qps_set, output_dir, "Token Throughput", "Scheduler",
                   xt_rotation='horizontal', legend_title="QPS")
    requests_throughput_df = pd.DataFrame(requests_throughput, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(requests_throughput_df, qps_set, output_dir, "Request Throughput", "Scheduler",
                   xt_rotation='horizontal', legend_title="QPS")
    average_ttft_df = pd.DataFrame(average_ttft, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(average_ttft_df, qps_set, output_dir, "Average TTFT", "Scheduler",
                   xt_rotation='horizontal', legend_title="QPS")
    average_tbt_df = pd.DataFrame(average_tbt, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(average_tbt_df, qps_set, output_dir, "Average TBT", "Scheduler",
                   xt_rotation='horizontal', legend_title="QPS")
    p99_ttft_df = pd.DataFrame(p99_ttft, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(p99_ttft_df, qps_set, output_dir, "TTFT P99", "Scheduler",
                   xt_rotation='horizontal', legend_title="QPS")
    p99_tbt_df = pd.DataFrame(p99_tbt, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(p99_tbt_df, qps_set, output_dir, "TBT P99", "Scheduler",
                   xt_rotation='horizontal', legend_title="QPS")
    average_e2e_df = pd.DataFrame(average_e2e, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(average_e2e_df, qps_set, output_dir, "Request Latency", "Scheduler",
                   xt_rotation='horizontal', legend_title="QPS")
    p99_e2e_df = pd.DataFrame(p99_e2e, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(p99_e2e_df, qps_set, output_dir, "Request Latency P99", "Scheduler",
                   xt_rotation='horizontal', legend_title="QPS")


def plot_per_qps(experiments_set, output_dir, min_qps = 16.0):
    qps_output_dir = output_dir + "/qps"
    if os.path.exists(qps_output_dir):
        shutil.rmtree(qps_output_dir)
    os.makedirs(qps_output_dir)
    token_throughput = []
    requests_throughput = []
    average_ttft = []
    average_tbt = []
    p99_ttft = []
    p99_tbt = []
    average_e2e = []
    p99_e2e = []

    ttft_cdfs = {}
    tbt_cdfs = {}
    e2e_cdfs = {}

    avg_free_gpu = {}
    var_free_gpu_per_node = {}

    qps_set = sorted(set([record["qps"] for record in experiments_set]))
    if min_qps > 0:
        qps_set = [qps for qps in qps_set if qps >= min_qps]
    sorted_keys = []
    for qps in qps_set:
        token_s_data = [f"{qps}"]
        requests_throughput_data = [f"{qps}"]
        average_ttft_data = [f"{qps}"]
        average_tbt_data = [f"{qps}"]
        p99_ttft_data = [f"{qps}"]
        p99_tbt_data = [f"{qps}"]

        average_e2e_data = [f"{qps}"]
        p99_e2e_data = [f"{qps}"]

        ttft_cdf_per_qps = {}
        tbt_cdfs_per_qps = {}
        e2e_cdfs_per_qps = {}
        ttft_cdfs[qps] = ttft_cdf_per_qps
        tbt_cdfs[qps] = tbt_cdfs_per_qps
        e2e_cdfs[qps] = e2e_cdfs_per_qps
        qps_experiments = [record for record in experiments_set if record["qps"] == qps]
        map_from_name_exp = {}
        for experiment in qps_experiments:
            experiment_name = f"{experiment['scheduler_name']}".replace("_", " ")
            for key in experiment_name_replacement.keys():
                if key in experiment_name:
                    experiment_name = experiment_name.replace(key, experiment_name_replacement[key])
            map_from_name_exp[experiment_name] = experiment
        if len(sorted_keys) == 0:
            sorted_keys = sorted(map_from_name_exp.keys())
            for key in scheduler_name_ordered:
                if key in sorted_keys:
                    sorted_keys.remove(key)
            sorted_keys = sorted_keys + scheduler_name_ordered
        for index_name in sorted_keys:
            experiments = map_from_name_exp[index_name]
            token_s_data.append(float(experiments['token_throughput']))
            requests_throughput_data.append(float(experiments['request_throughput']))
            average_ttft_data.append(np.mean(experiments['ttft']))
            average_tbt_data.append(np.mean(experiments['tbt']))
            p99_tbt_data.append(np.percentile(experiments['tbt'], 99))
            p99_ttft_data.append(np.percentile(experiments['ttft'], 99))
            average_e2e_data.append(np.mean(experiments['e2e']))
            p99_e2e_data.append(np.percentile(experiments['e2e'], 99))
            ttft_cdf_per_qps[index_name] = experiments['ttft']
            tbt_cdfs_per_qps[index_name] = experiments['tbt']
            e2e_cdfs_per_qps[index_name] = experiments['e2e']
            avg_free_gpu[index_name] = experiments['avg_gpu_blocks']
            var_free_gpu_per_node[index_name] = experiments['var_gpu_blocks']

        token_throughput.append(token_s_data)
        requests_throughput.append(requests_throughput_data)
        average_ttft.append(average_ttft_data)
        average_tbt.append(average_tbt_data)
        p99_ttft.append(p99_ttft_data)
        p99_tbt.append(p99_tbt_data)
        average_e2e.append(average_e2e_data)
        p99_e2e.append(p99_e2e_data)

    index_names = sorted_keys

    token_s_df = pd.DataFrame(token_throughput, columns=['QPS'] + list(index_names))
    plot_bar_chart(token_s_df, index_names, qps_output_dir, "Token Throughput", "QPS")
    requests_throughput_df = pd.DataFrame(requests_throughput, columns=['QPS'] + list(index_names))
    plot_bar_chart(requests_throughput_df, index_names, qps_output_dir, "Request Throughput", "QPS")
    average_ttft_df = pd.DataFrame(average_ttft, columns=['QPS'] + list(index_names))
    plot_bar_chart(average_ttft_df, index_names, qps_output_dir, "Average TTFT", "QPS", zoom_out=True)
    average_tbt_df = pd.DataFrame(average_tbt, columns=['QPS'] + list(index_names))
    plot_bar_chart(average_tbt_df, index_names, qps_output_dir, "Average TBT", "QPS")
    p99_ttft_df = pd.DataFrame(p99_ttft, columns=['QPS'] + list(index_names))
    plot_bar_chart(p99_ttft_df, index_names, qps_output_dir, "TTFT P99", "QPS", zoom_out=True)
    p99_tbt_df = pd.DataFrame(p99_tbt, columns=['QPS'] + list(index_names))
    plot_bar_chart(p99_tbt_df, index_names, qps_output_dir, "TBT P99", "QPS", zoom_out=True)

    average_e2e_df = pd.DataFrame(average_e2e, columns=['QPS'] + list(index_names))
    plot_bar_chart(average_e2e_df, index_names, qps_output_dir, "Average Request Latency", "QPS", zoom_out=True)
    p99_e2e_df = pd.DataFrame(p99_e2e, columns=['QPS'] + list(index_names))
    plot_bar_chart(p99_e2e_df, index_names, qps_output_dir, "Request Latency P99", "QPS", zoom_out=True)

    plot_latency_cdf_per_qps(ttft_cdfs, qps_output_dir, "TTFT", " (ms)", zoom_out=True)
    plot_latency_cdf_per_qps(tbt_cdfs, qps_output_dir, "TBT", " (ms)",
                             zoom_out=True, max_x_range_for_zoom=100000)
    plot_latency_cdf_per_qps(e2e_cdfs, qps_output_dir, "Request Latency", " (ms)",
                             zoom_out=True, max_x_range_for_zoom=100000)

    plot_linear(avg_free_gpu, "Average Free GPU Blocks", qps_output_dir, sigma=10,
                title_appendix=f" Under QPS {qps} ")
    plot_linear(var_free_gpu_per_node, "Free GPU Blocks Var", qps_output_dir, sigma=10,
                title_appendix=f" Under QPS {qps} ")


def main():
    parser = argparse.ArgumentParser(description='Plot the results of the experiments')
    parser.add_argument("--experiments-dir", type=str, default="./experiments_analysis/experiment_output")
    parser.add_argument("--output-dir", type=str, default="./experiments_analysis/exp_plots")
    parser.add_argument("--plot-per-qps", type=bool, default=True)
    parser.add_argument("--plot-per-scheduler", type=bool, default=True)
    # parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    data_dir = os.getcwd() + "/" + args.experiments_dir

    experiments_set = []
    for scheduler_name in os.listdir(data_dir):
        scheduler_dir = data_dir + "/" + scheduler_name
        if scheduler_name == 'logs':
            continue
        for root, dirs, files in os.walk(scheduler_dir):
            for directory in dirs:
                record = {"scheduler_name": scheduler_name}
                experiments_set.append(record)
                qps, n = directory_name_parser(directory)
                record["qps"] = float(qps)
                record["n"] = int(n)
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
    if args.plot_per_qps:
        plot_per_qps(experiments_set, args.output_dir)

    if args.plot_per_scheduler:
        plot_per_scheduler(experiments_set, args.output_dir)


if __name__ == "__main__":
    main()
