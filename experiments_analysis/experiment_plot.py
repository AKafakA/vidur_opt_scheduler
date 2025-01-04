import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


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


def plot_linear(data, metric_name, output_dir, y_dim_appendix="Per Node"):
    plt.figure()
    output_dir = output_dir + "/linear_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for key, value in data.items():
        plt.plot(value, label=key)

    plt.xlabel("Request ID")
    plt.ylabel(metric_name + " " + y_dim_appendix)
    plt.title(metric_name.upper())
    plt.legend()
    plt.savefig(f"{output_dir}/{metric_name}_linear.png")


def plot_bar_chart(dataframe, index_names, output_dir, metric_name, x_dim="QPS", stack_data=False, plot_kind='bar',
                   xt_rotation='horizontal'):
    plt.figure()
    output_dir = output_dir + "/bar_charts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataframe.plot(x=x_dim, y=list(index_names), kind=plot_kind, stacked=stack_data)
    plt.xlabel(x_dim)
    plt.ylabel(metric_name)
    plt.title(metric_name + " Per " + x_dim)
    plt.xticks(rotation=xt_rotation)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))
    plt.savefig(f"{output_dir}/{metric_name}_bar_chart.png")


def plot_cdf(data, output_dir, metric_name, x_dim_appendix=""):
    plt.figure()
    output_dir = output_dir + "/cdf_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for key, value in data.items():
        plt.ecdf(value, label=key)

    plt.xlabel(metric_name.lower() + x_dim_appendix)
    plt.ylabel("CDF")
    plt.title(metric_name + " CDF")
    plt.legend()
    plt.savefig(f"{output_dir}/{metric_name}_cdf.png")


def plot_per_scheduler(experiments_set, output_dir):
    exp_output_dir = output_dir + "/scheduler"
    if not os.path.exists(exp_output_dir):
        os.makedirs(exp_output_dir)
    experiments_data = {}
    qps_set = set([record["qps"] for record in experiments_set])
    for record in experiments_set:
        experiments_name = f"{record['scheduler_name']}"
        if experiments_name.startswith("min") or experiments_name.startswith("max"):
            experiments_name = experiments_name + f"_{record['n']}"
        if experiments_name not in experiments_data:
            experiments_data[experiments_name] = [record]
        else:
            experiments_data[experiments_name].append(record)

    token_throughput = []
    requests_throughput = []
    average_ttft = []
    average_tbt = []
    p99_ttft = []
    p99_tbt = []
    average_e2e = []
    p99_e2e = []

    for experiment_name, records in experiments_data.items():
        output_dir = exp_output_dir + "/" + experiment_name
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
                   xt_rotation='horizontal')
    requests_throughput_df = pd.DataFrame(requests_throughput, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(requests_throughput_df, qps_set, output_dir, "Request Throughput", "Scheduler",
                   xt_rotation='horizontal')
    average_ttft_df = pd.DataFrame(average_ttft, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(average_ttft_df, qps_set, output_dir, "Average TTFT", "Scheduler",
                   xt_rotation='horizontal')
    average_tbt_df = pd.DataFrame(average_tbt, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(average_tbt_df, qps_set, output_dir, "Average TBT", "Scheduler",
                   xt_rotation='horizontal')
    p99_ttft_df = pd.DataFrame(p99_ttft, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(p99_ttft_df, qps_set, output_dir, "TTFT P99", "Scheduler",
                   xt_rotation='horizontal')
    p99_tbt_df = pd.DataFrame(p99_tbt, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(p99_tbt_df, qps_set, output_dir, "TBT P99", "Scheduler",
                   xt_rotation='horizontal')
    average_e2e_df = pd.DataFrame(average_e2e, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(average_e2e_df, qps_set, output_dir, "Request Latency", "Scheduler",
                   xt_rotation='horizontal')
    p99_e2e_df = pd.DataFrame(p99_e2e, columns=['Scheduler'] + list(qps_set))
    plot_bar_chart(p99_e2e_df, qps_set, output_dir, "Request Latency P99", "Scheduler",
                   xt_rotation='horizontal')


def plot_per_qps(experiments_set, output_dir):
    qps_output_dir = output_dir + "/qps"
    if not os.path.exists(qps_output_dir):
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

    index_names = set()
    qps_set = set([record["qps"] for record in experiments_set])
    for qps in qps_set:
        token_s_data = [f"{qps}"]
        requests_throughput_data = [f"{qps}"]
        average_ttft_data = [f"{qps}"]
        average_tbt_data = [f"{qps}"]
        p99_ttft_data = [f"{qps}"]
        p99_tbt_data = [f"{qps}"]
        qps_experiments = [record for record in experiments_set if record["qps"] == qps]
        for experiments in qps_experiments:
            experiment_name = f"{experiments['scheduler_name']}"
            if experiment_name.startswith("min") or experiment_name.startswith("max"):
                experiment_name = experiment_name + f"_{experiments['n']}"
            index_names.add(experiment_name)
            token_s_data.append(float(experiments['token_throughput']))
            requests_throughput_data.append(float(experiments['request_throughput']))
            average_ttft_data.append(np.mean(experiments['ttft']))
            average_tbt_data.append(np.mean(experiments['tbt']))
            average_e2e.append(np.mean(experiments['e2e']))
            p99_tbt_data.append(np.percentile(experiments['tbt'], 99))
            p99_ttft_data.append(np.percentile(experiments['ttft'], 99))
            p99_e2e.append(np.percentile(experiments['e2e'], 99))
            ttft_cdfs[experiment_name] = experiments['ttft']
            tbt_cdfs[experiment_name] = experiments['tbt']
            e2e_cdfs[experiment_name] = experiments['e2e']
            avg_free_gpu[experiment_name] = experiments['avg_gpu_blocks']
            var_free_gpu_per_node[experiment_name] = experiments['var_gpu_blocks']

        token_throughput.append(token_s_data)
        requests_throughput.append(requests_throughput_data)
        average_ttft.append(average_ttft_data)
        average_tbt.append(average_tbt_data)
        p99_ttft.append(p99_ttft_data)
        p99_tbt.append(p99_tbt_data)

    token_s_df = pd.DataFrame(token_throughput, columns=['QPS'] + list(index_names))
    plot_bar_chart(token_s_df, index_names, qps_output_dir, "Token Throughput", "QPS")
    requests_throughput_df = pd.DataFrame(requests_throughput, columns=['QPS'] + list(index_names))
    plot_bar_chart(requests_throughput_df, index_names, qps_output_dir, "Request Throughput", "QPS")
    average_ttft_df = pd.DataFrame(average_ttft, columns=['QPS'] + list(index_names))
    plot_bar_chart(average_ttft_df, index_names, qps_output_dir, "Average TTFT", "QPS")
    average_tbt_df = pd.DataFrame(average_tbt, columns=['QPS'] + list(index_names))
    plot_bar_chart(average_tbt_df, index_names, qps_output_dir, "Average TBT", "QPS")
    p99_ttft_df = pd.DataFrame(p99_ttft, columns=['QPS'] + list(index_names))
    plot_bar_chart(p99_ttft_df, index_names, qps_output_dir, "TTFT P99", "QPS")
    p99_tbt_df = pd.DataFrame(p99_tbt, columns=['QPS'] + list(index_names))
    plot_bar_chart(p99_tbt_df, index_names, qps_output_dir, "TBT P99", "QPS")
    plot_cdf(ttft_cdfs, qps_output_dir, "TTFT", " (ms)")
    plot_cdf(tbt_cdfs, qps_output_dir, "TBT", " (ms)")
    plot_cdf(e2e_cdfs, qps_output_dir, "Request Latency", " (ms)")

    plot_linear(avg_free_gpu, "Average Free GPU Blocks", qps_output_dir)
    plot_linear(var_free_gpu_per_node, "Free GPU Blocks Var", qps_output_dir)


def main():
    parser = argparse.ArgumentParser(description='Plot the results of the experiments')
    parser.add_argument("--experiments-dir", type=str, default="./experiments_analysis/exp_output")
    parser.add_argument("--output-dir", type=str, default="./experiments_analysis/exp_plots")
    parser.add_argument("--plot-per-qps", type=bool, default=True)
    parser.add_argument("--plot-per-scheduler", type=bool, default=True)
    # parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    data_dir = os.getcwd() + "/" + args.experiments_dir

    experiments_set = []
    for scheduler_name in os.listdir(data_dir):
        scheduler_dir = data_dir + "/" + scheduler_name
        for root, dirs, files in os.walk(scheduler_dir):
            for directory in dirs:
                record = {"scheduler_name": scheduler_name}
                experiments_set.append(record)
                qps, n = directory_name_parser(directory)
                record["qps"] = qps
                record["n"] = n
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
