import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

from vidur.types.optimal_global_scheduler_target_metric import from_value_to_short_metrics_name, TargetMetric

MAX_QPS_UNDER_SLO_MARKER = "Max QPS under SLO for"


def generate_label_from_global_scheduler_config(row):
    row = ast.literal_eval(row['cluster_config'])
    global_scheduler_name = row["global_scheduler_config"]["name"]
    if global_scheduler_name == "opt":
        target_metric = row['global_scheduler_config']['target_metric']
        if isinstance(target_metric, int):
            target_metric = from_value_to_short_metrics_name(target_metric)
        else:
            target_metric = from_value_to_short_metrics_name(TargetMetric.from_str(target_metric))
        return f"{global_scheduler_name}_{target_metric}"
    else:
        return global_scheduler_name


def flatten_dict(value, g, c=None):
    if c is None:
        c = []
    if isinstance(value, dict):
        for key, val in value.items():
            flatten_dict(val, g, c + [key])
    else:
        c.append(value)
        g.append(c)


def plot_bar_chart(stats_dir, stats, metric_name, print_stats=True):
    plt.figure()
    target_dir = f"{stats_dir}/bar_charts"
    os.makedirs(target_dir, exist_ok=True)
    for key, value in stats.items():
        plt.bar(key, value)

    plt.xlabel("Configuration")
    plt.ylabel(metric_name)
    plt.title(metric_name + " Bar Chart")
    plt.savefig(f"{target_dir}/{metric_name}_bar_chart.png")

    if print_stats:
        print(metric_name)
        for key, value in stats.items():
            print(f"\t {key}: {value}")


def if_condition(row, condition_list) -> bool:
    target_value = condition_list[-1]
    keys = condition_list[0:-1]
    for key in keys:
        if isinstance(row, str) and row.startswith("{"):
            row = ast.literal_eval(row)
        row = row[key]
    if isinstance(row, float) and isinstance(target_value, int):
        return target_value == int(row)
    return row == target_value


def data_filter(dataframe, rows_filtering_condition: dict):
    g = []
    flatten_dict(rows_filtering_condition, g)
    selected_rows_index = set()
    for index, row in dataframe.iterrows():
        selected = []
        for condition in g:
            selected.append(if_condition(row, condition))
        if all(selected):
            selected_rows_index.add(index)
    return dataframe.loc[list(selected_rows_index)]


def parse_cdfs(cdf_str):
    cdf_str = cdf_str.replace("[", "").replace("]", "").replace(" ", "")
    cdfs = cdf_str.split(",")
    return [float(cdf) for cdf in cdfs if cdf]


def plot_cdf(stats_dir, cdfs, metric_name):
    plt.figure()
    target_dir = f"{stats_dir}/cdf_plots"
    os.makedirs(target_dir, exist_ok=True)
    for key, cdf in cdfs.items():
        values = parse_cdfs(cdf)
        values = np.array(values)
        num_bins = len(values)
        counts, bin_edges = np.histogram(values, bins=num_bins)
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[1:], cdf, label=key)

    plt.xlabel(metric_name)
    plt.ylabel("CDF")
    plt.title(metric_name + " CDF")
    plt.legend()
    plt.savefig(f"{target_dir}/{metric_name}_cdf.png")


def process_stats(stats_dir, data_filtering_condition: dict, label_getter: callable):
    ttft_cdfs = {}
    tbt_cdfs = {}
    batch_size_cdfs = {}
    scheduling_delay_cdfs = {}

    normalized_e2e_time_mean = {}
    normalized_e2e_p99 = {}
    ttft_mean = {}
    tbt_max = {}
    scheduling_delay_mean = {}
    scheduling_delay_p99 = {}
    max_qps_data = {}
    stat_files = os.listdir(os.getcwd() + "/" + stats_dir)
    for stat_file in stat_files:
        if stat_file.endswith(".csv"):
            with open(stats_dir + "/" + stat_file, "r") as f:
                data_frame = data_filter(pd.read_csv(f), data_filtering_condition)
                for index, row in data_frame.iterrows():
                    label = label_getter(row)
                    ttft_cdfs[label] = row["ttft_cdf"]
                    tbt_cdfs[label] = row["tbt_cdf"]
                    batch_size_cdfs[label] = row["batch_size_cdf"]
                    scheduling_delay_cdfs[label] = row["scheduling_delay_cdf"]

                    normalized_e2e_time_mean[label] = row["request_e2e_time_normalized_mean"]
                    normalized_e2e_p99[label] = row["request_e2e_time_normalized_99%"]

                    ttft_mean[label] = row["ttft_mean"]
                    tbt_max[label] = row["tbt_max"]
                    scheduling_delay_mean[label] = row["request_scheduling_delay_mean"]
                    scheduling_delay_p99[label] = row["request_scheduling_delay_99%"]
        elif stat_file.endswith('out'):
            with open(stats_dir + "/" + stat_file, "r") as f:
                for line in f.readlines():
                    if MAX_QPS_UNDER_SLO_MARKER in line:
                        sub_str = line.split(MAX_QPS_UNDER_SLO_MARKER)[1]
                        s = sub_str.split(", ")
                        tp = int(s[4].split(": ")[1])
                        pp = int(s[5].split(": ")[1])
                        if data_filtering_condition["cluster_config"]["replica_config"]["tensor_parallel_size"] == tp and \
                                data_filtering_condition["cluster_config"]["replica_config"]["num_pipeline_stages"] == pp:
                            global_scheduler = s[9].split(": ")[1].strip()
                            max_qps = s[9].split(": ")[-1]
                            max_qps = float(max_qps)
                            max_qps_data[global_scheduler] = max_qps
    new_max_qps_data = {}
    for key, value in scheduling_delay_mean.items():
        if key in max_qps_data:
            new_max_qps_data[key] = max_qps_data[key]

    plot_cdf(stats_dir, ttft_cdfs, "TTFT")
    plot_cdf(stats_dir, tbt_cdfs, "TBT")
    plot_cdf(stats_dir, batch_size_cdfs, "Batch Size")
    plot_cdf(stats_dir, scheduling_delay_cdfs, "Scheduling Delay")

    plot_bar_chart(stats_dir, ttft_mean, "TTFT mean")
    plot_bar_chart(stats_dir, tbt_max, "TBT max")
    plot_bar_chart(stats_dir, scheduling_delay_mean, "Scheduling Delay mean")
    plot_bar_chart(stats_dir, scheduling_delay_p99, "Scheduling Delay p99")

    plot_bar_chart(stats_dir, normalized_e2e_time_mean, "Normalized E2E Time mean")
    plot_bar_chart(stats_dir, normalized_e2e_p99, "Normalized E2E Time p99")

    if new_max_qps_data:
        plot_bar_chart(stats_dir, new_max_qps_data, "Max QPS under SLO", print_stats=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=str, required=True)
    parser.add_argument("--qps", type=int, required=True)
    parser.add_argument("--num_replicas", type=int, required=True)
    parser.add_argument("--device", type=str, required=False, default="a100")
    parser.add_argument("--tensor-parallel-size", type=int, required=False, default=1)
    parser.add_argument("--num-pipeline-stages", type=int, required=False, default=1)
    args = parser.parse_args()

    data_filtering_condition = {
        "poisson_request_interval_generator_qps": args.qps,
        "cluster_config": {
            "num_replicas": args.num_replicas,
            "replica_config": {
                "device": args.device,
                "tensor_parallel_size": args.tensor_parallel_size,
                "num_pipeline_stages": args.num_pipeline_stages
            },
        }
    }

    label_getter = generate_label_from_global_scheduler_config

    process_stats(args.stats_dir, data_filtering_condition, label_getter)


if __name__ == "__main__":
    main()
