import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly_express as px


def plot_bar_chart(stats_dir, stats, metric_name):
    plt.figure()
    target_dir = f"{stats_dir}/bar_charts"
    os.makedirs(target_dir, exist_ok=True)
    for key, value in stats.items():
        plt.bar(key, value)

    plt.xlabel("Configuration")
    plt.ylabel(metric_name)
    plt.title(metric_name + " Bar Chart")
    plt.savefig(f"{target_dir}/{metric_name}_bar_chart.png")


def parse_cdfs(cdf_str):
    cdf_str = cdf_str.replace("[", "").replace("]", "").replace(" ", "")
    cdfs = cdf_str.split(",")
    return [float(cdf) for cdf in cdfs]


def plot_cdf(stats_dir, cdfs, metric_name):
    plt.figure()
    target_dir = f"{stats_dir}/cdf_plots"
    os.makedirs(target_dir, exist_ok=True)
    for key, cdf in cdfs.items():
        values = parse_cdfs(cdf[0])
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


def process_stats(stats_dir):
    ttft_cdfs = {}
    tbt_cdfs = {}
    batch_size_cdfs = {}
    scheduling_delay_cdfs = {}



    normalized_e2e_time_mean = {}
    normalized_e2e_p99 = {}
    ttft_mean = {}
    tbt_max = {}
    scheduling_delay_mean = {}

    stat_files = os.listdir(stats_dir)
    for stat_file in stat_files:
        if not stat_file.endswith(".csv"):
            continue
        with open(stats_dir + "/" + stat_file, "r") as f:
            data_frame = pd.read_csv(f)
            label = "_".join(stat_file.split("_")[:-1])
            ttft_cdfs[label] = data_frame["ttft_cdf"]
            tbt_cdfs[label] = data_frame["tbt_cdf"]
            batch_size_cdfs[label] = data_frame["batch_size_cdf"]
            scheduling_delay_cdfs[label] = data_frame["scheduling_delay_cdf"]

            normalized_e2e_time_mean[label] = data_frame["request_e2e_time_normalized_mean"]
            normalized_e2e_p99[label] = data_frame["request_e2e_time_normalized_99%"]

            ttft_mean[label] = data_frame["ttft_mean"]
            tbt_max[label] = data_frame["tbt_max"]
            scheduling_delay_mean[label] = data_frame["request_scheduling_delay_mean"]

    plot_cdf(stats_dir, ttft_cdfs, "TTFT")
    plot_cdf(stats_dir, tbt_cdfs, "TBT")
    plot_cdf(stats_dir, batch_size_cdfs, "Batch Size")
    plot_cdf(stats_dir, scheduling_delay_cdfs, "Scheduling Delay")

    plot_bar_chart(stats_dir, ttft_mean, "TTFT mean")
    plot_bar_chart(stats_dir, tbt_max, "TBT max")
    plot_bar_chart(stats_dir, scheduling_delay_mean, "Scheduling Delay mean")

    plot_bar_chart(stats_dir, normalized_e2e_time_mean, "Normalized E2E Time mean")
    plot_bar_chart(stats_dir, normalized_e2e_p99, "Normalized E2E Time p99")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=str, required=True)
    args = parser.parse_args()

    process_stats(args.stats_dir)


if __name__ == "__main__":
    main()
