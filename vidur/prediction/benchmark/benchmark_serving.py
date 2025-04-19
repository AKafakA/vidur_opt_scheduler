# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import functools

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import aiohttp
import argparse
import asyncio
import json
import os
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import jsonlines
from scipy.stats import zipf
from enum import Enum
from transformers import AutoTokenizer
from typing import List
import resource

resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

num_finished_requests = 0
server_num_requests = {}


def get_wait_time(qps: float, distribution: str, burstiness: float = 1.0) -> float:
    mean_time_between_requests = 1.0 / qps
    if distribution == "uniform":
        return mean_time_between_requests
    elif distribution == "gamma":
        assert burstiness > 0, (
            f"A positive burstiness factor is expected, but given {burstiness}.")
        theta = 1.0 / (qps * burstiness)
        return np.random.gamma(shape=burstiness, scale=theta)
    else:
        return np.random.exponential(mean_time_between_requests)


def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return


async def async_request_gen(generator, qps: float, distribution="uniform", burstiness: float = 0.0):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                await asyncio.sleep(get_wait_time(qps, distribution, burstiness))
        except StopIteration:
            return


class GenerationBackend(str, Enum):
    vLLM = "vLLM"
    block = "block"
    llumnix = "llumnix"


async def query_model_block(prompt, verbose, ip_ports):
    prompt, prompt_len, max_response_len, estimated_response_len, request_id = prompt
    global server_num_requests
    global_scheduler_ip_port = ip_ports[0]
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    global num_finished_requests

    request_dict = {
        "request_id": request_id,
        "prompt": prompt,
        "max_response_len": max_response_len,
        "predicted_response_len": estimated_response_len,
        "prompt_len": prompt_len,
    }

    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        if verbose:
            print('Querying model')
        try:
            async with session.post(f'http://{global_scheduler_ip_port}/generate_benchmark', json=request_dict,
                                    ssl=False) as resp:
                if verbose:
                    print('Done')

                output = await resp.json()
                num_finished_requests += 1
                if 'per_token_latency' in output:
                    output['response_len'] = len(output['per_token_latency'])
                elif 'generated_text' in output:
                    output['response_len'] = len(output['generated_text'].split())
                else:
                    output['response_len'] = 0
                print("num_finised_requests: {}".format(num_finished_requests))
                return prompt, output
        except aiohttp.ClientError as e:
            print(f"Connect to {global_scheduler_ip_port} failed with: {str(e)}")
            sys.exit(1)


async def query_model_vllm(prompt, verbose, ip_ports, with_request_id=True):
    prompt, prompt_len, max_response_len, _, request_id = prompt

    # Evenly dispatch request to the given api servers.
    global server_num_requests
    server_id = min(server_num_requests, key=server_num_requests.get)
    server_num_requests[server_id] += 1
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    global num_finished_requests

    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        best_of = 1
        use_beam_search = False
        output_len = max_response_len
        request_dict = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "max_tokens": output_len,
            "top_k": 1,
            "ignore_eos": True,
            "stream": False,
        }
        if with_request_id:
            request_dict["request_id"] = request_id

        if verbose:
            print('Querying model')
        try:
            async with session.post(f'http://{ip_ports[server_id]}/generate_benchmark', json=request_dict,
                                    ssl=False) as resp:
                if verbose:
                    print('Done')

                output = await resp.json()
                # necessary for latency calc
                if 'per_token_latency' in output:
                    output['response_len'] = len(output['per_token_latency'])
                elif 'generated_text' in output:
                    output['response_len'] = len(output['generated_text'].split())
                else:
                    output['response_len'] = 0
                if verbose and 'generated_text' in output:
                    print(json.dumps(output['generated_text']))
                num_finished_requests += 1
                print("num_finised_requests: {}".format(num_finished_requests))
                return (prompt, output)
        except aiohttp.ClientError as e:
            print(f"Connect to {ip_ports[server_id]} failed with: {str(e)}")
            sys.exit(1)


def load_prompts(prompt_file):
    with open(prompt_file) as f:
        prompts = [json.loads(l) for l in f.readlines()]
    return prompts


def get_tok_id_lens(tokenizer, batch):
    tokenized = tokenizer.batch_encode_plus(batch)
    lens = [len(s) for s in tokenized['input_ids']]
    return lens


def calculate_throughput(queries,
                         dur_s,
                         backend,
                         tokenizer,
                         mean_token_latency,
                         mean_e2e_latency,
                         all_e2e_latencies,
                         all_per_token_latencies,
                         all_inference_latencies,
                         all_request_ids,
                         all_decode_token_latencies,
                         all_request_lens,
                         all_waiting_latencies,
                         global_scheduling_overhead,
                         log_latencies,
                         fail_on_response_failure):
    # either should be provided
    if backend == GenerationBackend.block:
        assert all_inference_latencies or all_waiting_latencies
    else:
        all_waiting_latencies = [-1] * len(all_e2e_latencies)
        all_inference_latencies = [-1] * len(all_e2e_latencies)
        global_scheduling_overhead = [-1] * len(all_e2e_latencies)

    prompts = []
    responses = []
    naive_hf_lens = []
    ft_lens = []
    expected_response_lens = []
    ray_gen_lens = []
    cf_gen_lens = []
    for prompt, response in queries:
        if 'generated_text' in response:
            prompts.append(prompt)
            responses.append(response['generated_text'])
        if 'naive_hf_lens' in response:
            naive_hf_lens.append(response['naive_hf_lens'])
        if 'ray_gen_len' in response:
            ray_gen_lens.append(response['ray_gen_len'])
        if 'num_output_tokens_cf' in response:
            cf_gen_lens.append(response['num_output_tokens_cf'])
        if 'response_len' in response:
            expected_response_lens.append(response['response_len'])
    prompt_ids = [p for p in tokenizer.batch_encode_plus(prompts)['input_ids']]
    response_ids = [r for r in tokenizer.batch_encode_plus(responses)['input_ids']]

    # print(f'check_len actual {list(sorted(len(response) for response in response_ids))}')
    # print(f'check_len expect {list(sorted(expected_response_lens))}')
    # print(f'self-reported {list(sorted(cf_gen_lens))}')
    # for prompt, response, expected_response_len in zip(prompt_ids, response_ids, expected_response_lens):
    #     print(f'check lens {len(prompt)=} {len(response)=} {expected_response_len=}')

    try:
        prompt_lens = get_tok_id_lens(tokenizer, prompts)
        response_lens = get_tok_id_lens(tokenizer, responses)
    except Exception:
        print(prompts)
        print(responses)
        raise

    if naive_hf_lens:
        print(f'naive_hf_lens {list(sorted(naive_hf_lens))}')
    print(f'prompt_lens {list(sorted(prompt_lens))}')
    print(f'response_lens {list(sorted(response_lens))}')
    if ray_gen_lens:
        print(f'ray_gen_lens {list(sorted(ray_gen_lens))}')

    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)

    all_prompt_lens = prompt_lens
    all_response_lens = response_lens
    all_total_tokens = [all_prompt_lens[i] + all_response_lens[i] for i in range(len(all_prompt_lens))]
    # if all waiting latencies are not provided, calculate them by e2e - inference
    if not all_waiting_latencies and all_inference_latencies and len(all_inference_latencies) == len(all_e2e_latencies):
        all_waiting_latencies = [all_e2e_latencies[i] - all_inference_latencies[i] for i in
                                 range(len(all_e2e_latencies))]
    elif not all_inference_latencies and all_waiting_latencies and len(all_waiting_latencies) == len(all_e2e_latencies):
        all_inference_latencies = [all_e2e_latencies[i] - all_waiting_latencies[i] for i in
                                   range(len(all_e2e_latencies))]

    def calculate_mean(latencies):
        if latencies:
            return np.mean(latencies)
        else:
            return -1

    mean_waiting_latency = calculate_mean(all_waiting_latencies)
    mean_inference_latency = calculate_mean(all_inference_latencies)
    mean_global_scheduling_overhead = calculate_mean(global_scheduling_overhead)

    if naive_hf_lens:
        # Manually count naive hf tok len
        total_resp_tokens = sum(
            [response_len for _, response_len in naive_hf_lens])
        total_prompt_tokens = sum(
            [prompt_len for prompt_len, _ in naive_hf_lens])
        response_token_count = total_prompt_tokens + total_resp_tokens
    if ray_gen_lens:
        response_token_count = sum(ray_gen_lens)
    if cf_gen_lens:
        response_token_count = sum(cf_gen_lens)

    # print(f'prompt_token_count {prompt_token_count} response_token_count {response_token_count}')
    throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    print(f'throughput_tok_s {throughput_tok_s:.02f}')
    qps = len(responses) / dur_s
    msg1 = f'backend {backend} dur_s {dur_s:.04f} tokens_per_s {throughput_tok_s:.02f} qps {qps:.04f}\n'
    msg2 = f'successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}\n'
    msg3 = (f'{mean_token_latency=:.04f}(ms), {mean_e2e_latency=:.04f}(ms), {mean_inference_latency=:.04f}(ms), '
            f'{mean_waiting_latency=:.04f}(ms), {mean_global_scheduling_overhead=:.04f}(ms) \n')

    msg = msg1 + msg2 + msg3
    if log_latencies:
        msg += f'{all_request_lens=}\n{all_request_ids=}\n'
        msg += f'{all_total_tokens=}\n{all_prompt_lens=}\n{all_response_lens=}\n'
        msg += f'{all_e2e_latencies=}\n{all_per_token_latencies=}\n{all_inference_latencies=}\n{all_waiting_latencies=}\n{all_decode_token_latencies=}\n'
        msg += f'{global_scheduling_overhead=}\n'

    print(msg)

    if fail_on_response_failure:
        assert len(responses) == len(queries), \
            f"{fail_on_response_failure=}, expected number of successful respones to equal number of queries, got {len(responses)} vs {len(queries)}"

    return throughput_tok_s, qps, msg


def calculate_cdf(latencies):
    hist, bin_edges = np.histogram(latencies, bins=50)
    cumsum = np.cumsum(hist)
    print("Latency: ")
    print(f"{bin_edges=}")
    print(f"{hist=}")
    print(f"{cumsum=}")


def plot_latency_cdf(req_latencies, prefill_latencies, decode_latencies, scheduling_overhead, waiting_latency,
                     log_filename, backend, output_dir='.'):
    fig_filename = os.path.splitext(log_filename)[0] + "_latency.png"
    fig, (ax_req, ax_prefill, ax_decode, ax_scheduling_overhead, ax_waiting_latency) = plt.subplots(1, 5,
                                                                                                    figsize=(5 * 7, 6))

    def plot_single(ax, latencies, is_prefill=False):
        hist, bin_edges = np.histogram(latencies, bins=50)
        cumsum = np.cumsum(hist)
        p50 = np.percentile(latencies, 50)
        p80 = np.percentile(latencies, 80)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        p999 = np.percentile(latencies, 99.9)
        ax.plot(bin_edges[1:], cumsum / np.sum(hist) * 100, color='red')
        ax.axvline(p50, color='blue', linestyle='--', label='P50')
        ax.text(p50, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p50:.2f}", va='bottom',
                ha='right', color='blue')
        ax.axvline(p80, color='green', linestyle='--', label='P80')
        ax.text(p80, ax.get_ylim()[0] + 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p80:.2f}", va='bottom',
                ha='right', color='green')
        ax.axvline(p95, color='orange', linestyle='--', label='P95')
        ax.text(p95, ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p95:.2f}", va='bottom',
                ha='right', color='orange')
        ax.axvline(p99, color='purple', linestyle='--', label='P99')
        ax.text(p99, ax.get_ylim()[0] + 0.20 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p99:.2f}", va='bottom',
                ha='right', color='purple')
        ax.axvline(p999, color='gray', linestyle='--', label='P99.9')
        ax.text(p999, ax.get_ylim()[0] + 0.25 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p999:.2f}", va='bottom',
                ha='right', color='gray')
        mean = np.mean(latencies)
        mean_value = bin_edges[:-1][np.where(bin_edges[:-1] <= mean)][-1]
        mean_percentage = cumsum[np.where(bin_edges[:-1] <= mean)][-1] / np.sum(hist) * 100
        ax.axvline(mean_value, color='black', linestyle='-', label='mean={:.2f}'.format(mean))
        ax.text(mean_value, mean_percentage, f"{mean_percentage:.2f}", va='bottom', ha='right', color='black')
        # ax.legend(loc='best')
        ax.set_ylabel('Cumulative Percentage(%)')

    plot_single(ax_req, req_latencies)
    plot_single(ax_prefill, prefill_latencies, is_prefill=True)
    plot_single(ax_decode, decode_latencies)
    if backend == GenerationBackend.block:
        plot_single(ax_scheduling_overhead, scheduling_overhead)
        plot_single(ax_waiting_latency, waiting_latency)
    ax_req.set_xlabel('Latency/req(ms)')
    ax_req.set_title('request cdf')
    ax_prefill.set_xlabel('Latency/token(ms)')
    ax_prefill.set_title('prefill cdf')
    ax_decode.set_xlabel('Latency/token(ms)')
    ax_decode.set_title('decode cdf')
    ax_scheduling_overhead.set_xlabel('Latency/request(ms)')
    ax_scheduling_overhead.set_title('scheduling overhead cdf')
    ax_waiting_latency.set_xlabel('Latency/request(ms)')
    ax_waiting_latency.set_title('time in queue cdf')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    # set the labels
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', ncol=4)
    # save the figure
    fig.savefig(output_dir + '/' + fig_filename)


def plot_len_cdf(prompt_lens, response_lens, total_tokens, log_filename, estimated_length=None, output_dir='.'):
    fig_filename = os.path.splitext(log_filename)[0] + "_len.png"
    if estimated_length:
        fig, (ax_prompt, ax_response, ax_total, ax_estimated) = plt.subplots(1, 4, figsize=(4 * 7, 4.8))
    else:
        fig, (ax_prompt, ax_response, ax_total) = plt.subplots(1, 3, figsize=(3 * 7, 4.8))
        ax_estimated = None

    def plot_single(ax, lens, x_label_str, title_str):
        hist, bin_edges = np.histogram(lens, bins=50)
        cumsum = np.cumsum(hist)
        p50 = np.percentile(lens, 50)
        p80 = np.percentile(lens, 80)
        p95 = np.percentile(lens, 95)
        p99 = np.percentile(lens, 99)
        ax.plot(bin_edges[1:], cumsum / np.sum(hist) * 100, color='red')
        ax.axvline(p50, color='blue', linestyle='--', label='P50')
        ax.text(p50, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p50:.2f}", va='bottom',
                ha='right', color='blue')
        ax.axvline(p80, color='green', linestyle='--', label='P80')
        ax.text(p80, ax.get_ylim()[0] + 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p80:.2f}", va='bottom',
                ha='right', color='green')
        ax.axvline(p95, color='orange', linestyle='--', label='P95')
        ax.text(p95, ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p95:.2f}", va='bottom',
                ha='right', color='orange')
        ax.axvline(p99, color='purple', linestyle='--', label='P99')
        ax.text(p99, ax.get_ylim()[0] + 0.20 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{p99:.2f}", va='bottom',
                ha='right', color='purple')
        mean = np.mean(lens)
        mean_value = bin_edges[:-1][np.where(bin_edges[:-1] <= mean)][-1]
        mean_percentage = cumsum[np.where(bin_edges[:-1] <= mean)][-1] / np.sum(hist) * 100
        ax.axvline(mean_value, color='black', linestyle='-', label='mean={:.2f}'.format(mean))
        ax.text(mean_value, mean_percentage, f"{mean_percentage:.2f}", va='bottom', ha='right', color='black')
        ax.set_xlabel(x_label_str)
        ax.set_ylabel('Cumulative Percentage(%)')
        ax.set_title(title_str)

    plot_single(ax_prompt, prompt_lens, 'prompt len', 'prompt len cdf')
    plot_single(ax_response, response_lens, 'response len', 'response len cdf')
    plot_single(ax_total, total_tokens, 'total token', 'total token cdf')
    if ax_estimated:
        plot_single(ax_estimated, [estimated_length[i] - response_lens[i] for i in range(len(estimated_length))],
                    'Diff between res real/estimated len', 'estimated len cdf')
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    # set the labels
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', ncol=4)
    # save the figure
    fig.savefig(output_dir + "/" + fig_filename)


def plot_sampled_timestamp_metrics(data, log_filename, metric_name, output_dir='.'):
    fig_filename = os.path.splitext(log_filename)[0] + f"_{metric_name}.png"
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'], data['metric'])
    ax.set_xlabel('timestamp (ms)')
    ax.set_ylabel(metric_name)
    plt.suptitle(metric_name, fontsize=6)
    fig.savefig(output_dir + "/" + fig_filename)


def plot_instance(log_filename_0):
    current_dir = os.path.dirname(os.path.abspath(log_filename_0))
    log_files = glob.glob(os.path.join(current_dir, '*.log_instance.csv'))
    log_files.sort(key=os.path.getmtime, reverse=True)
    df_0 = pd.read_csv(log_files[0]).sort_values(by=["timestamp"])
    timestamp_list_0 = df_0["timestamp"].to_numpy()
    num_instances_list_0 = df_0["num_instances"].to_numpy()
    time_0 = 0
    sum_0 = 0
    for idx, t in enumerate(timestamp_list_0):
        if t > time_0:
            time_0 += 1
            sum_0 += num_instances_list_0[idx]
    print(f"{sum_0 / time_0} gpu/s")
    avg_instance_num = np.round(sum_0 / time_0, 2)

    fig, ax = plt.subplots()
    ax.plot(timestamp_list_0, num_instances_list_0, color="red", label=f"instance_num(avg {avg_instance_num} /s)")
    ax.legend(loc='upper left')
    fig_filename = os.path.splitext(log_filename_0)[0] + "_instance.png"
    index1 = fig_filename.rfind('/')
    index2 = fig_filename.rfind('/', 0, index1)
    fig_filename_title = fig_filename[index2 + 1:]
    plt.suptitle(fig_filename_title, fontsize=6)
    fig.savefig(fig_filename)

    return avg_instance_num


def save_all_decode_token_latencies_npy(all_token_latencies: List[np.ndarray], log_filename, output_dir='.'):
    dtype = [('timestamp', float), ('latency', float)]
    all_lat_pairs = []
    for arr in all_token_latencies:
        # use decode latencies
        for pair in arr[1:]:
            all_lat_pairs.append((pair[0], pair[1]))
    all_lat_pairs = np.array(all_lat_pairs, dtype=dtype)
    all_lat_pairs = np.sort(all_lat_pairs, order='timestamp')
    np.save(output_dir + '/' + os.path.splitext(log_filename)[0], all_lat_pairs)


class MeasureLatency:
    def __init__(self):
        self._request_ids = []
        self._request_lens = []
        self._request_latencies = []
        self._per_token_latencies = []
        self._decode_token_latencies = []
        self._prefill_token_latencies = []
        self._all_token_latencies = []
        self._decode_sum_latencies = []
        self._all_decode_token_latencies = []
        self._inference_latencies = []
        self._waiting_latencies = []
        self._engine_ttft = []
        self._global_scheduling_overhead = []
        self._avg_gpu_blocks = []
        self._avg_num_waiting_requests = []
        self._var_gpu_blocks = []
        self._var_num_waiting_requests = []
        self._requested_timestamps = []
        self._num_preempted = []

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            prompt, output = await f(*args, **kwargs)
            # Do not record latency if request failed.
            latency = (time.time() - start) * 1000
            if 'generated_text' in output:
                self._request_latencies.append(latency)
                try:
                    self._per_token_latencies.append(
                        latency / output['response_len'])
                except ZeroDivisionError:
                    # Not currently using this metric..
                    pass
            client_ttft = -1.0
            engine_ttft = -1.0
            if 'request_id' in output:
                self._request_ids.append(output['request_id'])
            if 'per_token_latency' in output:
                lat_arr = np.array(output['per_token_latency'])
                mean_decode_token_latency = 0 if len(lat_arr) == 1 else np.mean(lat_arr[1:, 1])
                decode_sum_latency = 0 if len(lat_arr) == 1 else np.sum(lat_arr[1:, 1])
                self._decode_token_latencies.append(mean_decode_token_latency)
                self._request_lens.append(len(lat_arr[1:, 1]))
                self._all_token_latencies.append(lat_arr)
                self._decode_sum_latencies.append(decode_sum_latency)
                self._all_decode_token_latencies.extend(lat_arr[1:, 1])
                self._prefill_token_latencies.append(lat_arr[0][1])
                if 'time_on_probe' in output:
                    self._global_scheduling_overhead.append(output['time_on_probe'])
                else:
                    if 'time_on_backend' in output:
                        time_on_backend = output['time_on_backend']
                    else:
                        start_time_on_backend = lat_arr[0][0] - lat_arr[0][1] / 1000
                        time_on_backend = (lat_arr[-1][0] - start_time_on_backend) * 1000
                    self._global_scheduling_overhead.append(latency - time_on_backend)
            if 'per_token_latency_breakdown_dict' in output:
                self._inference_latencies.append(
                    np.mean(output['per_token_latency_breakdown_dict']['step_latency_engine']))
            else:
                if 'inference_latency' in output:
                    self._inference_latencies.append(output['inference_latency'])
            if 'waiting_latency' in output:
                self._waiting_latencies.append(output['waiting_latency'])
            if 'ttft' in output:
                self._engine_ttft.append(output['ttft'])
                engine_ttft = output['ttft']
            record_timestamp = False
            if 'sampled_avg_gpu_blocks' in output:
                self._avg_gpu_blocks.append(output['sampled_avg_gpu_blocks'])
                self._var_gpu_blocks.append(output['sampled_var_gpu_blocks'])
                record_timestamp = True
            if 'sampled_avg_n_request' in output:
                self._avg_num_waiting_requests.append(output['sampled_avg_n_request'])
                self._var_num_waiting_requests.append(output['sampled_var_n_request'])
                record_timestamp = True
            if 'num_preempted' in output:
                self._num_preempted.append(output['num_preempted'])
            if record_timestamp:
                self._requested_timestamps.append(start)
            return prompt, output

        return measured


def get_token_ids(input_str, tokenizer):
    t = tokenizer(input_str)
    return t['input_ids']


async def benchmark(
        backend: GenerationBackend,
        tokenizer,
        prompts: List[str],
        verbose: bool,
        log_filename: str,
        ip_ports: List[int],
        distribution: str,
        qps: float,
        burstiness: float,
        log_latencies: bool,
        fail_on_response_failure: bool,
        output_gen_lens: bool = False,
        output_dir: str = '.'
):
    if backend == GenerationBackend.vLLM:
        query_model = query_model_vllm
    elif backend == GenerationBackend.block:
        query_model = query_model_block
        assert len(ip_ports) == 1
    elif backend == GenerationBackend.llumnix:
        query_model = functools.partial(query_model_vllm, with_request_id=False)
    else:
        raise ValueError(f'unknown backend {backend}')

    global server_num_requests
    num_servers = len(ip_ports)
    for server_id in range(num_servers):
        server_num_requests[server_id] = 0

    m = MeasureLatency()

    query_model = m.measure(query_model)

    if distribution == "burst":
        qps = float('inf')

    print(
        f'Starting with backend={backend}, num_prompts={len(prompts)}')
    print(f'traffic distribution={distribution}, qps={qps}, burstiness={burstiness}')

    total_requests = len(prompts)

    async_prompts = async_request_gen(
        iter(prompts), qps=qps, distribution=distribution, burstiness=burstiness)

    start_time = time.time()
    tasks = []
    async for prompt in async_prompts:
        tasks.append(asyncio.create_task(query_model(prompt, verbose, ip_ports)))
    queries = await asyncio.gather(*tasks)
    dur_s = time.time() - start_time
    mean_token_latency = np.mean(m._per_token_latencies)
    mean_e2e_latency = np.mean(m._request_latencies)

    sampled_prompts = []
    sampled_responses = []
    sampled_responses_length = []
    if output_gen_lens:
        for prompt, output in queries:
            if 'generated_text' in output:
                sampled_prompts.append(prompt)
                sampled_responses.append(output['generated_text'])
                sampled_responses_length = get_tok_id_lens(tokenizer, sampled_responses)

    throughput, actual_qps, msg = calculate_throughput(queries,
                                                       dur_s,
                                                       backend,
                                                       tokenizer,
                                                       mean_token_latency,
                                                       mean_e2e_latency,
                                                       m._request_latencies,
                                                       m._per_token_latencies,
                                                       m._inference_latencies,
                                                       m._request_ids,
                                                       m._decode_token_latencies,
                                                       m._request_lens,
                                                       m._waiting_latencies,
                                                       m._global_scheduling_overhead,
                                                       log_latencies,
                                                       fail_on_response_failure)
    calculate_cdf(m._request_latencies)
    plot_latency_cdf(m._request_latencies, m._prefill_token_latencies, m._decode_token_latencies, m._waiting_latencies,
                     m._global_scheduling_overhead, log_filename, backend=backend, output_dir=output_dir)
    save_all_decode_token_latencies_npy(m._all_token_latencies, log_filename, output_dir=output_dir)
    timestamps = [int((x - start_time) * 1000) for x in m._requested_timestamps]
    if timestamps:
        # data = {'timestamp': m._requested_timestamps, 'metric': m._avg_gpu_blocks}
        # plot_sampled_timestamp_metrics(data, log_filename, "avg_gpu_blocks")
        # data = {'timestamp': m._requested_timestamps, 'metric': m._avg_num_waiting_requests}
        # plot_sampled_timestamp_metrics(data, log_filename, "avg_num_waiting_requests")
        if m._avg_gpu_blocks:
            data = {'timestamp': timestamps, 'metric': m._avg_gpu_blocks}
            plot_sampled_timestamp_metrics(data, log_filename, "avg_gpu_blocks", output_dir)
        if m._avg_num_waiting_requests:
            data = {'timestamp': timestamps, 'metric': m._avg_num_waiting_requests}
            plot_sampled_timestamp_metrics(data, log_filename, "avg_num_waiting_requests", output_dir=output_dir)
        if m._var_gpu_blocks:
            data = {'timestamp': timestamps, 'metric': m._var_gpu_blocks}
            plot_sampled_timestamp_metrics(data, log_filename, "var_gpu_blocks", output_dir)
        if m._var_num_waiting_requests:
            data = {'timestamp': timestamps, 'metric': m._var_num_waiting_requests}
            plot_sampled_timestamp_metrics(data, log_filename, "var_num_waiting_requests", output_dir)
        if m._num_preempted:
            data = {'timestamp': timestamps, 'metric': m._num_preempted}
            plot_sampled_timestamp_metrics(data, log_filename, "num_preempted", output_dir)

    # avg_instance_num = plot_instance(log_filename)
    avg_instance_num = 0.0
    return throughput, \
        actual_qps, \
        m._prefill_token_latencies, \
        m._decode_token_latencies, \
        m._inference_latencies, \
        avg_instance_num, \
        m._request_latencies, \
        m._request_ids, \
        m._decode_sum_latencies, \
        m._request_lens, \
        m._all_decode_token_latencies, \
        m._waiting_latencies, \
        m._global_scheduling_overhead, \
        sampled_prompts, \
        sampled_responses, \
        sampled_responses_length, \
        m._avg_gpu_blocks, \
        m._var_gpu_blocks, \
        m._avg_num_waiting_requests, \
        m._var_num_waiting_requests, \
        m._num_preempted, \
        timestamps, \
        msg


def gen_random_response_lens(distribution: str, len_mean, len_range, num_prompts):
    if distribution == 'uniform':
        if len_range == 0:
            return [len_mean for _ in range(num_prompts)]

        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        response_lens = list(
            map(lambda _: random.randint(low, high), range(num_prompts)))
    elif distribution == 'exponential':
        response_lens = [min(round(s), len_range) for s in np.random.exponential(scale=len_mean, size=num_prompts)]
    elif distribution == 'capped_exponential':
        response_lens = []
        while len(response_lens) < num_prompts:
            sample = round(np.random.exponential(scale=len_mean))
            if len_range >= sample >= 1:
                response_lens.append(sample)
    elif distribution == 'zipf':
        rank = np.arange(1, len_mean * 2)
        if len_mean == 1024 and len_range == 6144:
            alpha = 1.0005
        elif len_mean == 512 and len_range == 6144:
            alpha = 1.15
        elif len_mean == 256 and len_range == 6144:
            alpha = 1.5
        elif len_mean == 128 and len_range == 6144:
            alpha = 2.0
        else:
            alpha = 1.0
        probabilities = zipf.pmf(rank, alpha)
        probabilities /= np.sum(probabilities)
        response_lens = np.random.choice(np.arange(1, len_mean * 2), size=num_prompts, p=probabilities)
    else:
        raise ValueError(f'unknown distribution {distribution=}')

    scaling_factor = len_mean / np.mean(response_lens)
    response_lens = np.ceil(np.array(response_lens) * scaling_factor).astype(int)
    if distribution == 'zipf':
        response_lens = [response_len if response_len <= len_range else len_range for response_len in response_lens]
    elif distribution == 'uniform':
        capped_response_lens = []
        for response_len in response_lens:
            if response_len < low:
                capped_response_lens.append(low)
            elif response_len > high:
                capped_response_lens.append(high)
            else:
                capped_response_lens.append(response_len)
        response_lens = capped_response_lens
    else:
        response_lens = [response_len if response_len <= len_range else len_range for response_len in response_lens]
    response_lens = [int(x) for x in response_lens]

    return response_lens


def get_dataset_list(dataset_path: str, start_idx: int = 0, num_samples: int = 10):
    dataset_list = []
    for file in os.listdir(dataset_path):
        path = os.path.join(dataset_path, file)
        if path.endswith('.jsonl'):
            with open(path) as f:
                for line in f:
                    dataset_list.append(json.loads(line))
        elif path.endswith('.json'):
            with open(path) as f:
                dataset_list.extend(json.load(f))
        elif path.endswith('.parquet'):
            dataset_list.extend(pd.read_parquet(path).to_dict(orient='records'))
    end_idx = min(len(dataset_list), start_idx + num_samples)
    if start_idx >= len(dataset_list):
        raise ValueError(f"start_idx {start_idx} is out of range for dataset with {len(dataset_list)} samples.")
    return dataset_list[start_idx:end_idx]


def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        max_seqlen: int,
        use_estimated_response_lens: bool,
        start_idx: int,
        task: str = 'chat'
):
    prompts = []
    prompt_lens = []
    max_response_lens = []
    estimated_response_lens = []

    # Load the dataset.
    dataset = get_dataset_list(dataset_path, start_idx, num_requests)

    # Filter dataset for conversation mode
    if task == 'chat':
        dataset = [
            data for data in dataset
            if (
                    ("conversations" in data and len(data["conversations"]) >= 2) or
                    ("conversation" in data and len(data["conversation"]) >= 2)
            )
        ]

    random.shuffle(dataset)

    for data in dataset:
        if task == 'chat':
            if "conversations" in data:
                prompt = data["conversations"][0]["value"]
                res = data["conversations"][1]["value"]
            elif "conversation" in data:
                prompt = data["conversation"][0]["content"]
                res = data["conversation"][1]["content"]
            else:
                raise ValueError(f"Unknown dataset format: {data.keys()}")
        elif task == 'arxiv':
            prompt = "Summarize this paper: " + data["article"]
            res = data["abstract"]
            print(f"response: {res}")
        else:
            raise ValueError(f"Unknown task: {task}")

        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(res).input_ids

        if (len(prompt_token_ids) > 0 and len(completion_token_ids) > 0
                and max_seqlen >= len(prompt_token_ids) + len(completion_token_ids)):
            prompts.append(prompt)
            prompt_lens.append(len(prompt_token_ids))
            max_response_lens.append(len(completion_token_ids))
            estimated_response_lens.append(
                int(data.get("predicted_length", len(completion_token_ids)))
                if use_estimated_response_lens else len(completion_token_ids)
            )

        if len(prompts) > num_requests:
            break

    sampled_ids = random.sample(range(len(prompts)), min(num_requests, len(prompts)))
    sampled_prompts = [prompts[idx] for idx in sampled_ids]
    sampled_prompt_lens = [prompt_lens[idx] for idx in sampled_ids]
    sampled_response_lens = [max_response_lens[idx] for idx in sampled_ids]
    sampled_estimated_response_lens = [estimated_response_lens[idx] for idx in sampled_ids]

    return sampled_prompts, sampled_prompt_lens, sampled_response_lens, sampled_estimated_response_lens


def generate_lens_files(
        length_output_file,
        prompt_lens,
        response_lens):
    import csv
    assert length_output_file.endswith('.csv')
    with open(length_output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['num_prefill_tokens', 'num_decode_tokens', 'num_total_tokens', 'pd_ratio'])
        for prompt_len, response_len in zip(prompt_lens, response_lens):
            writer.writerow([prompt_len, response_len, prompt_len + response_len, (response_len * 1.0) / prompt_len])
    print(f"CSV files for length information saved to {length_output_file}")


def tag_dataset_with_real_response(
        start_id: int,
        prompts,
        responses,
        new_dataset_path: str):
    data = []
    record_id = start_id
    filtered_count = 0
    for prompt, response in zip(prompts, responses):
        if response.replace(' ', ''):
            record = {'id': record_id,
                      'conversations': [{'from': 'human', 'value': prompt}, {'from': 'model', 'value': response}]}
            data.append(record)
            record_id += 1
        else:
            filtered_count += 1
    if new_dataset_path.endswith('.jsonl'):
        with jsonlines.open(new_dataset_path, 'w') as writer:
            writer.write_all(data)
    elif new_dataset_path.endswith('.json'):
        with open(new_dataset_path, 'w') as fp:
            json.dump(data, fp)
    print(f"Dataset with real responses saved to {new_dataset_path} and tagged with {len(data)} records and "
          f" filtered out {filtered_count} empty requests.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument('--trust_remote_code',
                        action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--backend', type=GenerationBackend,
                        choices=[e.name for e in GenerationBackend], default='vLLM')
    parser.add_argument('--log_filename', type=str, default='benchmark.log')
    parser.add_argument('--ip_ports', nargs='+', required=True, help='List of ip:port')
    parser.add_argument('--num_sampled_requests', type=int, default=10)
    parser.add_argument('--data_start_index', type=int, default=0,
                        help="Start index of the dataset to sample from.")
    parser.add_argument('--max_request_len', type=int, default=8192)
    parser.add_argument(
        '--distribution', choices=["uniform", "gamma", "exponential"], default="gamma")
    parser.add_argument('--qps', type=float, default=4.0)
    parser.add_argument('--burstiness', type=float, default=1.0)
    parser.add_argument('--log_latencies', action="store_true",
                        help="Whether or not to write all latencies to the log file.")
    parser.add_argument('--fail_on_response_failure', type=bool, default=False,
                        help="Whether or not to fail the benchmarking script if any request fails")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_type', type=str, choices=['sharegpt', 'arxiv', 'lmsys'], default='sharegpt')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_path', type=str)
    parser.add_argument('--print_generation_lens_and_exit',
                        action='store_true')
    parser.add_argument('--tag_dataset_with_real_response',
                        type=bool, default=False)
    parser.add_argument('--enable_csv_files', type=bool, default=False)
    parser.add_argument('--keep_all_metrics', type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument("--use_estimated_response_lens", type=bool, default=False)

    args = parser.parse_args()

    args.output_dir = "experiment_output/" + args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    backend = GenerationBackend[args.backend]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)

    random.seed(0xCADE)
    np.random.seed(0xCADE)
    if args.dataset_type == "sharegpt" or args.dataset_type == "lmsys":
        prompts, prompt_lens, max_response_lens, estimated_response_lens = sample_requests(
            args.dataset_path,
            args.num_sampled_requests,
            tokenizer,
            args.max_request_len,
            args.use_estimated_response_lens,
            args.data_start_index,
            task='chat'
        )
    elif args.dataset_type == "arxiv":
        prompts, prompt_lens, max_response_lens, estimated_response_lens = sample_requests(
            args.dataset_path,
            args.num_sampled_requests,
            tokenizer,
            args.max_request_len,
            args.use_estimated_response_lens,
            args.data_start_index,
            task='arxiv'
        )
    else:
        raise ValueError("unknown dataset type")

    for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, max_response_lens)):
        total = prompt_len + gen_len
        if total > args.max_request_len:
            print(f'truncating long prompt+gen_len {prompt_len=} {gen_len=}')
            gen_len = args.max_request_len - prompt_len
        max_response_lens[i] = gen_len

    if args.print_generation_lens_and_exit:
        print(f'{prompt_lens=}')
        print(f'{max_response_lens=}')
        print('Exiting...')
        return

    if args.verbose or True:
        print('prompt lens', sorted(list(prompt_lens)))
        print('response lens', sorted(list(max_response_lens)))
        total_tokens = []
        for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, max_response_lens)):
            total_tokens.append(prompt_len + gen_len)
        print('total tokens', sorted(list(total_tokens)))

    prompts = list(zip(prompts, prompt_lens, max_response_lens, estimated_response_lens, range(len(prompt_lens))))

    (throughput,
     actual_qps,
     prefill_token_latencies,
     decode_token_latencies,
     inference_latencies,
     avg_instance_num,
     request_latencies, request_ids,
     decode_sum_latencies, request_lens,
     all_decode_token_latencies,
     waiting_latency,
     scheduling_overhead,
     sampled_prompts,
     sampled_responses,
     sampled_responses_length,
     avg_gpu_blocks, var_gpu_blocks,
     avg_num_waiting_requests,
     var_num_waiting_requests,
     num_preempted,
     request_timestamps,
     messages) = asyncio.run(benchmark(
        backend,
        tokenizer,
        prompts,
        args.verbose,
        args.log_filename,
        args.ip_ports,
        args.distribution,
        args.qps,
        args.burstiness,
        args.log_latencies,
        args.fail_on_response_failure,
        args.tag_dataset_with_real_response or args.enable_csv_files,
        args.output_dir
    )
    )

    with open(args.output_dir + '/' + os.path.splitext(args.log_filename)[0] + "_logs.txt", 'w') as f:
        f.write(messages)

    if args.tag_dataset_with_real_response or args.enable_csv_files:
        assert sampled_responses_length
        # dataset_path is the path to the dataset directory
        generated_dataset_path = args.dataset_path + "/" + "generate"
        if not os.path.exists(generated_dataset_path):
            os.makedirs(generated_dataset_path)

        if args.tag_dataset_with_real_response:
            generated_dataset_files = [file for file in os.listdir(generated_dataset_path)
                                       if file.endswith('with_real_response.json')]
            tagged_dataset_path = os.path.join(generated_dataset_path,
                                               f'{args.dataset_type}_{args.num_sampled_requests}_'
                                               f'{len(generated_dataset_files) + 1}'
                                               f'_with_real_response.json')
            tag_dataset_with_real_response(
                args.data_start_index, sampled_prompts, sampled_responses, tagged_dataset_path)
        if args.enable_csv_files:
            # csv_file_name = os.path.join(args.dataset_path,
            #                              f'{args.dataset_type}_{args.num_sampled_requests}_lens.csv')
            generated_csv_files = [file for file in os.listdir(generated_dataset_path)
                                    if file.endswith('lens.csv')]
            csv_file_name = os.path.join(generated_dataset_path,
                                         f'{args.dataset_type}_{args.num_sampled_requests}_'
                                         f'{len(generated_csv_files) + 1}'
                                         f'_lens.csv')
            generate_lens_files(csv_file_name, prompt_lens, sampled_responses_length)

    if args.keep_all_metrics:
        plot_len_cdf(prompt_lens, max_response_lens, total_tokens, args.log_filename,
                     estimated_length=estimated_response_lens,
                     output_dir=args.output_dir)
        data = {
            "Throughput": np.float32(throughput),
            "prefill_token_latencies": np.array(prefill_token_latencies),
            "decode_token_latencies": np.array(decode_token_latencies),
            "decode_sum_latencies": np.array(decode_sum_latencies),
            "inference_latencies": np.array(inference_latencies),
            "request_latencies": np.array(request_latencies),
            "waiting_latency": np.array(waiting_latency),
            "scheduling_overhead": np.array(scheduling_overhead),
            "actual_qps": np.float32(actual_qps),
            "avg_gpu_blocks": np.array(avg_gpu_blocks),
            "var_gpu_blocks": np.array(var_gpu_blocks),
            "avg_num_waiting_requests": np.array(avg_num_waiting_requests),
            "var_num_waiting_requests": np.array(var_num_waiting_requests),
            "num_preempted": np.array(num_preempted),
            "request_timestamps_in_ms": np.array(request_timestamps),
        }
        np.savez(args.output_dir + '/' + os.path.splitext(args.log_filename)[0] + f"_all_metrics.npz", **data)

        results = []
        file_name = args.output_dir + '/' + os.path.splitext(args.log_filename)[0] + "_latency_info.json"
        try:
            with open(file_name, 'r') as f:
                results = json.load(f)
        except json.decoder.JSONDecodeError:
            pass
        except FileNotFoundError:
            os.mknod(file_name)
        with open(file_name, 'w') as f:
            results.append({"qps": args.qps,
                            "burstiness": args.burstiness,
                            "request_ids": request_ids,
                            "request_lens": request_lens,
                            "request_latencies": request_latencies,
                            "prefill_token_latencies": prefill_token_latencies,
                            "decode_token_latencies": decode_token_latencies,
                            "decode_sum_latencies": decode_sum_latencies,
                            "all_decode_token_latencies": all_decode_token_latencies,
                            "inference_latencies": inference_latencies,
                            "scheduling_overhead": scheduling_overhead,
                            "throughput": throughput,
                            "instance_num": avg_instance_num})
            json.dump(results, f)


if __name__ == '__main__':
    main()
