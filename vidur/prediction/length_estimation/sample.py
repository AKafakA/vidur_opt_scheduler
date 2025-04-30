import argparse
import os
import random

import numpy as np
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/sharegpt/generate")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--val-samples", type=int, default=10000)
    parser.add_argument("--shuffle", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    seed = 42
    utils.set_seed(seed)
    args = parse_args()
    data_path = args.data_path
    data = []
    for file in os.listdir(data_path):
        if file.endswith(".json"):
            file_data_path = os.path.join(data_path, file)
            data += utils.jload(file_data_path)
    print(f"data size: {len(data)}")
    if args.shuffle:
        random.shuffle(data)
    N = args.num_samples
    if N > len(data):
        N = len(data)
    data_mask = np.random.choice(len(data), N, replace=False)
    sampled_data = [data[i] for i in data_mask]
    print(f"sampled data size: {len(sampled_data)}")
    val_size = args.val_samples
    if val_size > len(sampled_data):
        raise ValueError("val size is larger than sampled data size")
    train_size = len(sampled_data) - val_size
    data_train = sampled_data[:train_size]
    data_train = utils.jsort(data_train, key="id", integer=True)
    data_val = sampled_data[train_size:]
    data_val = utils.jsort(data_val, key="id", integer=True)
    val_size = len(data_val)

    train_data_path = os.path.join(args.data_path, f"train-{train_size // 1000}k.json")
    val_data_path = os.path.join(args.data_path, f"val-{val_size // 1000}k.json")
    # save to json
    utils.jdump(data_train, train_data_path)
    utils.jdump(data_val, val_data_path)

    print(f"train data size: {len(data_train)}")
    print(f"val data size: {len(data_val)}")
