import argparse

import numpy as np
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/sharegpt_v3_full.json")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--train-size", type=int, default=40000)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="./data/")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    seed = 42
    utils.set_seed(seed)
    args = parse_args()
    data_path = args.data_path
    data = utils.jload(data_path)

    # random sample 40k
    N = args.num_samples
    data_mask = np.random.choice(len(data), N, replace=False)
    data = [data[i] for i in data_mask]
    data = data[:N]
    train_size = args.train_size
    data_train = data[:train_size]

    val_size = args.val_size
    data_train = utils.jsort(data_train, key="id", integer=True)
    data_val = data[train_size: train_size + val_size]
    data_val = utils.jsort(data_val, key="id", integer=True)

    train_data_path = args.data_path.replace("full", "train-{}-k".format(train_size // 1000))
    val_data_path = args.data_path.replace("full", "val-{}-k".format(val_size // 1000))
    # save to json
    utils.jdump(data_train, train_data_path)
    utils.jdump(data_val, val_data_path)
