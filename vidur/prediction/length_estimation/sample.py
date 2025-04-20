import argparse
import os

import numpy as np
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/sharegpt/generate")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
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
            data_path = os.path.join(data_path, file)
            data += utils.jload(data_path)
    print(f"data size: {len(data)}")
    # random sample 40k
    N = args.num_samples
    if N < len(data):
        N = len(data)
    data_mask = np.random.choice(len(data), N, replace=False)
    data = [data[i] for i in data_mask]
    data = data[:N]
    train_size = int(args.train_ratio * N)
    data_train = data[:train_size]
    data_train = utils.jsort(data_train, key="id", integer=True)
    data_val = data[train_size:]
    data_val = utils.jsort(data_val, key="id", integer=True)
    val_size = len(data_val)

    train_data_path = args.data_path.replace("with_real_response", "-train-{}-k".format(train_size // 1000))
    val_data_path = args.data_path.replace("fwith_real_response", "-val-{}-k".format(val_size // 1000))
    # save to json
    utils.jdump(data_train, train_data_path)
    utils.jdump(data_val, val_data_path)

    print(f"train data size: {len(data_train)}")
    print(f"val data size: {len(data_val)}")
