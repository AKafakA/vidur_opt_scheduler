import io
import json
import os
import random
import re
import time
import pandas as pd

import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def timeit(T0=None):
    torch.cuda.synchronize()
    T1 = time.time()
    if T0 is not None:
        T1 = T1 - T0
    return T1


def describe(input_len, name=""):
    print(f"Statistics of {name}:")
    print(f"\tMean: {np.mean(input_len):.2f}, Std: {np.std(input_len):.2f}")
    # print(f"\tquartiles: {np.quantile(input_len, [0, 0.25, 0.5, 0.75, 1])}")


def buckit(x, cell=50):
    x = int(x)
    x = (x // cell + 1) * cell if x % cell != 0 else x
    return x


def extract_all_numbers(string):
    all_number = [int(s) for s in re.findall(r"\d+", string)]
    if len(all_number) == 1:
        return all_number[0]
    elif len(all_number) == 0:
        return 0
    else:
        return (all_number[0] + all_number[1]) / 2


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    file_name = f
    f = _make_r_io_base(f, mode)
    if file_name.endswith("json"):
        jdict = json.load(f)
    else:
        jdict = [json.loads(line) for line in f]
    f.close()
    return jdict


def jsort(obj, key, integer=False):
    assert isinstance(obj, list)
    if integer:
        return sorted(obj, key=lambda x: int(x[key]))
    return sorted(obj, key=lambda x: x[key])


def generate_regression_dataframe(tokenizer_model, raw_data, num_sampled=-1):
    regression_dataset = []
    for i in range(len(raw_data)):
        new_data = []
        new_data.append(raw_data[i]["conversations"][0]["value"])
        len_to_predict = len(tokenizer_model.tokenize(raw_data[i]["conversations"][1]["value"]))
        new_data.append(len_to_predict)
        regression_dataset.append(new_data)
    if 0 < num_sampled < len(regression_dataset):
        regression_dataset = random.sample(regression_dataset, num_sampled)
    regression_df = pd.DataFrame(regression_dataset)
    regression_df.columns = ["text", "labels"]
    return regression_df, regression_dataset
