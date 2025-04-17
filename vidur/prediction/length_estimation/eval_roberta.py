import argparse
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from transformers import AutoTokenizer
import utils
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="asdwb/roberta_base_llama_responses_length_prediction")
    parser.add_argument("--val-data-path", type=str, default="data/sharegpt-val-10k.json")
    parser.add_argument("--num-eval-examples", type=int, default=10000)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--tag-data", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    val_data = utils.jload(args.val_data_path)
    num_eval_examples = args.num_eval_examples
    tokenizer_model = AutoTokenizer.from_pretrained(args.tokenizer)
    sampled_val_data, val_data_list = utils.generate_regression_dataframe(tokenizer_model, val_data, num_eval_examples)

    model = ClassificationModel("roberta", args.model_path)
    result, model_outputs, wrong_predictions = model.eval_model(sampled_val_data)

    d_max = []
    assert len(val_data_list) == len(model_outputs)
    for i in range(len(val_data_list)):
        x = val_data_list[i][1]
        y = max(1, int(model_outputs[i]))
        d_max.append(abs(x - y))

    diff = sum(d_max)
    acc_10t = sum([1 if x <= 10 else 0 for x in d_max])
    acc_50t = sum([1 if x <= 50 else 0 for x in d_max])
    acc_100t = sum([1 if x <= 100 else 0 for x in d_max])

    print(f"# Samples: {num_eval_examples}")
    print(f"Error: {diff / num_eval_examples}")
    print(f"Acc-10: {acc_10t / num_eval_examples}")
    print(f"Acc-50: {acc_50t / num_eval_examples}")
    print(f"Acc-100: {acc_100t / num_eval_examples}")

    if args.tag_data:
        conversations = [dp["conversations"][0]["value"] for dp in val_data]
        predictions, raw_outputs = model.predict(conversations)
        tagged_data = []
        for i in range(len(val_data)):
            data_record = val_data[i]
            data_record["predicted_length"] = int(max(1, predictions[i]))
            tagged_data.append(data_record)

        tagged_val_data_path = args.val_data_path.replace(".json", "-predicted.json")
        with open(tagged_val_data_path, "w+") as fp:
            json.dump(tagged_data, fp)
