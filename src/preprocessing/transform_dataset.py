import json
from pprint import pprint

from html_summarizer import Summarizer


if __name__ == "__main__":
    with open("dataset/dataset.json", "r") as f:
        data = json.load(f)

    print(len(data["training"]))
    print(len(data["validation"]))

    summarizer = Summarizer(model_name="google/long-t5-tglobal-large")

    training_data = data["training"]
    validation_data = data["validation"]

    for data in training_data:
        data["text"] = summarizer.summarize_text(data["text"])
        pprint(data["text"])
    for data in validation_data:
        data["text"] = summarizer.summarize_text(data["text"])
        pprint(data["text"])

    with open("dataset/dataset_summarized.json", "w") as f:
        json.dump({
            "training": training_data,
            "validation": validation_data
        }, f)



