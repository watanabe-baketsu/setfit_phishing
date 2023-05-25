import json
from pprint import pprint

from bs4 import BeautifulSoup


if __name__ == "__main__":
    with open("dataset/dataset.json", "r") as f:
        data = json.load(f)

    print(len(data["training"]))
    print(len(data["validation"]))

    training_data = data["training"]
    validation_data = data["validation"]

    for i, data in enumerate(training_data):
        print(f"\rtraining data : {i + 1}/{len(training_data)}", end="")
        soup = BeautifulSoup(data["text"], "html.parser")
        script_meta_tags = soup.find_all(["script", "meta", "p", "title", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = ""
        for tag in script_meta_tags:
            text += f"{tag}"
        data["text"] = text
    for i, data in enumerate(validation_data):
        print(f"\rvalidation data : {i + 1}/{len(validation_data)}", end="")
        soup = BeautifulSoup(data["text"], "html.parser")
        script_tags = soup.find_all("script")
        meta_tags = soup.find_all("meta")
        text = ""
        for tag in script_tags:
            text += f"{tag}"
        data["text"] = text

    with open("dataset/dataset_script_only.json", "w") as f:
        json.dump({
            "training": training_data,
            "validation": validation_data
        }, f)



