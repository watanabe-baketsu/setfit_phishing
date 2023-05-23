import json
import os
from pprint import pprint


def data_generator(data_dir: str) -> dict:
    """
    data_dir: str
        Path to the directory containing the data
    """
    for filename in os.listdir(data_dir):
        with open(data_dir + "/" + filename, 'r', errors='ignore') as f:
            data = f.read()
        dict_data = {
            "text": data,
            "label": 1 if data_dir.split("/")[-1] == "phish" else 0,
            "label_text": "positive" if data_dir.split("/")[-1] == "phish" else "negative"
        }
        yield dict_data


def create_dataset(data_generator: iter) -> list:
    training_data = []
    for data in data_generator:
        training_data.append(data)
    return training_data


def main():
    phishing_data_dir = "/phish"
    not_phishing_data_dir = "/not_phish"
    training_data_dir = os.path.join(os.path.dirname(__file__), "dataset/training")
    validation_data_dir = os.path.join(os.path.dirname(__file__), "dataset/validation")

    # Create training data
    training_data = create_dataset(data_generator(training_data_dir + phishing_data_dir))
    training_data.extend(create_dataset(data_generator(training_data_dir + not_phishing_data_dir)))

    # Create validation data
    validation_data = create_dataset(data_generator(validation_data_dir + phishing_data_dir))
    validation_data.extend(create_dataset(data_generator(validation_data_dir + not_phishing_data_dir)))

    print(f"training dataset : {len(training_data)}")
    print(f"validation dataset : {len(validation_data)}")

    # Save training data
    with open(os.path.join(os.path.dirname(__file__) , "dataset/dataset.json"), "w") as f:
        dataset = {
            "training": training_data,
            "validation": validation_data
        }
        json.dump(dataset, f)


main()