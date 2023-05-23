import json

import datasets
from datasets import Dataset, DatasetDict
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer


def read_dataset(file_path: str) -> DatasetDict:
    """
    file_path: str
        Path to the dataset file
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = DatasetDict({
        "training": Dataset.from_list(data["training"]),
        "validation": Dataset.from_list(data["validation"]),
    })

    return dataset


def build_trainer(
        model_name: str,
        train_dataset: datasets.Dataset,
        validation_data: datasets.Dataset) -> SetFitTrainer:
    """
    model_name: str
        Name of the model to be used for training
    train_dataset: dict
        Training dataset
    validation_data: dict
        Validation dataset
    """
    # Build training model
    model = SetFitModel.from_pretrained(model_name, max_length=16384)

    # Build training trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_data,
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_epochs=20,
    )

    return trainer


if __name__ == "__main__":
    # Read dataset
    dataset = read_dataset(file_path="preprocessing/dataset/dataset.json")
    training_data = dataset["training"].shuffle().select(range(30))
    validation_data = dataset["validation"].shuffle()

    print(f"training dataset count : {len(training_data)}")
    print(f"validation dataset count : {len(validation_data)}")

    # Build trainer
    trainer = build_trainer(
        model_name= "sentence-transformers/all-MiniLM-L6-v2",  # "google/long-t5-tglobal-xl",
        train_dataset=training_data,
        validation_data=validation_data
    )

    # Train model
    trainer.train()

    # validate model
    metrics = trainer.evaluate()
    print(metrics)