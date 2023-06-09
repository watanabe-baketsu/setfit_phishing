import argparse
import json
from typing import Dict

import datasets
import evaluate
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


def compute_metrics(y_pred, y_test) -> Dict[str, float]:
    accuracy_metric = evaluate.load("accuracy")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    f1_metric = evaluate.load("f1")
    # or "macro" or "weighted" or None as the average
    return {
        "accuracy": accuracy_metric.compute(references=y_test, predictions=y_pred)['accuracy'],
        "recall": recall_metric.compute(references=y_test, predictions=y_pred)['recall'],
        "precision": precision_metric.compute(references=y_test, predictions=y_pred)['precision'],
        "f1": f1_metric.compute(references=y_test, predictions=y_pred)['f1'],
    }


def build_trainer(
        model_name: str,
        training_data: datasets.Dataset,
        validation_data: datasets.Dataset,
        mode: str) -> SetFitTrainer:
    """
    model_name: str
        Name of the model to be used for training
    train_dataset: dict
        Training dataset
    validation_data: dict
        Validation dataset
    mode: str
    """
    # Build training model
    if mode == "end-to-end":
        model = SetFitModel.from_pretrained(model_name, use_differentiable_head=False).to(args.device)
    else:
        model = SetFitModel.from_pretrained(model_name, use_differentiable_head=True)

    # Build training trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=training_data,
        eval_dataset=validation_data,
        loss_class=CosineSimilarityLoss,
        batch_size=args.batch_size,
        num_iterations=20,
        num_epochs=1,
        metric=compute_metrics,
    )

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--training_mode", type=str, default="end-to-end")
    parser.add_argument("--training_data_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Read dataset
    dataset = read_dataset(file_path=args.dataset_path)
    training_data = dataset["training"].shuffle(seed=25).select(range(args.training_data_size))
    validation_data = dataset["validation"].shuffle()

    print(f"training dataset count : {len(training_data)}")
    print(f"validation dataset count : {len(validation_data)}")

    model_name = args.model_name  # "sentence-transformers/all-MiniLM-L6-v2"

    # Build trainer
    trainer = build_trainer(
        model_name=model_name,
        training_data=training_data,
        validation_data=validation_data,
        mode=args.training_mode
    )

    if args.training_mode == "end-to-end":
        # Train model
        trainer.train()
    elif args.training_mode == "differentiable-head":
        # Freeze model
        trainer.freeze()
        # Train model
        trainer.train()

        # Unfreeze model (keep body frozen)
        trainer.unfreeze(keep_body_frozen=True)
        # Train model
        trainer.train(
            num_epochs=25,
            batch_size=args.batch_size,
            learning_rate=1e-5,
            l2_weight=0.0
        )
    else:
        raise ValueError("Invalid training mode")

    save_directory = f"../tuned_models/{model_name.split('/')[-1]}"
    trainer.model.save_pretrained(save_directory=save_directory)

    # validate model
    metrics = trainer.evaluate()
    print(metrics)

