from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

from training import read_dataset, compute_metrics


if __name__ == "__main__":
    # Load dataset
    dataset = read_dataset("preprocessing/dataset/dataset.json")
    validation_data = dataset["validation"].shuffle()
    print(f"validation dataset count : {len(validation_data)}")

    # Build training model
    model = SetFitModel.from_pretrained("../tuned_models/all-MiniLM-L6-v2", local_files_only=True)

    # Validate model
    trainer = SetFitTrainer(
        model=model,
        train_dataset=None,
        eval_dataset=validation_data,
        loss_class=CosineSimilarityLoss,
        metric=compute_metrics
    )

    metrics = trainer.evaluate()
    print(metrics)
