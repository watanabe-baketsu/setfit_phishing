import argparse

from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

from training import read_dataset, compute_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="../tuned_models/all-MiniLM-L6-v2", help="Name of the model to be used for training")
    parser.add_argument("--validation_data", type=str, default="preprocessing/dataset/dataset_full.json", help="Training dataset")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = read_dataset(file_path=args.validation_data)
    validation_data = dataset["validation"].shuffle()
    print(f"validation dataset count : {len(validation_data)}")

    # Build training model
    model = SetFitModel.from_pretrained(args.model_name, local_files_only=True)

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
