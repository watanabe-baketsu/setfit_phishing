import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, pipeline


class Summarizer:
    def __init__(self, model_name: str):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.config = AutoConfig.from_pretrained(model_name)
        self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=device)

    def summarize_dataset(self, dataset: Dataset) -> Dataset:
        """
        dataset: dict
            Dataset to be summarized
        """
        from pprint import pprint
        # Summarize dataset
        for data in dataset:
            data["text"] = self.pipe(data["text"], max_length=128, min_length=50, do_sample=False)[0]["summary_text"]
            pprint(data)

        return dataset

    def summarize_text(self, text: str) -> str:
        """
        text: str
            Text to be summarized
        """
        return self.pipe(text, max_length=128, min_length=50, do_sample=False)[0]["summary_text"]