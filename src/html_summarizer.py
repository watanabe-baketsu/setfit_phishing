import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Summarizer:
    def __init__(self, model_name: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(f"Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def summarize_dataset(self, dataset: Dataset) -> Dataset:
        """
        dataset: dict
            Dataset to be summarized
        """
        from pprint import pprint
        # Summarize dataset
        for data in dataset:
            data["text"] = "summarize: " + data["text"]
            inputs = self.tokenizer(data["text"], return_tensors="pt", max_length=16384, truncation=True).input_ids
            summary_ids = self.model.generate(inputs, max_length=512, min_length=50, do_sample=False)
            data["text"] = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            pprint(data)

        return dataset

    def summarize_text(self, text: str) -> str:
        """
        text: str
            Text to be summarized
        """
        text = "summarize: " + text
        inputs = self.tokenizer(text, return_tensors="pt", max_length=16384, truncation=True).input_ids
        summary_ids = self.model.generate(inputs, max_length=512, min_length=50, do_sample=False)

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
