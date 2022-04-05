import torch
from torch.utils.data import Dataset
from typing import List, Any, Dict
from .preprocessing import process_train_data, process_inference_data
import ipdb


class NERDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, dataset: List[Dict], tokenizer: Any, cfg: Any):
        self.tokenizer = tokenizer
        if "templates" in dataset[0].keys():
            self.train = True
            self.processed_dataset = process_train_data(
                dataset, self.tokenizer, cfg)
        else:
            self.train = False
            self.processed_dataset = process_inference_data(
                dataset, self.tokenizer, cfg
            )

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        # item = {key: val[idx] for key, val in self.processed_dataset["encodings"].items()}
        item = {}
        item["input_ids"] = self.processed_dataset[idx]["input_ids"]
        item["attention_mask"] = self.processed_dataset[idx]["attention_mask"]
        item["docid"] = self.processed_dataset[idx]["docid"]
        if self.train:
            item["labels"] = torch.tensor(
                self.processed_dataset[idx]["labels"])
        return item

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """

        # docids = [ex["docid"] for ex in batch]
        input_ids = torch.stack([ex["input_ids"] for ex in batch])
        attention_mask = torch.stack([ex["attention_mask"] for ex in batch])
        labels = torch.stack([ex["labels"] for ex in batch])

        return {
            # "docid": docids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
