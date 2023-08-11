import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MentionDS(Dataset):
    def __init__(
            self,
            texts,
            tokenizer: AutoTokenizer,
            force_reload: bool = False
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.force_reload=force_reload
        self.length = sum(len(text.segments) for text in self.texts)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        path = self.all_paths[idx]
        tokenized_data_point = self.cached_load(path)
        return tokenized_data_point
