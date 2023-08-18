import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

id2label = {
    0: "O",
    1: "B",
    2: "I",
    3: "E",
    4: "S"
}
label2id = {
    "O": 0,
    "B": 1,
    "I": 2,
    "E": 3,
    "S": 4,
}


def add_bioes(texts):
    for text in texts:
        annotation = ['O'] * len(text.segments)
        mentions = sorted(
            [mention for cluster in text.clusters for mention in cluster], key=lambda x: x[0])
        for mention in mentions:
            start, end = mention
            if start == end:
                annotation[start] = 'S'
            else:
                annotation[start] = 'B'
                annotation[end] = 'E'
            for ind in range(start, end):
                if annotation[ind] == 'O':
                    annotation[ind] = 'I'
        annotation = [label2id[ann] for ann in annotation]
        text.mention_annotation = annotation


def tokenize_and_align_labels(text, tokenizer):
    tokenized_inputs = tokenizer(
        text["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        # return_tensors='pt',
    )

    labels = []
    for i, label in enumerate(text[f'labels']):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        label_ids = label_ids[:len(word_ids)]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class MentionDS(Dataset):
    def __init__(
            self,
            texts,
            tokenizer: AutoTokenizer,
            force_reload: bool = False
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.force_reload = force_reload
        # self.length = sum(len(text.segments) for text in self.texts)
        add_bioes(self.texts)
        self.tokenized = [
            tokenize_and_align_labels(
            {
                'tokens': [text.segments],
                'labels': [text.mention_annotation],
            }, self.tokenizer
        ) for text in self.texts]
        for text in self.tokenized:
            text['input_ids'] = text['input_ids'][0]
            text['token_type_ids'] = text['token_type_ids'][0]
            text['attention_mask'] = text['attention_mask'][0]
            text['labels'] = text['labels'][0]

        self.labels = [tokenized['labels'][0] for tokenized in self.tokenized]
        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # path = self.all_paths[idx]
        # tokenized_data_point = self.cached_load(path)
        # return tokenized_data_point
        return self.tokenized[idx] #, self.labels[idx]
