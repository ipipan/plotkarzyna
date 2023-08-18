from pathlib import Path

from dotenv import dotenv_values, load_dotenv
import seqeval

import coref_ds
from coref_ds.text import Text
from coref_ds.mmax.mmax_doc import MmaxDoc
from coref_ds.tei.tei_doc import TEIDocument

from plotkarzyna.config import *


import evaluate

seqeval = evaluate.load("seqeval")

try:
    local_config = dotenv_values(".env")
except FileNotFoundError:
    local_config = dict()
load_dotenv()
n_samples = 100_000_000
mmax_paths = list(Path(local_config['NKJP_MMAX_ROOT']).glob('**/*.mmax'))[:n_samples]
tei_train_paths = list(
    (Path(local_config['PCC_ROOT']) / 'train').iterdir()
    )[:n_samples]

test_paths = list(
    (Path(local_config['PCC_ROOT']) / 'test').iterdir()
    )[:n_samples]

mmax_docs = []
for p in mmax_paths:
    try: 
        m = MmaxDoc.from_file(p)
        mmax_docs.append(m)
    except Exception as e:
        print(e)


tei_docs = []
for p in tei_train_paths:
    try: 
        m = TEIDocument(p)
        tei_docs.append(m)
    except Exception as e:
        print(e)


test_docs = []
for p in test_paths:
    try: 
        m = TEIDocument(p)
        test_docs.append(m)
    except Exception as e:
        print(e)

texts = []
test_texts = []

for m in tei_docs:
    texts.append(m.text)

for m in test_docs:
    test_texts.append(m.text)

from transformers import AutoTokenizer
from plotkarzyna.dataset import MentionDS
tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-base-cased')

train_ds = MentionDS(
    texts,
    tokenizer
)

test_ds = MentionDS(
    test_texts,
    tokenizer
)

# for text in train_ds:
#     assert len(text['input_ids']) == len(text['labels'])

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

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    "allegro/herbert-large-cased", num_labels=5, id2label=id2label, label2id=label2id
)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    max_length=512,
    padding='max_length',
    )

import numpy as np

def compute_metrics(p):
    predictions, labels = p
    label_list = list(label2id.keys())
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

batch_size = 32
training_args = TrainingArguments(
    output_dir=Path(local_config['MODEL_DIR']) / 'herbert-large-2',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=60,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

from torch.utils.data import DataLoader
for ind, sample in enumerate(train_ds):
    # print(sample['input_ids'].shape, len(sample['labels']))
    if ind > 5:
        break

dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    )
for ind, batch in enumerate(dl):
    print(batch)
    print(batch['input_ids'].shape, batch['token_type_ids'].shape, batch['attention_mask'].shape, batch['labels'].shape)
    if ind > 5:
        break



trainer.train()