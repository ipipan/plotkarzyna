from pathlib import Path
from dotenv import dotenv_values, load_dotenv

import torch
import spacy
import coref_ds
from coref_ds.text import Text
from coref_ds.mmax.mmax_doc import MmaxDoc
from coref_ds.tei.tei_doc import TEIDocument
from coref_ds.tei.mention import Mention

from plotkarzyna.model import get_model, predict
from plotkarzyna.utils import get_head

try:
    local_config = dotenv_values(".env")
except FileNotFoundError:
    local_config = dict()
load_dotenv()


def predict(model_path, input_dir, n_samples=None):
    paths = list((input_dir).iterdir())
    if n_samples:
        paths = paths[:n_samples]

    tei_docs = []
    for p in paths:
        try: 
            m = TEIDocument(p)
            tei_docs.append(m)
        except Exception as e:
            print(e)

    tokenizer, model = get_model(checkpoint_path)
    preds = []
    for ind, doc in enumerate(tei_docs):
        print(ind, end=' ')
        pred = predict(doc.text.segments, model, tokenizer)
        preds.append(pred)

    nlp = spacy.load("pl_core_news_lg")
    pred_path = Path(local_config['PCC_ROOT']) / f'plotkarzyna_{model_path.parent.name}_{model_path.stem}'
    pred_path.mkdir(exist_ok=True)
    for doc, pred in zip(tei_docs, preds):
        doc.layers['mentions'].remove_mentions()
        for ind, span in enumerate(pred[-2]):
            start, end = span
            segments = doc.text.segments[start:(end)]
            head = get_head(segments, nlp)
            try:
                doc.layers['mentions'].add_mention(
                    Mention(
                        id=f"mention_{ind}",
                        text=' '.join(segments),
                        segments=segments,
                        span_start=start,
                        span_end=end,
                        lemmatized_text=None,
                        head_orth=None,
                        head=start + head,
                        cluster_id=None,
                    ),
                    doc.text.segments_meta
                )
            except:
                print(doc.doc_path)
                print(head, start, len(doc.text.segments_meta), segments, doc.text.segments)
        doc.to_file(pred_path / doc.doc_path.name)


if __name__ == '__main__':
    input_dir = Path(local_config['PCC_ROOT']) / 'test'
    checkpoint_path = '/home/ksaputa/mspace/plotkarzyna/models/herbert-large-2/checkpoint-6240'
    predict(checkpoint_path, input_dir)