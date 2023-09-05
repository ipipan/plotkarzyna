

from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
import nltk
from nltk.tokenize import sent_tokenize

from plotkarzyna.decode import *


id2label = {
    0: "O",
    1: "B",
    2: "I",
    3: "E",
    4: "S"
}

nltk.download('punkt')

def get_model(checkpoint_path):
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        padding=True,
        )
    return tokenizer, model

def chunkize(text, tokenizer):
    sentences = sent_tokenize(' '.join(text), language='polish')
    chunks = []
    current_chunk = []
    current_length = 0
    max_len = tokenizer.model_max_length - 2
    for sentence in sentences:
        # print(sentence)
        sentence_tokens = tokenizer.tokenize(sentence, padding=False, is_split_into_words=False, add_special_tokens=False)
        input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
        if current_length + len(sentence_tokens) <= max_len:
            current_chunk.extend(sentence_tokens)
            current_length += len(sentence_tokens)
        else:
            chunks.append(current_chunk)
            current_chunk = sentence_tokens
            current_length = len(sentence_tokens)
    if current_chunk:  # Add any remaining tokens as a chunk
        chunks.append(current_chunk)
    print([len(sent) for sent in sentences], [len(chunk) for chunk in chunks])
    return chunks

def predict(text, model, tokenizer, verbose=False):
    if isinstance(text, list):
        is_split_into_words = True
    else:
        is_split_into_words = False

    chunks = chunkize(text, tokenizer)
    print(f"chunks: {chunks}")
    print([len(chunk) for chunk in chunks])
    tokenized = tokenizer.batch_encode_plus(chunks, return_tensors='pt', is_split_into_words=True)
    print(tokenized.input_ids.shape, tokenized.attention_mask.shape)
    print(tokenized)
    pred = model(
    **tokenized
    ).logits.argmax(-1)

    if verbose:
        print(' '.join([
            f"|{tokenizer.decode(tok, clean_up_tokenization_spaces=False)}| ({id2label[int(el)]})" for el, tok in zip(pred, list(tokenized['input_ids'][0]))
        ]))
    tokens = []
    labels = []
    spans_inds = []

    for ind, text in enumerate(chunks):
        tokens.extend([tokenizer.decode(tok) for tok in list(tokenized['input_ids'][ind])])
        labels.extend([id2label[int(el)] for el, tok in zip(pred[ind], list(tokenized['input_ids'][ind]))])
        spans_inds.extend(decode_bioes(labels, tokens))
    return tokens, labels, spans_inds


# def predict(text, model, tokenizer, verbose=False):
#     if isinstance(text, list):
#         is_split_into_words = True
#     else:
#         is_split_into_words = False

#     if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length and is_split_into_words:
#         max_len = tokenizer.model_max_length - 2
#         texts = [text[start:(start+max_len)] for start in range(0, len(text) + max_len, max_len)]
#     else:
#         texts = [text]
#     texts = list(filter(lambda t: t, texts))
#     print([len(text) for text in texts])

#     tokenized = tokenizer(texts, return_tensors='pt', is_split_into_words=is_split_into_words)
#     pred = model(
#     **tokenized
#     ).logits.argmax(-1)

#     if verbose:
#         print(' '.join([
#             f"|{tokenizer.decode(tok, clean_up_tokenization_spaces=False)}| ({id2label[int(el)]})" for el, tok in zip(pred, list(tokenized['input_ids'][0]))
#         ]))
#     tokens = []
#     labels = []
#     spans_inds = []

#     for ind, text in enumerate(texts):
#         tokens.extend([tokenizer.decode(tok) for tok in list(tokenized['input_ids'][ind])])
#         labels.extend([id2label[int(el)] for el, tok in zip(pred[ind], list(tokenized['input_ids'][ind]))])
#         spans_inds.extend(decode_bioes(labels, tokens))
#     return tokens, labels, spans_inds
