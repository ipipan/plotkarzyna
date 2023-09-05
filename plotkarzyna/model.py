

import spacy_alignments
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


def split_and_tokenize(text: str, tokenizer: AutoTokenizer) -> dict:
    """
    Splits the input text into smaller chunks fitting within the tokenizer's max_len 
    and returns the tokenized form of each chunk as a batch. Attempts to split by sentences.
    
    Args:
    - text (str): The input text to split and tokenize.
    - tokenizer (AutoTokenizer): The tokenizer used for tokenization.

    Returns:
    - dict: A dictionary containing the tokenized inputs (input_ids, attention_mask, etc.)
    """
    
    max_len = tokenizer.model_max_length - 2  # account for [CLS] and [SEP] tokens
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check the length with current chunk + new sentence
        total_length = len(tokenizer.encode(current_chunk + sentence, add_special_tokens=True))
        
        # If the sentence with the current chunk is too long
        if total_length > max_len:
            # If even the sentence alone is too long
            if len(tokenizer.encode(sentence, add_special_tokens=True)) > max_len:
                # Split the sentence into words and add them until max_len
                words = sentence.split()
                temp_sentence = ""
                for word in words:
                    if len(tokenizer.encode(temp_sentence + " " + word, add_special_tokens=True)) <= max_len:
                        temp_sentence += " " + word
                    else:
                        chunks.append(temp_sentence)
                        temp_sentence = word
                chunks.append(temp_sentence)
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        else:
            current_chunk += " " + sentence

    # Add any remaining text in the current chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Convert each chunk into model inputs
    encoding = tokenizer.batch_encode_plus(chunks, padding='longest', return_tensors='pt', return_offsets_mapping=True)

    return encoding


def predict(text: str | list[str], model: AutoModel, tokenizer: AutoTokenizer, verbose: bool = False):
    if isinstance(text, list):
        segments = text
        text = ' '.join(segments)
    elif isinstance(text, str):
        segments = text.split(' ')

    tokenized = split_and_tokenize(text, tokenizer)
    pred = model(
    input_ids = tokenized.input_ids,
    attention_mask=tokenized.attention_mask,
    ).logits.argmax(-1)

    if verbose:
        print(' '.join([
            f"|{tokenizer.decode(tok, clean_up_tokenization_spaces=False)}| ({id2label[int(el)]})" 
            for el, tok in zip(pred, list(tokenized.input_ids[0]))
        ]))

    tokens = []
    labels = []
    spans_inds = []
    for ind in range(tokenized.input_ids.shape[0]):
        tokens.extend([tokenizer.decode(tok) for tok in list(tokenized.input_ids[ind])])
        labels.extend([id2label[int(el)] for el, tok in zip(pred[ind], list(tokenized.input_ids[ind]))])
        spans_inds.extend(decode_bioes(labels, tokens))
        
    mention_inds = [
        (span[0], (span[-1])) for span in spans_inds
        ]

    aligned_span_inds = align(segments, tokens, mention_inds)
    spans = [segments[span[0]:(span[-1])] for span in aligned_span_inds]
    
    return tokens, labels, aligned_span_inds, spans


def align_mention(mention_inds, subtoken2token_indices):
    start, end = mention_inds
    if subtoken2token_indices[start] and subtoken2token_indices[end]:
        start, end = (
            subtoken2token_indices[start][0],
            subtoken2token_indices[end][0]
        )

        return start, (end + 1)
    else:
        return None

def align(segments, tokens, mentions_inds):
    a2b, b2a = spacy_alignments.get_alignments(tokens, segments)
    # for ind, token in enumerate(tokens):
    #     print(token, a2b[ind], [segments[el] for el in a2b[ind]])
    aligned_mention_inds = []
    for mention_inds in mentions_inds:
        aligned = align_mention(mention_inds, a2b)
        if aligned:
            aligned_mention_inds.append(aligned)

    return aligned_mention_inds