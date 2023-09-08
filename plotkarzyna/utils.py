import spacy


def get_head(segments, nlp):
    doc = spacy.tokens.Doc(nlp.vocab, words=segments, spaces=[True]*len(segments))
    doc = nlp(doc)
    root_index = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_index = token.i
            break
    if root_index is None:
        return 0
    else:
        return root_index
