

def rindex(lst, value='B'):
    lst.reverse()
    try:
        i = lst.index(value)
    except ValueError:
        return 0
    else:
        lst.reverse()
        index = len(lst) - i - 1
        if index in range(len(lst)):
            return index
        else:
            return 0


def decode_bioes(labels, tokens):
    spans = []
    curr_spans = []
    for ind, (token, token_label) in enumerate(zip(tokens, labels)):
        if token_label == 'O':
            if curr_spans: # finish all open spans
                spans.extend(curr_spans)
                curr_spans = []
            else:
                continue
        else:
            for curr_span in curr_spans:
                curr_span.append(ind)

            prev_label = labels[ind + 1] if len(labels) - 1 > ind else None
            next_label = labels[ind + 1] if len(labels) - 1 > ind else None

            if token_label in ['S']:
                if prev_label and prev_label == 'S':
                    continue
                else:
                    spans.append([ind])
            elif token_label in ['B']:
                if prev_label and prev_label == 'B':
                    continue
                else:
                    curr_spans.append([ind])
            elif token_label == 'I':
                continue
            elif token_label == 'E': # finish one shortest span
                if next_label and next_label == 'E':
                    continue

                if curr_spans:
                    try:
                        span_to_finish = curr_spans.pop()
                        spans.append(span_to_finish)
                    except IndexError:
                        pass
    spans.extend(curr_spans)
    return spans