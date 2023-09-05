

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
    curr_spans_types = []
    for ind, (token, token_label) in enumerate(zip(tokens, labels)):
        if token_label == 'O':
            if curr_spans:
                spans.extend(curr_spans)
                curr_spans = []
                curr_spans_types = []
            else:
                continue
        else:
            for curr_span in curr_spans:
                curr_span.append(ind)

            if token_label in ['B', 'S']:
                if curr_spans_types and curr_spans_types[-1] == 'S':
                    continue
                else:
                    curr_spans_types.append(token_label)
                    curr_spans.append([ind])
            elif token_label == 'I':
                continue
            elif token_label == 'E':
                next_label = labels[ind + 1] if len(labels) - 1 > ind else None
                if next_label and next_label == 'E':
                    # print(curr_spans, curr_spans_types)
                    curr_spans_types.append(token_label)
                    continue

                span_to_finish_ind = rindex(curr_spans_types)
                # print(curr_spans, curr_spans_types, span_to_finish_ind)
                if curr_spans_types and curr_spans:
                    try:
                        curr_spans_types.pop(span_to_finish_ind)
                        span_to_finish = curr_spans.pop(span_to_finish_ind)
                        spans.append(span_to_finish)
                    except IndexError:
                        pass
    spans.extend(curr_spans)
    return spans