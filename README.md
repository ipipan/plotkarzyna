# plotkarzyna

## a baseline for mention detection in Polish


```
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification

from plotkarzyna.model import predict

checkpoint_path = 'ipipan/plotkarzyna-large'
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, return_tensors='pt')

text = """Jest jeszcze jeden sposób, by ocalić chwasty. Tak naprawdę może się do tego przyczynić każdy, kto ma ogród lub działkę, kawałek ziemi."""

tokens, labels, aligned_span_inds, spans = predict(text, model, tokenizer, verbose=False)
for span in sorted(spans):
    print(span)
```
