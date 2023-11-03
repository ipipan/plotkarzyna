# plotkarzyna

## a baseline for mention detection in Polish


```

from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
checkpoint_path = Path(local_config['PREDICTION_CHECKPOINT_PATH'])
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, return_tensors='pt')
from plotkarzyna.model import predict

text = """
Jest jeszcze jeden sposób, by ocalić chwasty. Tak naprawdę może się do tego przyczynić każdy, kto ma ogród lub działkę, kawałek ziemi.
"""

tokens, labels, aligned_span_inds, spans = predict(text, model, tokenizer, verbose=False)
for span in sorted(spans):
    print(span)
```
