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

# Additional Information

This work was supported by the European Regional Development Fund as a part of 2014–2020 Smart Growth Operational Programme, CLARIN — Common Language Resources and Technology Infrastructure, project no. POIR.04.02.00-00C002/19 and by the project co-financed by the Minister of Education and Science under the agreement 2022/WK/09.
