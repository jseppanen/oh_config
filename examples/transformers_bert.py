"""
Adapted from the snippet at https://huggingface.co/bert-base-uncased
"""

import oh

oh.config.load_str(
"""
[tokenizer]
@call = transformers/BertTokenizer.from_pretrained
0 = "bert-base-uncased"

[model]
@call = transformers/BertModel.from_pretrained
0 = "bert-base-uncased"
"""
)

tokenizer = oh.config.tokenizer()
model = oh.config.model()
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)
