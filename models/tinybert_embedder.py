from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
model.eval()

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[0][0]  # CLS token
