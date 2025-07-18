import json
import joblib
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel
from models.feature_extractor import build_feature_vector

# === Load TinyBERT ===
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
model.eval()

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[0][0]  # CLS token

# === Load Your Labeled JSON Data ===
with open("train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X = []
y = []

for item in data:
    layout_vec = build_feature_vector(item)            # layout features
    embed_vec = get_text_embedding(item["text"])       # semantic features
    combined = torch.cat([layout_vec, embed_vec], dim=-1)
    X.append(combined.numpy())
    y.append(item["label"])

X = np.array(X)
y = np.array(y)

# === Train Multiclass Logistic Regression ===
clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
clf.fit(X, y)

# === Save Classifier ===
joblib.dump(clf, "classifier_model.pkl")
print("✅ Model trained and saved as classifier_model.pkl")
