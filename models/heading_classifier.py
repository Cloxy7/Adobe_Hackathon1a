import joblib
import torch

def load_classifier(path="classifier_model.pkl"):
    return joblib.load(path)

def classify(clf, tensor):
    vec = tensor.detach().numpy().reshape(1, -1)
    return clf.predict(vec)[0]  # 0 = Not heading, 1 = H1, 2 = H2, 3 = H3
