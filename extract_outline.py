from pdf_parser import extract_lines_with_features
from models.feature_extractor import build_feature_vector
from models.tinybert_embedder import get_text_embedding
from models.heading_classifier import load_classifier, classify
import torch
import json

clf = load_classifier()

def extract_outline(pdf_path, output_json_path):
    lines = extract_lines_with_features(pdf_path)

    outline = []
    title = lines[0]["text"] if lines else "Untitled Document"

    for line in lines:
        layout_vec = build_feature_vector(line)
        embed_vec = get_text_embedding(line["text"])
        combined_vec = torch.cat([layout_vec, embed_vec], dim=-1)

        label = classify(clf, combined_vec)

        if label > 0:  # label 1 = H1, 2 = H2, 3 = H3
            outline.append({
                "level": f"H{label}",
                "text": line["text"],
                "page": line["page"]
            })

    result = {
        "title": title,
        "outline": outline
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
