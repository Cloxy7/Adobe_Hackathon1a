import pymupdf  # PyMuPDF
import re
import json
from collections import Counter
import uuid
import os
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------
# Utility Functions
# ---------------------------------------------

def normalize(txt):
    """Normalizes text by stripping, lowercasing, and removing punctuation."""
    if not txt:
        return ""
    txt = txt.strip().lower()
    txt = re.sub(r'[-:;·•.,]+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt

# ---------------------------------------------
# Stage 1: Block-Level Feature Extraction
# ---------------------------------------------

def extract_blocks_with_features(pdf_path):
    """
    Extracts text blocks from a PDF and computes features for each block.
    This version aggregates features from lines into a single block representation.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []

    blocks_data = []

    all_font_sizes = []
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if b['type'] == 0 and "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        if s["text"].strip():
                            all_font_sizes.append(s["size"])
    avg_doc_font = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 12.0

    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        blocks = page.get_text("dict").get("blocks", [])
        prev_block_bottom = 0

        for block in blocks:
            if "lines" not in block:
                continue

            all_spans = [s for l in block["lines"] for s in l["spans"] if s["text"].strip()]
            if not all_spans:
                continue

            block_text = " ".join(s["text"] for s in all_spans).strip()

            # Refined noise filter
            if not block_text or (len(block_text) < 3 and not block_text.isdigit()):
                 continue

            font_sizes = [s["size"] for s in all_spans]
            font_flags = [s["flags"] for s in all_spans]

            max_font_size = max(font_sizes)
            is_bold = any(f & 16 for f in font_flags)

            first_line_text = block["lines"][0]["spans"][0]["text"].strip()
            has_number_prefix = bool(re.match(r'^(\d+(\.\d+)*|[IVXLCDM]+)[\.\)]?\s+', first_line_text))

            x0, y0, x1, y1 = block["bbox"]

            # --- Assemble the feature dictionary ---
            block_features = {
                "id": str(uuid.uuid4()),
                "text": block_text,
                "page": page_num,
                "word_count": len(block_text.split()), # Important new feature
                "font_size_norm": round(min(max_font_size / avg_doc_font, 3.0) / 3.0, 3),
                "is_bold": int(is_bold),
                "is_all_caps": int(block_text.upper() == block_text and block_text.replace(" ", "").isalpha()),
                "block_width_ratio": round((x1 - x0) / page_width, 3),
                "block_ends" : (x1/page_width),
                "is_centered": int(abs(((x0 + x1) / 2) - (page_width / 2)) < page_width * 0.1),
                "has_number_prefix": int(has_number_prefix),
                "first_letter_capital": int(block_text[0].isupper() if block_text else 0),
            }
            blocks_data.append(block_features)
            prev_block_bottom = y1

    doc.close()
    return blocks_data


def label_pdf_blocks(blocks, y_path):
    """
    Applies labels to extracted blocks based on a ground truth JSON file.
    The matching is done against the entire block's text.
    """
    with open(y_path, 'r', encoding='utf-8') as yf:
        y_data = json.load(yf)

    outlines = y_data.get("outline", [])
    title = y_data.get("title", "")

    # Create a lookup for faster matching
    outline_map = {normalize(o.get("text", "")): o for o in outlines if o.get("text")}
    normalized_title = normalize(title)

    for block in blocks:
        block['label'] = 0  # Default label is 0 (e.g., paragraph)
        normalized_block_text = normalize(block.get("text", ""))

        # Check for title match
        if normalized_title and (normalized_block_text in normalized_title or normalized_title in normalized_block_text):
            block['label'] = 1 # Label 1 for TITLE
            continue

        # Check for outline (heading) match
        for norm_outline_text, outline_data in outline_map.items():
            if norm_outline_text in normalized_block_text:
                level_str = outline_data.get("level", "H0")
                # Assuming level is like "H1", "H2", etc.
                level_num = int(re.search(r'\d+', level_str).group()) if re.search(r'\d+', level_str) else 0
                block['label'] = level_num + 1 # H1 -> 2, H2 -> 3, etc.
                break # Stop after first match

    return blocks


