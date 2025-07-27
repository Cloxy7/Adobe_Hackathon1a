import pymupdf  
import re
import uuid
import warnings

warnings.filterwarnings('ignore')


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
                "word_count": len(block_text.split()), 
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

    doc.close()
    return blocks_data



