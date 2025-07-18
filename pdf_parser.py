import fitz  # PyMuPDF

def extract_lines_with_features(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        prev_bottom = 0

        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    spans = l["spans"]
                    line_text = " ".join([s["text"] for s in spans]).strip()
                    if not line_text or len(line_text.split()) > 15:
                        continue

                    font_size = max(s["size"] for s in spans)
                    font_flags = [s["flags"] for s in spans]
                    bbox = spans[0]["bbox"]
                    y0 = bbox[1]
                    x0 = bbox[0]

                    line = {
                        "text": line_text,
                        "font_size": font_size,
                        "font_flags": font_flags,
                        "is_bold": any(f & 2 for f in font_flags),
                        "is_all_caps": line_text.upper() == line_text,
                        "line_length": len(line_text.split()),
                        "ends_with_dot": line_text.endswith("."),
                        "line_gap": y0 - prev_bottom,
                        "x_position": x0,
                        "page": page_num
                    }

                    lines.append(line)
                    prev_bottom = bbox[3]

    return lines
