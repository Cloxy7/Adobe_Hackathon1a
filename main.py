import os
from extract_outline import extract_outline

input_dir = "./input"
output_dir = "./output"

print("Running PDF heading extraction...")

for file in os.listdir(input_dir):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".pdf", ".json"))
        extract_outline(pdf_path, output_path)
        print(f"Saved: {output_path}")
