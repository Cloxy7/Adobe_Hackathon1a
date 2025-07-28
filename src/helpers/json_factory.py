import json
import os


class JSONFactory:

    def __init__(self, pdf_path, corrected_lines,output_dir):
        self.pdf_path = pdf_path
        self.output_json_data = {
                                    "title": "",
                                    "outline": []
                                }
        self.corrected_lines = corrected_lines
        self.output_dir = output_dir

    def generate_json_output(self):
        level_map_text = {1: "TITLE", 2: "H1", 3: "H2", 4: "H3"} # Map numeric labels to text levels

        found_title_for_json_output = False # New flag to ensure only one title for JSON 'title' field

        for line in self.corrected_lines:
            predicted_label = line['predicted_label']
            
            if predicted_label == 1:
                if not found_title_for_json_output:
                    self.output_json_data["title"] = line['text']
                    found_title_for_json_output = True
                else:
                    # If subsequent label 1s are encountered (after first true title), add them as H1s to outline
                    self.output_json_data["outline"].append({
                        "level": "H1", 
                        "text": line['text'],
                        "page": line['page'] # KEEP 0-INDEXED PAGE
                    })
            elif predicted_label > 1: # H1, H2, H3 (labels 2, 3, 4) go directly into the outline
                # Ensure the level exists in map, fallback to a generic Hx if not
                level_text = level_map_text.get(predicted_label, f"H{predicted_label-1}") # predicted_label 2 means H1
                self.output_json_data["outline"].append({
                    "level": level_text,
                    "text": line['text'],
                    "page": line['page'] # KEEP 0-INDEXED PAGE
                })

        # --- Save JSON Output to File ---
        output_file_name = os.path.splitext(os.path.basename(self.pdf_path))[0] + "_outline.json"
        output_file_path = os.path.join(self.output_dir, output_file_name)

        os.makedirs(self.output_dir, exist_ok=True) # Ensure the output directory exists

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.output_json_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Structured outline saved to: {output_file_path}")
        print("\n--- Generated Outline Preview ---")
