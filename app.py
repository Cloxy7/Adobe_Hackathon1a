import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import warnings
import torch
import os

from extrationAndLabel import extract_blocks_with_features
# Removed unused imports: sklearn.model_selection, sklearn.metrics, huggingface_hub

warnings.filterwarnings('ignore')


model_filename = 'xgb_enriched_model.json'  
loaded_model = xgb.XGBClassifier()
loaded_model.load_model(model_filename)
print("\n✅ Model loaded successfully.")


pdf_path = r"L:\Adobehack_1A\Input\sample.pdf"

# --- MODIFIED PART: Loading SentenceTransformer model from local project directory ---
# Define the path where you saved the SentenceTransformer model
local_sbert_model_path = os.path.join(os.getcwd(), 'sbert_model') # Points to .\sbert_model

if not os.path.exists(local_sbert_model_path):
    print(f"Error: Local SentenceTransformer model not found at '{local_sbert_model_path}'.")
    print("Please run 'save_sbert_model.py' first to download and prepare the model.")
    exit()

# Load the SentenceTransformer model directly from the local folder
embedding_model = SentenceTransformer(local_sbert_model_path)
print("✅ Embedding model loaded successfully from local directory.")
# --- END OF MODIFIED PART ---

print(f"Processing PDF: {pdf_path}...")

# 1. Extract tabular features from the new PDF
lines_data = extract_blocks_with_features(pdf_path)
if not lines_data:
    print("Could not find any text lines in the PDF.")

df_new = pd.DataFrame(lines_data)

df_new.to_csv('pdf.csv', index=False)
text_embeddings_new2 = embedding_model.encode(df_new['text'].tolist(), show_progress_bar=True)

# Create a new DataFrame for the embeddings
embeddings_df_new2 = pd.DataFrame(text_embeddings_new2, index=df_new.index)
embeddings_df_new2 = embeddings_df_new2.add_prefix('embed_')

print(f"Embeddings generated with shape: {embeddings_df_new2.shape}")

# 3. Combine tabular features and embeddings
# Drop the same columns that were dropped during final training
X_original_1 = df_new.drop(columns=['text','id','word_count','page'])
X_predict_2 = pd.concat([X_original_1, embeddings_df_new2], axis=1)

# 4. Ensure column order matches the model's training data
# This is a crucial step to prevent errors
model_features = loaded_model.get_booster().feature_names

X_predict = X_predict_2[model_features]

def enforce_document_hierarchy(lines_data):
    """
    Corrects the predicted labels based on logical document structure rules.
    """
    if not lines_data:
        return []

    last_heading_level = 0
    has_title = False

    for i, line in enumerate(lines_data):
        current_label = line['predicted_label']

        # Rule 1: Demote long text blocks predicted as headings.
        # Headings are short. If it's long, it's a paragraph.
        if current_label > 0 and line['word_count'] > 6:
            current_label = 0

        # Rule 2: Enforce a single title.
        # Only the first "TITLE" (label 1) is kept. Others are demoted to H1.
        if current_label == 1:
            if has_title:
                current_label = 2  # Demote to H1
            else:
                has_title = True

        # Rule 3: Enforce logical heading hierarchy.
        # An H3 can't follow an H1. It must be an H2.
        if current_label > 1: # If it's a heading (H1, H2, etc.)
            # A heading level can't jump by more than 1 from the previous heading
            if current_label > last_heading_level + 1:
                current_label = last_heading_level + 1
            last_heading_level = current_label

        # If the current block is a paragraph, reset the heading level tracker
        # This allows a new H1 to start a new section.
        elif current_label == 0:
            # We don't reset to 0 because the next heading could be an H2
            # continuing the previous section. A full reset is too simple.
            pass

        lines_data[i]['predicted_label'] = current_label

    return lines_data

print("Predicting heading levels...")
predictions = loaded_model.predict(X_predict)

# 6. Add predictions back to the original data for easy review
for i, line in enumerate(lines_data):
    line['predicted_label'] = int(predictions[i])

print("Applying hierarchy rules to correct predictions...")
corrected_lines = enforce_document_hierarchy(lines_data)

print("✅ Prediction complete.")

# 4. Display the results
print("\n--- Predicted Headings ---")
if corrected_lines:
    for line in corrected_lines:
        # Only print lines that were predicted as some type of heading (label > 0)
        if line['predicted_label'] > 0:
            # Map label to a more readable format
            level_map = {1: "TITLE", 2: "H1", 3: "H2", 4: "H3"}
            heading_type = level_map.get(line['predicted_label'], f"H{line['predicted_label']}")
            print(f"[{heading_type}]\t {line['text']}")