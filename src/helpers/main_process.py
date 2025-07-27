import os
import sys
import xgboost as xgb
import pandas as pd
from save_sbert_model import local_sbert_model_path
from sentence_transformers import SentenceTransformer
from src.helpers.json_factory import JSONFactory
from src.helpers.post_processing import enforce_document_hierarchy
from src.helpers.pre_processing import extract_blocks_with_features


# --- Load the pre-trained XGBoost model ---
model_filename = 'models/xgb_enriched_model.json'  
loaded_model = xgb.XGBClassifier()
loaded_model.load_model(model_filename)
print("\n✅ Model loaded successfully.")
embedding_model = SentenceTransformer(local_sbert_model_path)



def process_input_directory(input_directory_path,output_directory_path):
    print(f"Scanning directory: {input_directory_path}")
    if not os.path.isdir(input_directory_path):
        print(f"Error: Directory not found at '{input_directory_path}'.", file=sys.stderr)
        print("Please ensure the input volume is correctly mounted to /app/input.", file=sys.stderr)
        sys.exit(1)

    # List all files and directories in the given path.
    try:
        all_entries = os.listdir(input_directory_path)
    except OSError as e:
        print(f"Error: Could not access directory '{input_directory_path}'. Reason: {e}", file=sys.stderr)
        sys.exit(1)

    if not all_entries:
        print("No files found in the input directory.")
        return

    print("Found files. Processing...")
    
    for filename in all_entries:
        # Construct the full path to the file.
        full_path = os.path.join(input_directory_path, filename)
        if os.path.isfile(full_path) and filename.lower().endswith('.pdf'):
            print(f"Processing file: {full_path}")
            # 1. Extract tabular features from the new PDF
            lines_data = extract_blocks_with_features(full_path)
            if not lines_data:
                print("Could not find any text lines in the PDF.")
                exit()

            df_new = pd.DataFrame(lines_data)
            text_embeddings_new2 = embedding_model.encode(df_new['text'].tolist(), show_progress_bar=True)
            embeddings_df_new2 = pd.DataFrame(text_embeddings_new2, index=df_new.index)
            embeddings_df_new2 = embeddings_df_new2.add_prefix('embed_')
            print(f"Embeddings generated with shape: {embeddings_df_new2.shape}")
            X_original_1 = df_new.drop(columns=['text']) 
            X_predict_2 = pd.concat([X_original_1, embeddings_df_new2], axis=1)
            model_features = loaded_model.get_booster().feature_names
            X_predict = X_predict_2.reindex(columns=model_features, fill_value=0)


            print("Predicting heading levels...")
            predictions = loaded_model.predict(X_predict)

            for i, line in enumerate(lines_data):
                line['predicted_label'] = int(predictions[i])

            print("Applying hierarchy rules to correct predictions...")
            corrected_lines = enforce_document_hierarchy(lines_data)
            json_client = JSONFactory(pdf_path=full_path, corrected_lines=corrected_lines, output_dir=output_directory_path)
            json_client.generate_json_output()
            print("✅ Prediction complete.")




        else:
            print(f"Skipping non-PDF or directory entry: {filename}")



