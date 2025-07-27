import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb  # Changed from lightgbm to xgboost
from sklearn.metrics import classification_report

# # ---------------------------------------------
# # 1. Load Data
# # ---------------------------------------------
# # It's recommended to handle potential file not found errors
# try:
#     with open('/content/real_output.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     all_lines = [line for document in data for line in document]
#     df = pd.DataFrame(all_lines)
#     print(f"📄 Data loaded successfully. Total lines: {len(df)}")
# except FileNotFoundError:
#     print("❌ Error: '/content/real_output.json' not found. Please check the file path.")


# ---------------------------------------------
# 2. Embed Text and Combine Features
# ---------------------------------------------
print("\n⚙️ Generating multilingual text embeddings... This may take a moment.")

# Load a powerful pre-trained MULTILINGUAL model
# This model is great for handling text from various languages
embedding_model = SentenceTransformer(r"models--sentence-transformers--all-MiniLM-L6-v2")

# Convert the 'text' column into numerical embeddings
# This process turns text into a format the model can understand
text_embeddings = embedding_model.encode(df_balanced['text'].tolist(), show_progress_bar=True)

# Create a new DataFrame for the embeddings
embeddings_df = pd.DataFrame(text_embeddings, index=df_balanced.index)
embeddings_df = embeddings_df.add_prefix('embed_')

print(f"Embeddings generated with shape: {embeddings_df.shape}")

# Get the original tabular features (all columns except 'text' and 'label')
X_original = df_balanced.drop(columns=['text', 'label','id','word_count','page'])

# Combine original features with the new text embeddings
X_enriched = pd.concat([X_original, embeddings_df], axis=1)

print(f"Enriched feature set created with shape: {X_enriched.shape}")

# Define the target variable we want to predict
y = df_balanced['label']


# ---------------------------------------------
# 3. Split Data and Train Model
# ---------------------------------------------
# Splitting data into training and testing sets
# stratify=y ensures the distribution of labels is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_enriched, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# Train an XGBoost model on the new enriched data
print("\n🚀 Training XGBoost on multilingual enriched data...")
# Using XGBClassifier for multi-class classification
xgb_enriched_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(y.unique()),
    random_state=42,
    use_label_encoder=False, # Suppress a potential future warning
    eval_metric='mlogloss'    # A common metric for multiclass classification
)
xgb_enriched_model.fit(X_train, y_train)


# ---------------------------------------------
# 4. Evaluate the New Model
# ---------------------------------------------
print("\n📊 Classification Report for XGBoost with Multilingual Text Embeddings:\n")
y_pred_enriched = xgb_enriched_model.predict(X_test)
print(classification_report(y_test, y_pred_enriched, digits=3))



# ---------------------------------------------
# 5. Save the Model
# ---------------------------------------------
# Save the trained model to a file for later use
model_filename = 'xgb_enriched_model.json'
xgb_enriched_model.save_model(model_filename)
print(f"\n💾 Model saved successfully to '{model_filename}'")

# To load the model later, you can use:



