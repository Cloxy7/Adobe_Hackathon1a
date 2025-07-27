import os
from sentence_transformers import SentenceTransformer

# Define the path where you want to save the model locally
# This will be inside your L:\Adobehack_1A\ directory, in a 'sbert_model' folder
local_model_path = os.path.join(os.getcwd(), 'sbert_model') # This creates .\sbert_model

print(f"Attempting to download and save 'all-MiniLM-L6-v2' to: {local_model_path}")

try:
    # Initialize the model (this will download to cache first if not present)
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Save the model to your specified local path
    sbert_model.save_pretrained(local_model_path)
    
    print(f"\n✅ 'all-MiniLM-L6-v2' model successfully downloaded and saved to: {local_model_path}")
    print("You can now zip this 'sbert_model' folder along with your project.")

except Exception as e:
    print(f"\n❌ Error saving SentenceTransformer model: {e}")
    print("Please ensure you have an internet connection for the first run if the model is not cached.")
    print("Also, check write permissions for the target directory.")