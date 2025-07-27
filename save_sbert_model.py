import os
from sentence_transformers import SentenceTransformer


local_sbert_model_path = os.path.join(os.getcwd(), 'sbert_model') 



if not os.path.exists(local_sbert_model_path):
    print(f"Attempting to download and save 'all-MiniLM-L6-v2' to: {local_sbert_model_path}")

    try:
        # Initialize the model (this will download to cache first if not present)
        sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Save the model to your specified local path
        sbert_model.save_pretrained(local_sbert_model_path)
        print(f"\n✅ 'all-MiniLM-L6-v2' model successfully downloaded and saved to: {local_sbert_model_path}")
        print("You can now zip this 'sbert_model' folder along with your project.")
        local_sbert_model_path = os.path.join(os.getcwd(), 'sbert_model') 


    except Exception as e:
        print(f"\n❌ Error saving SentenceTransformer model: {e}")
        print("Please ensure you have an internet connection for the first run if the model is not cached.")
        print("Also, check write permissions for the target directory.")