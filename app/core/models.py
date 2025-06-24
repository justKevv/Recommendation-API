import joblib
from sentence_transformers import SentenceTransformer
import os
import requests
from huggingface_hub import snapshot_download # <-- Add this import

TEMP_DIR = "/tmp"

# This function for your .pkl files is still correct and needed
def download_and_load_model(model_url, model_filename):
    # ... (no changes needed in this function)
    local_path = os.path.join(TEMP_DIR, model_filename)
    if not os.path.exists(local_path):
        print(f"Downloading model from {model_url} to {local_path}...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download model {model_filename}. Error: {e}")
            return None
    try:
        return joblib.load(local_path)
    except Exception as e:
        print(f"Failed to load model {local_path}. Error: {e}")
        return None

# --- NEW FUNCTION FOR THE SENTENCE TRANSFORMER ---
def get_sentence_transformer(model_name='all-MiniLM-L6-v2'):
    """
    Downloads the SentenceTransformer model to /tmp if it doesn't exist,
    then loads it from there.
    """
    local_model_path = os.path.join(TEMP_DIR, model_name)

    if not os.path.exists(local_model_path):
        print(f"Downloading SentenceTransformer model '{model_name}' to {local_model_path}...")
        # Use snapshot_download to get all files for the model from Hugging Face
        try:
            snapshot_download(repo_id=f"sentence-transformers/{model_name}",
                              local_dir=local_model_path,
                              local_dir_use_symlinks=False) # This is important for Vercel
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download SentenceTransformer model. Error: {e}")
            return None

    # Load the model from the local path in /tmp
    try:
        print(f"Loading SentenceTransformer model from {local_model_path}...")
        return SentenceTransformer(local_model_path)
    except Exception as e:
        print(f"Failed to load SentenceTransformer model from {local_model_path}. Error: {e}")
        return None

# --- Main Model Loading Logic ---
try:
    # Your .pkl model loading remains the same
    TFIDF_URL = "https://pub-4a389a9b2dc842a2a55678d2db0ec0c6.r2.dev/tfidf_vectorizer.pkl"
    LE_URL = "https://pub-4a389a9b2dc842a2a55678d2db0ec0c6.r2.dev/label_encoder.pkl"
    RF_MODEL_URL = "https://pub-4a389a9b2dc842a2a55678d2db0ec0c6.r2.dev/random_forest_model.pkl"

    tfidf_vectorizer = download_and_load_model(TFIDF_URL, "tfidf_vectorizer.pkl")
    le = download_and_load_model(LE_URL, "label_encoder.pkl")
    rf_model = download_and_load_model(RF_MODEL_URL, "random_forest_model.pkl")

    if all([tfidf_vectorizer, le, rf_model]):
        print("Classification models loaded successfully.")
    else:
        print("One or more classification models failed to load.")

    # --- THIS IS THE LINE TO CHANGE ---
    # Old line: sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    # New line:
    sentence_model = get_sentence_transformer()
    if sentence_model:
        print("SentenceTransformer model loaded successfully.")
    else:
        print("SentenceTransformer model failed to load.")

except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
