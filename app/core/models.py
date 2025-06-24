import joblib
from sentence_transformers import SentenceTransformer
import os
import requests

TEMP_DIR = "/tmp"

def download_and_load_model(model_url, model_filename):
    """
    Downloads a model from a URL to the /tmp directory if it doesn't exist,
    then loads it using joblib.
    """
    local_path = os.path.join(TEMP_DIR, model_filename)

    # Only download if the model isn't already in the temporary directory
    if not os.path.exists(local_path):
        print(f"Downloading model from {model_url} to {local_path}...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status() # Raises an exception for bad status codes
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download model {model_filename}. Error: {e}")
            return None

    # Load the model from the local path in /tmp
    try:
        return joblib.load(local_path)
    except Exception as e:
        print(f"Failed to load model {local_path}. Error: {e}")
        return None


try:

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

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded.")

except FileNotFoundError as e:
    print(f"MODEL LOADING ERROR: {e}")
    print("Make sure the .pkl files are in the 'models' directory.")
    # In a real production app, you might want the app to exit or handle this more gracefully.
    tfidf_vectorizer, le, rf_model, sentence_model = None, None, None, None

except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
