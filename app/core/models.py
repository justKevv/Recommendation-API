import joblib
from sentence_transformers import SentenceTransformer
import os

MODEL_DIR = "models"

try:
    tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    print("Classification models loaded.")

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded.")

except FileNotFoundError as e:
    print(f"MODEL LOADING ERROR: {e}")
    print("Make sure the .pkl files are in the 'models' directory.")
    # In a real production app, you might want the app to exit or handle this more gracefully.
    tfidf_vectorizer, le, rf_model, sentence_model = None, None, None, None

except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
