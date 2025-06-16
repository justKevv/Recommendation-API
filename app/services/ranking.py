from geopy.distance import geodesic
import json
import os
from geopy.geocoders import Nominatim

from ..core.models import tfidf_vectorizer, le, rf_model, sentence_model
from ..schemas.recommendation import RecommendationRequest
from ..utils.text import clean_resume

geolocator = Nominatim(user_agent="student_recommendation_api_v1")

GEO_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "GEO_CACHE.txt")
GEO_CACHE = {}

def load_geo_cache():
    global GEO_CACHE
    if os.path.exists(GEO_CACHE_FILE):
        with open(GEO_CACHE_FILE, "r") as f:
            try:
                GEO_CACHE = json.load(f)
            except json.JSONDecodeError:
                GEO_CACHE = {}

def save_geo_cache():
    with open(GEO_CACHE_FILE, "w") as f:
        json.dump(GEO_CACHE, f)

load_geo_cache()

def geo_coords(city_name: str) -> tuple | None:
    """
    Geocodes a city name to (latitude, longitude).
    Uses an in-memory cache to avoid repeated API calls.
    """
    city_name = city_name.lower().strip()
    if city_name in GEO_CACHE:
        return GEO_CACHE[city_name]
    try:
        print(f"--- Geocoding and caching new city: {city_name} ---")
        location = geolocator.geocode(f"{city_name}, Indonesia")

        if location:
            coords = (location.latitude, location.longitude)
            GEO_CACHE[city_name] = coords
            save_geo_cache()
            return coords
        else:
            print(f"Location not found for {city_name}")
            GEO_CACHE[city_name] = None
            save_geo_cache()
            return None
    except Exception as e:
        print(f"Error geocoding {city_name}: {e}")
        return None

def get_category_prediction(profile_text: str) -> str:
    """Processes text and predicts the job category."""
    cleaned_text = profile_text.lower()
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    prediction_encoded = rf_model.predict(vectorized_text)[0]
    category = le.inverse_transform([prediction_encoded])[0]
    return category

def  get_ranked_internships(request: RecommendationRequest) -> list[int]:
    """Performs two-stage ranking with dynamic geocoding."""

    profile_text_to_encode = request.profile_text

    if request.predicted_category:
        profile_text_to_encode = f"The user's predicted job category is {request.predicted_category}. Based on that, consider their profile: {request.profile_text}"

    profile_embedding = sentence_model.encode(profile_text_to_encode)
    internship_texts = [internship.internship_text for internship in request.internships]

    if not internship_texts:
        return []

    internship_embeddings = sentence_model.encode(internship_texts)
    cosine_score = sentence_model.similarity(profile_embedding, internship_embeddings)[0].tolist()

    print("--- FastAPI Debugging ---")
    print(f"Received {len(internship_texts)} internships to rank.")
    print(f"Calculated Cosine Scores: {cosine_score}")
    print("--------------------------")

    ranked_by_similarity = []
    for i, internship in enumerate(request.internships):
        ranked_by_similarity.append({
            "id": internship.id,
            "similarity_score": cosine_score[i],
            "location": internship.location,
        })

    final_ranked_list = []
    user_coords = geo_coords(request.preferred_location)

    print(user_coords, request.preferred_location)


    for internship in ranked_by_similarity:
        final_score = internship['similarity_score']

        if user_coords:
            internship_coords = geo_coords(internship['location'])
            if internship_coords:
                distance_km = geodesic(user_coords, internship_coords).kilometers
                if distance_km < 1:
                    final_score += 2.0
                elif distance_km < 150:
                    final_score += 0.75

        internship['final_score'] = final_score
        final_ranked_list.append(internship)

    final_ranked_list.sort(key=lambda x: x['final_score'], reverse=True)

    final_ids = [item['id'] for item in final_ranked_list]

    print(final_ranked_list)

    return final_ids
