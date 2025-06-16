from fastapi import APIRouter
from ...schemas.recommendation import ProfileRequest, RecommendationRequest
from ...services import ranking

router = APIRouter()

@router.post("/predict-category", tags=["Predictions"])
def predict_category(request: ProfileRequest):
    category = ranking.get_category_prediction(request.profile_text)
    return {"predicted_category": category}

@router.post("/recommend-internships", tags=["Predictions"])
def recommend_internships(request: RecommendationRequest):
    ranked_ids = ranking.get_ranked_internships(request)
    return {"recommendations": ranked_ids}
