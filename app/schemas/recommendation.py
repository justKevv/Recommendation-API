from pydantic import BaseModel
from typing import List, Optional

class ProfileRequest(BaseModel):
    profile_text: str

class InternshipItem(BaseModel):
    id: int
    internship_text: str
    location: str

class RecommendationRequest(BaseModel):
    profile_text: str
    predicted_category: Optional[str] = None
    preferred_location: str
    internships: List[InternshipItem]
