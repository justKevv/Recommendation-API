from fastapi import FastAPI
from .api.routes import recommendations

app = FastAPI(
    title="Student Recommendation API",
    description="An API that uses machine learning to predict job categories and recommend internships.",
    version="1.0.0"
)

app.include_router(recommendations.router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Student Recommendation API"}
