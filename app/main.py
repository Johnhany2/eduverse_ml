import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_service import ModelLoadError, ModelService
from app.schemas import (
    GbaClassificationRequest,
    GbaClassificationResponse,
    ModelsMetadataResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
    TrackRecommendationRequest,
    TrackRecommendationResponse,
)


def _allowed_origins() -> list[str]:
    raw_value = os.getenv("ALLOWED_ORIGINS", "*")
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


app = FastAPI(
    title="EduVerse ML API",
    version="1.0.0",
    description="Prediction API for EduVerse student risk, GBA classification, and track recommendation models.",
)

allowed_origins = _allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "EduVerse ML API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_model_service()
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "ok"}


@app.get("/models/metadata", response_model=ModelsMetadataResponse)
def models_metadata() -> dict[str, list[str]]:
    return get_model_service().metadata()


@app.post("/predict/risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest) -> dict:
    return get_model_service().predict_risk(payload.model_dump())


@app.post("/predict/gba-class", response_model=GbaClassificationResponse)
def predict_gba_class(payload: GbaClassificationRequest) -> dict:
    return get_model_service().predict_gba(payload.model_dump())


@app.post("/recommend/track", response_model=TrackRecommendationResponse)
def recommend_track(payload: TrackRecommendationRequest) -> dict:
    return get_model_service().recommend_track(payload.track_scores)
