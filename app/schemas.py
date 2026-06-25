from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskPredictionRequest(BaseModel):
    current_gba: float = Field(..., ge=0)
    total_credit_hours: float = Field(..., ge=0)
    semester: int = Field(..., ge=0)
    avg_grade: float = Field(..., ge=0)
    failed_courses_count: int = Field(..., ge=0)
    in_progress_count: int = Field(..., ge=0)
    total_courses: int = Field(..., ge=0)


class RiskPredictionResponse(BaseModel):
    at_risk: bool
    prediction: int
    risk_probability: Optional[float] = None
    probabilities: Dict[str, float]


class GbaClassificationRequest(BaseModel):
    total_credit_hours: float = Field(..., ge=0)
    semester: int = Field(..., ge=0)
    avg_grade: float = Field(..., ge=0)
    failed_courses_count: int = Field(..., ge=0)
    total_courses: int = Field(..., ge=0)
    current_gba: Optional[float] = Field(default=None, ge=0)


class GbaClassificationResponse(BaseModel):
    model_prediction: str
    rule_based_level: Optional[str] = None
    probabilities: Dict[str, float]

class CourseGrade(BaseModel):
    course_id: str
    grade: float

class TrackRecommendationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "courses": [
                    {
                        "course_id": "CS-210",
                        "grade": 90
                    },
                    {
                        "course_id": "CS-310",
                        "grade": 80
                    }
                ]
            }
        }
    )

    courses: List[CourseGrade]

class TrackProbability(BaseModel):
    track_id: Optional[int] = None
    track_name: Optional[str] = None
    probability: float


class TrackRecommendationResponse(BaseModel):
    recommended_track_id: Optional[int] = None
    recommended_track_name: str
    probabilities: List[TrackProbability]


class ModelsMetadataResponse(BaseModel):
    risk_features: List[str]
    gba_features: List[str]
    track_features: List[str]

