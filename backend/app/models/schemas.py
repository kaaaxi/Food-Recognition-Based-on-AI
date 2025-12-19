from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class UserProfile(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    activity_level: Optional[str] = None


class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    activity_level: Optional[str] = None


class AnalysisResult(BaseModel):
    dish_name: str
    calories: float
    protein: float
    fat: float
    carbs: float
    portion_grams: float
    confidence: float
    suggestions: list[str]
    breakdown: dict
    alternatives: Optional[list[str]] = None
    meal_pairing: Optional[list[str]] = None


class AnalysisResponse(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime
    result: AnalysisResult


class ManualCorrection(BaseModel):
    """User provides dish name, system calls LLM for nutrition analysis."""
    user_id: Optional[str] = None
    dish_name: str
    portion_grams: Optional[float] = 150.0  # Default portion size
    notes: Optional[str] = None  # Optional dietary preferences/context


class HistoryQuery(BaseModel):
    user_id: Optional[str] = None
    keyword: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = None


class HistoryUpdateRequest(BaseModel):
    dish_name: Optional[str] = None
    calories: Optional[float] = None
    protein: Optional[float] = None
    fat: Optional[float] = None
    carbs: Optional[float] = None
    portion_grams: Optional[float] = None


class TDEERequest(BaseModel):
    height_cm: float
    weight_kg: float
    age: int
    gender: str
    activity_level: str


class TDEEResponse(BaseModel):
    bmr: float
    tdee: float
    daily_protein: float
    daily_fat: float
    daily_carbs: float
    daily_fiber: float


class HealthReportResponse(BaseModel):
    user_id: Optional[str] = None
    period: str
    total_calories: float
    avg_calories: float
    total_protein: float
    total_fat: float
    total_carbs: float
    meals_count: int
    recommendations: list[str]
    goal_progress: dict
