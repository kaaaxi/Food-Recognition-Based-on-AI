from datetime import datetime

from fastapi import APIRouter, Depends, File, UploadFile

from ..controllers.analysis_controller import AnalysisController
from ..controllers.auth_controller import AuthController
from ..controllers.health_controller import HealthController
from ..controllers.history_controller import HistoryController
from ..models.schemas import (
    HistoryQuery,
    HistoryUpdateRequest,
    LoginRequest,
    ManualCorrection,
    RefreshRequest,
    RegisterRequest,
    TDEERequest,
    UserProfileUpdate,
)
from ..services.ai_pipeline import AIPipeline
from ..services.auth_service import AuthService
from ..services.health_service import HealthService
from ..services.history_service import HistoryService
from ..services.supabase_service import SupabaseService
from .deps import get_current_user, get_optional_user


def get_supabase_service() -> SupabaseService:
    return SupabaseService()


def get_auth_service(supabase: SupabaseService = Depends(get_supabase_service)) -> AuthService:
    return AuthService(supabase)


def get_history_service(supabase: SupabaseService = Depends(get_supabase_service)) -> HistoryService:
    return HistoryService(supabase)


def get_health_service(history_service: HistoryService = Depends(get_history_service)) -> HealthService:
    return HealthService(history_service)


_pipeline_instance = None


def get_pipeline() -> AIPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AIPipeline()
    return _pipeline_instance


router = APIRouter(prefix="/api")


@router.post("/auth/register")
def register_user(payload: RegisterRequest, auth_service: AuthService = Depends(get_auth_service)):
    controller = AuthController(auth_service)
    return controller.register(payload)


@router.post("/auth/login")
def login_user(payload: LoginRequest, auth_service: AuthService = Depends(get_auth_service)):
    controller = AuthController(auth_service)
    return controller.login(payload)


@router.post("/auth/refresh")
def refresh_token(payload: RefreshRequest, auth_service: AuthService = Depends(get_auth_service)):
    controller = AuthController(auth_service)
    return controller.refresh(payload)


@router.get("/auth/me")
def get_current_user_info(
    user=Depends(get_current_user),
    supabase: SupabaseService = Depends(get_supabase_service),
):
    user_id = str(user.get("sub"))
    user_data = supabase.find_user_by_id(user_id)
    if not user_data:
        return {"user_id": user_id, "email": user.get("email")}
    return {
        "user_id": user_id,
        "email": user_data.get("email"),
        "name": user_data.get("name"),
        "height_cm": user_data.get("height_cm"),
        "weight_kg": user_data.get("weight_kg"),
        "age": user_data.get("age"),
        "gender": user_data.get("gender"),
        "activity_level": user_data.get("activity_level"),
    }


@router.patch("/auth/profile")
def update_user_profile(
    payload: UserProfileUpdate,
    user=Depends(get_current_user),
    supabase: SupabaseService = Depends(get_supabase_service),
):
    user_id = str(user.get("sub"))
    update_data = {k: v for k, v in payload.model_dump().items() if v is not None}
    result = supabase.update_user_profile(user_id, update_data)
    return result or {"status": "updated"}


@router.delete("/auth/user/{user_id}")
def delete_user_data(user_id: str, auth_service: AuthService = Depends(get_auth_service)):
    controller = AuthController(auth_service)
    controller.delete_user_data(user_id)
    return {"status": "deleted"}


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    pipeline: AIPipeline = Depends(get_pipeline),
    history_service: HistoryService = Depends(get_history_service),
    user=Depends(get_optional_user),
):
    user_id = str(user.get("sub")) if user else None
    controller = AnalysisController(pipeline, history_service)
    return await controller.analyze(file, user_id)


@router.post("/manual")
def manual_override(
    payload: ManualCorrection,
    history_service: HistoryService = Depends(get_history_service),
    pipeline: AIPipeline = Depends(get_pipeline),
    user=Depends(get_optional_user),
):
    payload.user_id = str(user.get("sub")) if user else None
    controller = AnalysisController(pipeline, history_service)
    return controller.manual_correction(payload)


@router.get("/history")
def read_history(
    keyword: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = 50,
    history_service: HistoryService = Depends(get_history_service),
    user=Depends(get_current_user),
):
    query = HistoryQuery(
        user_id=str(user.get("sub")) if user else None,
        keyword=keyword,
        start_date=datetime.fromisoformat(start_date) if start_date else None,
        end_date=datetime.fromisoformat(end_date) if end_date else None,
        limit=limit,
    )
    controller = HistoryController(history_service)
    return controller.list_history(query)


@router.patch("/history/{record_id}")
def update_history_record(
    record_id: str,
    payload: HistoryUpdateRequest,
    history_service: HistoryService = Depends(get_history_service),
    user=Depends(get_current_user),
):
    user_id = str(user.get("sub")) if user else None
    controller = HistoryController(history_service)
    return controller.update_history(record_id, user_id, payload)


@router.delete("/history/{record_id}")
def delete_history_record(
    record_id: str,
    history_service: HistoryService = Depends(get_history_service),
    user=Depends(get_current_user),
):
    user_id = str(user.get("sub")) if user else None
    controller = HistoryController(history_service)
    return controller.delete_history(record_id, user_id)


@router.post("/health/tdee")
def calculate_tdee(
    payload: TDEERequest,
    health_service: HealthService = Depends(get_health_service),
):
    controller = HealthController(health_service)
    return controller.calculate_tdee(payload)


@router.get("/health/report")
def get_health_report(
    period: str = "week",
    target_calories: float = 2000,
    health_service: HealthService = Depends(get_health_service),
    user=Depends(get_current_user),
):
    user_id = str(user.get("sub")) if user else None
    controller = HealthController(health_service)
    return controller.get_health_report(user_id, period, target_calories)


@router.get("/health/alternatives")
def get_healthy_alternatives(
    dish_name: str,
    health_service: HealthService = Depends(get_health_service),
):
    controller = HealthController(health_service)
    return controller.get_alternatives(dish_name)
