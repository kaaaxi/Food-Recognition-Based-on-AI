from typing import List, Dict

from ..models.schemas import TDEERequest, TDEEResponse, HealthReportResponse
from ..services.health_service import HealthService


class HealthController:
    def __init__(self, health_service: HealthService) -> None:
        self.health_service = health_service

    def calculate_tdee(self, payload: TDEERequest) -> TDEEResponse:
        return self.health_service.calculate_tdee(
            height_cm=payload.height_cm,
            weight_kg=payload.weight_kg,
            age=payload.age,
            gender=payload.gender,
            activity_level=payload.activity_level,
        )

    def get_health_report(
        self,
        user_id: str,
        period: str = "week",
        target_calories: float = 2000,
    ) -> HealthReportResponse:
        return self.health_service.generate_health_report(
            user_id=user_id,
            period=period,
            target_calories=target_calories,
        )

    def get_alternatives(self, dish_name: str) -> List[Dict]:
        return self.health_service.get_healthy_alternatives(dish_name)
