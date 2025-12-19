from typing import Dict, List, Optional
from datetime import datetime, timedelta

from ..models.schemas import TDEEResponse, HealthReportResponse
from .history_service import HistoryService


class HealthService:
    ACTIVITY_MULTIPLIERS = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }

    def __init__(self, history_service: HistoryService) -> None:
        self.history_service = history_service

    def calculate_tdee(
        self,
        height_cm: float,
        weight_kg: float,
        age: int,
        gender: str,
        activity_level: str,
    ) -> TDEEResponse:
        if gender.lower() in ["male", "m"]:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

        multiplier = self.ACTIVITY_MULTIPLIERS.get(activity_level.lower(), 1.55)
        tdee = bmr * multiplier

        daily_protein = weight_kg * 1.6
        daily_fat = tdee * 0.25 / 9
        daily_carbs = (tdee - daily_protein * 4 - daily_fat * 9) / 4
        daily_fiber = 25 if gender.lower() in ["female", "f"] else 38

        return TDEEResponse(
            bmr=round(bmr, 1),
            tdee=round(tdee, 1),
            daily_protein=round(daily_protein, 1),
            daily_fat=round(daily_fat, 1),
            daily_carbs=round(daily_carbs, 1),
            daily_fiber=round(daily_fiber, 1),
        )

    def generate_health_report(
        self,
        user_id: str,
        period: str = "week",
        target_calories: float = 2000,
    ) -> HealthReportResponse:
        days_map = {"day": 1, "week": 7, "month": 30}
        days = days_map.get(period, 7)
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        records = self.history_service.list_records(
            user_id=user_id,
            keyword=None,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            limit=500,
        )

        total_calories = sum(r.get("calories", 0) for r in records)
        total_protein = sum(r.get("protein", 0) for r in records)
        total_fat = sum(r.get("fat", 0) for r in records)
        total_carbs = sum(r.get("carbs", 0) for r in records)
        meals_count = len(records)
        
        avg_calories = total_calories / max(days, 1)
        
        recommendations = self._generate_recommendations(
            avg_calories, total_protein / max(days, 1),
            total_fat / max(days, 1), total_carbs / max(days, 1),
            target_calories
        )

        goal_progress = {
            "calories_target": target_calories * days,
            "calories_consumed": total_calories,
            "calories_percent": round(total_calories / (target_calories * days) * 100, 1) if target_calories else 0,
        }

        return HealthReportResponse(
            user_id=user_id,
            period=period,
            total_calories=round(total_calories, 1),
            avg_calories=round(avg_calories, 1),
            total_protein=round(total_protein, 1),
            total_fat=round(total_fat, 1),
            total_carbs=round(total_carbs, 1),
            meals_count=meals_count,
            recommendations=recommendations,
            goal_progress=goal_progress,
        )

    def _generate_recommendations(
        self,
        avg_calories: float,
        avg_protein: float,
        avg_fat: float,
        avg_carbs: float,
        target_calories: float,
    ) -> List[str]:
        recommendations = []
        
        if avg_calories > target_calories * 1.1:
            recommendations.append("Your calorie intake is above target. Consider smaller portions or lower-calorie alternatives.")
        elif avg_calories < target_calories * 0.8:
            recommendations.append("Your calorie intake is below target. Ensure adequate nutrition for energy needs.")
        
        protein_ratio = (avg_protein * 4) / max(avg_calories, 1) * 100
        if protein_ratio < 15:
            recommendations.append("Increase protein intake for muscle maintenance. Add lean meats, fish, eggs, or legumes.")
        elif protein_ratio > 35:
            recommendations.append("Protein intake is high. Balance with more vegetables and whole grains.")
        
        fat_ratio = (avg_fat * 9) / max(avg_calories, 1) * 100
        if fat_ratio > 35:
            recommendations.append("Fat intake is elevated. Choose healthier fats from nuts, avocados, and olive oil.")
        
        carb_ratio = (avg_carbs * 4) / max(avg_calories, 1) * 100
        if carb_ratio > 60:
            recommendations.append("Consider reducing refined carbohydrates. Opt for whole grains and fiber-rich options.")
        
        if not recommendations:
            recommendations.append("Your nutrition balance looks good! Keep maintaining these healthy eating habits.")
        
        return recommendations[:3]

    def get_healthy_alternatives(self, dish_name: str) -> List[Dict]:
        alternatives_db = {
            "hamburger": [
                {"name": "Grilled Chicken Sandwich", "calories_saved": 150},
                {"name": "Turkey Burger", "calories_saved": 100},
                {"name": "Veggie Burger", "calories_saved": 200},
            ],
            "pizza": [
                {"name": "Cauliflower Crust Pizza", "calories_saved": 120},
                {"name": "Thin Crust Veggie Pizza", "calories_saved": 80},
                {"name": "Grilled Flatbread with Vegetables", "calories_saved": 150},
            ],
            "french fries": [
                {"name": "Baked Sweet Potato Fries", "calories_saved": 100},
                {"name": "Roasted Vegetables", "calories_saved": 150},
                {"name": "Side Salad", "calories_saved": 200},
            ],
            "fried rice": [
                {"name": "Steamed Brown Rice", "calories_saved": 80},
                {"name": "Cauliflower Rice", "calories_saved": 150},
                {"name": "Quinoa Bowl", "calories_saved": 50},
            ],
            "ice cream": [
                {"name": "Greek Yogurt with Berries", "calories_saved": 100},
                {"name": "Frozen Banana", "calories_saved": 120},
                {"name": "Fruit Sorbet", "calories_saved": 80},
            ],
        }
        
        dish_lower = dish_name.lower()
        for key, alts in alternatives_db.items():
            if key in dish_lower:
                return alts
        
        return [
            {"name": "Grilled Protein with Vegetables", "calories_saved": 100},
            {"name": "Fresh Salad with Lean Protein", "calories_saved": 150},
            {"name": "Steamed Dish with Light Sauce", "calories_saved": 80},
        ]
