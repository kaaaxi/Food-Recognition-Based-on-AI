from datetime import datetime
from typing import Optional

from fastapi import HTTPException, UploadFile, status

from ..models.schemas import AnalysisResponse, ManualCorrection
from ..services.ai_pipeline import AIPipeline
from ..services.history_service import HistoryService


class AnalysisController:
    def __init__(self, pipeline: AIPipeline, history_service: HistoryService) -> None:
        self.pipeline = pipeline
        self.history_service = history_service

    async def analyze(self, file: UploadFile, user_id: Optional[str]) -> AnalysisResponse:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
        result_dict = self.pipeline.analyze_image(image_bytes, file.filename, user_id)
        
        # Only save to history if user is authenticated
        record_id = None
        if user_id:
            saved = self.history_service.save_record(
                {
                    "user_id": user_id,
                    "dish_name": result_dict["result"]["dish_name"],
                    "calories": result_dict["result"]["calories"],
                    "protein": result_dict["result"]["protein"],
                    "fat": result_dict["result"]["fat"],
                    "carbs": result_dict["result"]["carbs"],
                    "portion_grams": result_dict["result"]["portion_grams"],
                    "confidence": result_dict["result"]["confidence"],
                    "created_at": datetime.utcnow().isoformat(),
                    "raw_payload": result_dict["result"],
                }
            )
            record_id = saved.get("id") if isinstance(saved, dict) else None
        
        result_dict["id"] = record_id
        return AnalysisResponse(**result_dict)

    def manual_correction(self, payload: ManualCorrection) -> AnalysisResponse:
        """User corrects dish name, system calls LLM for nutrition analysis."""
        # Call LLM to get nutrition data for the user-provided dish name
        portion = payload.portion_grams or 150.0
        nutrition = self.pipeline._call_llm(payload.dish_name, portion)
        
        # Only save to history if user is authenticated
        record_id = None
        if payload.user_id:
            record = self.history_service.save_record(
                {
                    "user_id": payload.user_id,
                    "dish_name": payload.dish_name,
                    "calories": nutrition.get("calories", 0),
                    "protein": nutrition.get("protein", 0),
                    "fat": nutrition.get("fat", 0),
                    "carbs": nutrition.get("carbs", 0),
                    "portion_grams": portion,
                    "confidence": 1.0,  # User confirmed dish name
                    "created_at": datetime.utcnow().isoformat(),
                    "raw_payload": {"user_input": payload.dish_name, "notes": payload.notes, **nutrition},
                }
            )
            record_id = record.get("id") if isinstance(record, dict) else None
        
        response = {
            "id": record_id,
            "user_id": payload.user_id,
            "created_at": datetime.utcnow(),
            "result": {
                "dish_name": payload.dish_name,
                "calories": nutrition.get("calories", 0),
                "protein": nutrition.get("protein", 0),
                "fat": nutrition.get("fat", 0),
                "carbs": nutrition.get("carbs", 0),
                "portion_grams": portion,
                "confidence": 1.0,
                "suggestions": nutrition.get("suggestions", []),
                "breakdown": payload.model_dump(),
                "alternatives": None,
                "meal_pairing": None,
            },
        }
        return AnalysisResponse(**response)
