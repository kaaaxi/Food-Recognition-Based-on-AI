from typing import Optional, Dict, List

from fastapi import HTTPException, status

from ..models.schemas import HistoryQuery, HistoryUpdateRequest
from ..services.history_service import HistoryService


class HistoryController:
    def __init__(self, history_service: HistoryService) -> None:
        self.history_service = history_service

    def list_history(self, query: HistoryQuery) -> List[Dict]:
        return self.history_service.list_records(
            user_id=query.user_id,
            keyword=query.keyword,
            start_date=query.start_date.isoformat() if query.start_date else None,
            end_date=query.end_date.isoformat() if query.end_date else None,
            limit=query.limit,
        )

    def update_history(self, record_id: str, user_id: str, updates: HistoryUpdateRequest) -> Dict:
        update_data = {k: v for k, v in updates.model_dump().items() if v is not None}
        if not update_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided")
        result = self.history_service.update_record(record_id, update_data)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
        return result

    def delete_history(self, record_id: str, user_id: str) -> Dict:
        success = self.history_service.delete_record(record_id, user_id)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found or unauthorized")
        return {"status": "deleted", "id": record_id}
