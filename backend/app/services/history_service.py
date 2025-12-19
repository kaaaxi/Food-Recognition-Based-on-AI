from datetime import datetime
from typing import Dict, List, Optional

from .supabase_service import SupabaseService


class HistoryService:
    def __init__(self, supabase: SupabaseService) -> None:
        self.supabase = supabase

    def save_record(self, payload: Dict) -> Dict:
        record = payload.copy()
        record.setdefault("created_at", datetime.utcnow().isoformat())
        return self.supabase.insert_history(record) or record

    def update_record(self, record_id: str, updates: Dict) -> Optional[Dict]:
        return self.supabase.update_history(record_id, updates)

    def delete_record(self, record_id: str, user_id: str) -> bool:
        return self.supabase.delete_history_record(record_id, user_id)

    def list_records(
        self,
        user_id: Optional[str],
        keyword: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        limit: Optional[int],
    ) -> List[Dict]:
        return self.supabase.fetch_history(
            user_id=user_id,
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
