import logging
from typing import Any, Dict, List, Optional

import requests

from ..config import settings


logger = logging.getLogger(__name__)


class SupabaseService:
    """Lightweight Supabase client using the REST endpoint to avoid SDK dependency."""

    def __init__(self) -> None:
        self.base_url = f"{settings.supabase_url}/rest/v1"
        self.headers = {
            "apikey": settings.supabase_key or "",
            "Authorization": f"Bearer {settings.supabase_key or ''}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def _request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None, json_body: Any = None
    ) -> Optional[List[Dict[str, Any]]]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method, url, headers=self.headers, params=params, json=json_body, timeout=8
            )
            response.raise_for_status()
            if response.text:
                return response.json()
            return None
        except Exception as exc:
            logger.warning("Supabase request failed: %s", exc)
            return None

    def find_user(self, email: str) -> Optional[Dict[str, Any]]:
        result = self._request("GET", "/users", params={"email": f"eq.{email}"})
        return result[0] if result else None

    def find_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        result = self._request("GET", "/users", params={"id": f"eq.{user_id}"})
        return result[0] if result else None

    def upsert_user(self, email: str, name: str, salt: str, password_hash: str) -> Optional[Dict[str, Any]]:
        payload = {"email": email, "name": name, "salt": salt, "password_hash": password_hash}
        result = self._request("POST", "/users", params={"on_conflict": "email"}, json_body=payload)
        return result[0] if result else None

    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = self._request("PATCH", "/users", params={"id": f"eq.{user_id}"}, json_body=profile_data)
        return result[0] if result else None

    def insert_history(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = self._request("POST", "/analysis_history", json_body=record)
        return result[0] if result else None

    def update_history(self, record_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = self._request("PATCH", "/analysis_history", params={"id": f"eq.{record_id}"}, json_body=updates)
        return result[0] if result else None

    def delete_history_record(self, record_id: str, user_id: str) -> bool:
        result = self._request("DELETE", "/analysis_history", params={"id": f"eq.{record_id}", "user_id": f"eq.{user_id}"})
        return result is not None or result == []

    def fetch_history(
        self,
        user_id: Optional[str] = None,
        keyword: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"order": "created_at.desc"}
        if user_id:
            params["user_id"] = f"eq.{user_id}"
        if keyword:
            params["dish_name"] = f"ilike.%{keyword}%"

        # 日期过滤 - 使用独立的参数而不是 and 语法
        if start_date:
            # 确保 start_date 从当天 00:00:00 开始
            if 'T' not in start_date:
                start_date = f"{start_date}T00:00:00"
            params["created_at"] = f"gte.{start_date}"
        if end_date:
            # 确保 end_date 到当天 23:59:59 结束
            if 'T' not in end_date:
                end_date = f"{end_date}T23:59:59"
            # 如果已经有 created_at 过滤，需要用 and 语法
            if "created_at" in params:
                # 使用 PostgREST 的 and 语法，需要括号
                start_filter = params.pop("created_at")
                params["and"] = f"(created_at.{start_filter},created_at.lte.{end_date})"
            else:
                params["created_at"] = f"lte.{end_date}"
        if limit:
            params["limit"] = limit
        result = self._request("GET", "/analysis_history", params=params)
        return result or []

    def delete_user_data(self, user_id: str) -> None:
        self._request("DELETE", "/analysis_history", params={"user_id": f"eq.{user_id}"})
        self._request("DELETE", "/users", params={"id": f"eq.{user_id}"})
