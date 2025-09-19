# notices/service.py
from typing import Dict, Any, Optional, Tuple, List
from . import dao

def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "y", "yes")
    return False

def parse_notice_create(body: Dict[str, Any]) -> Dict[str, Any]:
    title = (body.get("title") or "").strip()
    content = (body.get("content") or "").strip()
    is_pinned = _bool(body.get("is_pinned", False))
    if not title:
        raise ValueError("title is required.")
    if len(title) > 255:
        raise ValueError("title must be <= 255 chars.")
    if not content:
        raise ValueError("content is required.")
    return {"title": title, "content": content, "is_pinned": is_pinned}

def parse_notice_update(body: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "title" in body:
        title = (body.get("title") or "").strip()
        if not title:
            raise ValueError("title cannot be empty.")
        if len(title) > 255:
            raise ValueError("title must be <= 255 chars.")
        out["title"] = title
    if "content" in body:
        content = (body.get("content") or "").strip()
        if not content:
            raise ValueError("content cannot be empty.")
        out["content"] = content
    if "is_pinned" in body:
        out["is_pinned"] = _bool(body.get("is_pinned"))
    if not out:
        raise ValueError("no updatable fields.")
    return out

def list_notices(conn, q: Optional[str], page: int, page_size: int) -> Tuple[List[Dict[str, Any]], int]:
    return dao.list_notices(conn, q, page, page_size)

def get_notice(conn, nid: int) -> Optional[Dict[str, Any]]:
    return dao.get_notice(conn, nid)

def create_notice(conn, n: Dict[str, Any], admin_user_id: int) -> int:
    return dao.insert_notice(conn, n["title"], n["content"], n["is_pinned"], admin_user_id)

def update_notice(conn, nid: int, u: Dict[str, Any]) -> bool:
    return dao.update_notice(conn, nid, u)

def delete_notice(conn, nid: int) -> bool:
    return dao.delete_notice(conn, nid)
