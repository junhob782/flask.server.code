from flask import request
from .db import get_db

class AuthzError(Exception):
    pass

def require_admin():
    """
    헤더 X-User-Id 로 사용자 식별 → user.user_role='admin' 확인.
    성공 시 {'user_id': ..., 'user_role': ...} 반환, 실패 시 AuthzError 발생.
    """
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        raise AuthzError("X-User-Id header is required.")

    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT user_id, user_role FROM user WHERE user_id=%s", (user_id,))
        row = cur.fetchone()

    if not row:
        raise AuthzError("User not found.")
    if (row.get("user_role") or "").lower() != "admin":
        raise AuthzError("Admin only.")
    return row
