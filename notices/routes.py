# notices/routes.py
from flask import Blueprint, request, jsonify
from utils.db import get_db
from utils.authz import require_admin, AuthzError
from .service import (
    parse_notice_create, parse_notice_update,
    list_notices as svc_list, get_notice as svc_get,
    create_notice as svc_create, update_notice as svc_update, delete_notice as svc_delete
)

bp = Blueprint("notices", __name__)

# 공개 목록
@bp.get("/api/notices")
def public_list_notices():
    db = get_db()
    q = request.args.get("q")
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 10)), 1), 100)
    rows, total = svc_list(db, q, page, page_size)
    return jsonify({"items": rows, "total": total, "page": page, "page_size": page_size})

# 공개 단건
@bp.get("/api/notices/<int:nid>")
def public_get_notice(nid: int):
    db = get_db()
    row = svc_get(db, nid)
    if not row:
        return jsonify({"error": "Notice not found"}), 404
    return jsonify(row)

# 관리자 목록
@bp.get("/api/admin/notices")
def admin_list_notices():
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    q = request.args.get("q")
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 10)), 1), 100)
    rows, total = svc_list(db, q, page, page_size)
    return jsonify({"items": rows, "total": total, "page": page, "page_size": page_size})

# 생성
@bp.post("/api/admin/notices")
def admin_create_notice():
    try:
        admin = require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    body = request.get_json(silent=True) or {}
    try:
        n = parse_notice_create(body)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    nid = svc_create(db, n, admin_user_id=admin["user_id"])
    return jsonify({"id": nid, "message": "created"}), 201

# 수정
@bp.put("/api/admin/notices/<int:nid>")
def admin_update_notice(nid: int):
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    body = request.get_json(silent=True) or {}
    try:
        u = parse_notice_update(body)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    ok = svc_update(db, nid, u)
    if not ok:
        return jsonify({"error": "Notice not found or nothing to update"}), 404
    return jsonify({"message": "updated"})

# 삭제
@bp.delete("/api/admin/notices/<int:nid>")
def admin_delete_notice(nid: int):
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    ok = svc_delete(db, nid)
    if not ok:
        return jsonify({"error": "Notice not found"}), 404
    return jsonify({"message": "deleted"})
