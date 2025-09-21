# passes/routes.py
from flask import Blueprint, request, jsonify
from utils.db import get_db
from utils.authz import require_admin, AuthzError
from .service import (
    # plans
    parse_plan_create, parse_plan_update,
    list_plans as svc_list_plans, get_plan as svc_get_plan,
    create_plan as svc_create_plan, update_plan as svc_update_plan, delete_plan as svc_delete_plan,
    # passes
    parse_pass_create, parse_pass_update,
    list_passes as svc_list_passes, get_pass as svc_get_pass,
    create_pass as svc_create_pass, update_pass as svc_update_pass, delete_pass as svc_delete_pass,
    is_plate_active as svc_is_plate_active,
    # test purchase
    svc_test_purchase_with_payment,
    insert_payment as svc_insert_payment,
)
from . import dao  # insert_payment 테스트 라우트에서 사용
import urllib.parse  # URL 인코딩 보정용
from datetime import date, datetime

bp = Blueprint("passes", __name__)

# -------- Plans (Admin) ----------
# 목록 조회
@bp.get("/api/admin/passes/plans")
@bp.get("/api/admin/pass-plans")  # alias
def admin_list_plans():
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    q = request.args.get("q")
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 10)), 1), 100)

    rows, total = svc_list_plans(db, q, page, page_size)
    return jsonify({"items": rows, "total": total, "page": page, "page_size": page_size})

# 생성
@bp.post("/api/admin/passes/plans")
@bp.post("/api/admin/pass-plans")  # alias
def admin_create_plan():
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    body = request.get_json(silent=True) or {}
    try:
        p = parse_plan_create(body)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    pid = svc_create_plan(db, p)
    return jsonify({"id": pid, "message": "created"}), 201

# 단건 조회
@bp.get("/api/admin/passes/plans/<int:pid>")
@bp.get("/api/admin/pass-plans/<int:pid>")  # alias
def admin_get_plan(pid: int):
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    row = svc_get_plan(db, pid)
    if not row:
        return jsonify({"error": "Plan not found"}), 404
    return jsonify(row)

# 수정
@bp.put("/api/admin/passes/plans/<int:pid>")
@bp.put("/api/admin/pass-plans/<int:pid>")  # alias
def admin_update_plan(pid: int):
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    body = request.get_json(silent=True) or {}
    try:
        u = parse_plan_update(body)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    ok = svc_update_plan(db, pid, u)
    if not ok:
        return jsonify({"error": "Plan not found or nothing to update"}), 404
    return jsonify({"message": "updated"})

# 삭제
@bp.delete("/api/admin/passes/plans/<int:pid>")
@bp.delete("/api/admin/pass-plans/<int:pid>")  # alias
def admin_delete_plan(pid: int):
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    ok = svc_delete_plan(db, pid)
    if not ok:
        return jsonify({"error": "Plan not found"}), 404
    return jsonify({"message": "deleted"})

# -------- Passes (Admin) ----------
# 목록
@bp.get("/api/admin/passes")
def admin_list_passes():
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    q = request.args.get("q")
    status = request.args.get("status")  # active/expired/cancelled
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 10)), 1), 100)

    rows, total = svc_list_passes(db, q, page, page_size, status)
    return jsonify({"items": rows, "total": total, "page": page, "page_size": page_size})

# 생성
@bp.post("/api/admin/passes")
def admin_create_pass():
    try:
        admin = require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    body = request.get_json(silent=True) or {}

    plan_id = body.get("plan_id")
    plan = None
    if plan_id:
        plan = svc_get_plan(db, int(plan_id))

    try:
        n = parse_pass_create(body, plan=plan)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    sid = svc_create_pass(db, n, admin_user_id=admin["user_id"])
    return jsonify({"id": sid, "message": "created"}), 201

# 단건 조회
@bp.get("/api/admin/passes/<int:sid>")
def admin_get_pass(sid: int):
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    row = svc_get_pass(db, sid)
    if not row:
        return jsonify({"error": "Pass not found"}), 404
    return jsonify(row)

# 수정
@bp.put("/api/admin/passes/<int:sid>")
def admin_update_pass(sid: int):
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    body = request.get_json(silent=True) or {}
    try:
        u = parse_pass_update(body)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    ok = svc_update_pass(db, sid, u)
    if not ok:
        return jsonify({"error": "Pass not found or nothing to update"}), 404
    return jsonify({"message": "updated"})

# 삭제
@bp.delete("/api/admin/passes/<int:sid>")
def admin_delete_pass(sid: int):
    try:
        require_admin()
    except AuthzError as e:
        return jsonify({"error": str(e)}), 403

    db = get_db()
    ok = svc_delete_pass(db, sid)
    if not ok:
        return jsonify({"error": "Pass not found"}), 404
    return jsonify({"message": "deleted"})

# -------- Public check (for OCR/Front) ----------
@bp.get("/api/passes/check")
def public_check_plate():
    """
    번호판 쿼리 파라미터 인코딩 이슈 보정:
    - 일반 요청: /api/passes/check?plate=32라4227 → 일부 환경에서 한글이 깨질 수 있음
    - 안전 요청: --data-urlencode 사용 또는 브라우저 자동 인코딩
    여기서도 unquote_plus로 1차 보정 시도.
    """
    raw = request.args.get("plate") or ""
    # URL 인코딩 보정 시도 (예: 32%EB%9D%BC4227 또는 공백이 + 로 들어오는 케이스)
    try:
        decoded = urllib.parse.unquote_plus(raw)
    except Exception:
        decoded = raw

    plate = decoded.strip().replace(" ", "")
    if not plate:
        return jsonify({"error": "plate is required"}), 400

    db = get_db()
    row = svc_is_plate_active(db, plate)
    return jsonify({"active": bool(row), "match": row})

# -------- Test endpoints (public) ----------
@bp.post("/api/passes/test/purchase")
def test_purchase():
    """
    테스트용: 결제 레코드 + passes 발급 한 번에.
    body 예:
    {
      "user_id": 1,
      "car_id":  10,            # 선택
      "plan_id": 1,
      "method":  "app",         # 'card' | 'cash' | 'app'
      "auto_renew": true,       # 선택
      "start_date": "2025-09-21" # 선택(없거나 공백/빈문자열이면 오늘)
    }
    """
    db = get_db()
    body = request.get_json(silent=True) or {}

    # 필수항목
    try:
        user_id = int(body.get("user_id"))
        plan_id = int(body.get("plan_id"))
    except Exception:
        return jsonify({"error": "user_id and plan_id are required (int)"}), 400

    # car_id: 선택
    car_id = body.get("car_id")
    if car_id is not None:
        try:
            car_id = int(car_id)
        except Exception:
            return jsonify({"error": "car_id must be int or null"}), 400

    # 결제 수단 기본값 app
    method = (body.get("method") or "app").lower()
    if method not in ("card", "cash", "app"):
        return jsonify({"error": "method must be one of card/cash/app"}), 400

    auto_renew = bool(body.get("auto_renew", False))

    # start_date 파싱 견고화: 공백/빈문자열 → 오늘, 날짜+시간도 허용
    raw_s = body.get("start_date") or ""
    if isinstance(raw_s, str):
        raw_s = raw_s.strip()
    if not raw_s:
        start_d = date.today()
    else:
        try:
            start_d = date.fromisoformat(raw_s)
        except Exception:
            try:
                # "YYYY-MM-DDTHH:MM:SS" 또는 "YYYY-MM-DD HH:MM:SS"
                start_d = datetime.fromisoformat(raw_s).date()
            except Exception:
                try:
                    # "YYYY-MM-DD ..." 꼴이면 날짜만 취득
                    start_d = date.fromisoformat(raw_s.split("T")[0].split(" ")[0])
                except Exception:
                    return jsonify({"error": "start_date must be ISO date (YYYY-MM-DD)"}), 400

    try:
        res = svc_test_purchase_with_payment(
            db,
            user_id=user_id,
            car_id=car_id,
            plan_id=plan_id,
            method=method,
            auto_renew=auto_renew,
            start=start_d,
        )
        return jsonify({"message": "ok", **res}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.post("/api/passes/test/payments")
def test_create_payment_only():
    """
    테스트용: 결제 레코드만 생성.
    body: { "amount": 120000, "method": "card" }
    """
    db = get_db()
    body = request.get_json(silent=True) or {}
    try:
        amount = float(body.get("amount"))
        method = (body.get("method") or "app").lower()
        if method not in ("card", "cash", "app"):
            raise ValueError("invalid method")
    except Exception as e:
        return jsonify({"error": f"bad request: {e}"}), 400

    try:
        pid = svc_insert_payment(db, amount=amount, method=method, event_id=None, success=True)
        return jsonify({"message": "ok", "payment_id": pid}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
