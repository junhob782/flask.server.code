# passes/service.py
from typing import Any, Dict, Optional
from datetime import date, timedelta
from . import dao

# ------------------------
# 공통 유틸
# ------------------------
def _bool(v: Any) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str): return v.strip().lower() in ("1","true","t","y","yes")
    return False

def _date(d: Any) -> date:
    if isinstance(d, date): return d
    if isinstance(d, str): return date.fromisoformat(d)
    raise ValueError("invalid date")

# ------------------------
# Plans
# ------------------------
def parse_plan_create(body: Dict[str, Any]) -> Dict[str, Any]:
    name = (body.get("name") or "").strip()
    description = (body.get("description") or None)
    price = float(body.get("price", 0))
    duration_days = int(body.get("duration_days") or 0)
    is_active = _bool(body.get("is_active", True))
    if not name: raise ValueError("name is required")
    if duration_days <= 0: raise ValueError("duration_days must be > 0")
    if price < 0: raise ValueError("price must be >= 0")
    return dict(name=name, description=description, price=price, duration_days=duration_days, is_active=is_active)

def parse_plan_update(body: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "name" in body:
        name = (body.get("name") or "").strip()
        if not name: raise ValueError("name cannot be empty")
        out["name"] = name
    if "description" in body:
        out["description"] = body.get("description")
    if "price" in body:
        price = float(body.get("price"))
        if price < 0: raise ValueError("price must be >= 0")
        out["price"] = price
    if "duration_days" in body:
        dd = int(body.get("duration_days"))
        if dd <= 0: raise ValueError("duration_days must be > 0")
        out["duration_days"] = dd
    if "is_active" in body:
        out["is_active"] = _bool(body.get("is_active"))
    if not out: raise ValueError("no updatable fields")
    return out

def list_plans(conn, q, page, page_size):
    return dao.list_plans(conn, q, page, page_size)

def get_plan(conn, pid: int):
    return dao.get_plan(conn, pid)

def create_plan(conn, p: Dict[str, Any]) -> int:
    return dao.insert_plan(conn, **p)

def update_plan(conn, pid: int, u: Dict[str, Any]) -> bool:
    return dao.update_plan(conn, pid, u)

def delete_plan(conn, pid: int) -> bool:
    return dao.delete_plan(conn, pid)

# ------------------------
# Passes
# ------------------------
def parse_pass_create(body: Dict[str, Any], plan: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    user_id = int(body.get("user_id"))
    car_id = body.get("car_id")
    plan_id = int(body.get("plan_id"))
    auto_renew = _bool(body.get("auto_renew", False))

    start_date = body.get("start_date")
    end_date = body.get("end_date")
    if start_date and end_date:
        s, e = _date(start_date), _date(end_date)
    else:
        if not plan:
            raise ValueError("plan lookup required to compute end_date")
        s = date.today()
        e = s + timedelta(days=int(plan["duration_days"]))
    if e <= s:
        raise ValueError("end_date must be after start_date")

    return dict(
        user_id=user_id,
        car_id=int(car_id) if car_id is not None else None,
        plan_id=plan_id,
        start_date=s,
        end_date=e,
        auto_renew=auto_renew
    )

def parse_pass_update(body: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "user_id" in body: out["user_id"] = int(body["user_id"])
    if "car_id" in body:  out["car_id"]  = (int(body["car_id"]) if body["car_id"] is not None else None)
    if "plan_id" in body: out["plan_id"] = int(body["plan_id"])
    if "start_date" in body: out["start_date"] = _date(body["start_date"])
    if "end_date" in body:   out["end_date"]   = _date(body["end_date"])
    if "auto_renew" in body: out["auto_renew"] = _bool(body["auto_renew"])
    if "status" in body:
        st = str(body["status"]).lower()
        if st not in ("active","expired","cancelled"):
            raise ValueError("status must be one of active/expired/cancelled")
        out["status"] = st
    if not out: raise ValueError("no updatable fields")
    return out

def list_passes(conn, q, page, page_size, status):
    return dao.list_passes(conn, q, page, page_size, status)

def get_pass(conn, sid: int):
    return dao.get_pass(conn, sid)

def create_pass(conn, n: Dict[str, Any], admin_user_id: int) -> int:
    return dao.insert_pass(conn, n["user_id"], n["car_id"], n["plan_id"], n["start_date"], n["end_date"], n["auto_renew"], admin_user_id)

def update_pass(conn, sid: int, u: Dict[str, Any]) -> bool:
    return dao.update_pass(conn, sid, u)

def delete_pass(conn, sid: int) -> bool:
    return dao.delete_pass(conn, sid)

def is_plate_active(conn, plate_number: str):
    return dao.is_plate_active(conn, plate_number)

# ------------------------
# 결제 + 테스트 구매
# ------------------------
def insert_payment(conn, amount: float, method: str, event_id: Optional[int] = None, success: bool = True) -> int:
    """
    payment 테이블에 결제 레코드를 만들고 payment_id 반환.
    event_id 는 테스트에선 None/0 둘 다 허용. (스키마가 NULL 허용인지 확인)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO payment (event_id, amount, payment_method, success)
            VALUES (%s, %s, %s, %s)
            """,
            (event_id, amount, method, 1 if success else 0),
        )
        conn.commit()
        return cur.lastrowid

def _has_active_overlap(conn, user_id: int, car_id: Optional[int], start: date) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM passes
            WHERE status = 'active'
              AND (user_id = %s OR (%s IS NOT NULL AND car_id = %s))
              AND end_date >= %s
            LIMIT 1
            """,
            (user_id, car_id, car_id, start),
        )
        return bool(cur.fetchone())

def test_purchase_with_payment(conn, user_id: int, car_id: Optional[int], plan_id: int,
                               method: str = "app", auto_renew: bool = False,
                               start: Optional[date] = None) -> Dict[str, Any]:
    s = start or date.today()

    plan = dao.get_plan(conn, plan_id)
    if not plan:
        raise ValueError("plan not found")
    if int(plan.get("is_active", 0)) != 1:
        raise ValueError("plan is inactive")

    if _has_active_overlap(conn, user_id, car_id, s):
        raise ValueError("already has active pass")

    amount = float(plan["price"])
    payment_id = insert_payment(conn, amount=amount, method=method, event_id=None, success=True)

    duration_days = int(plan["duration_days"])
    e = s + timedelta(days=duration_days)  # 필요시 -1일 정책으로 바꿔도 됨

    sid = dao.insert_pass(
        conn,
        user_id=user_id,
        car_id=car_id,
        plan_id=plan_id,
        start_date=s,
        end_date=e,
        auto_renew=auto_renew,
        created_by=user_id,
    )

    return {
        "payment_id": payment_id,
        "pass_id": sid,
        "start_date": s.isoformat(),
        "end_date": e.isoformat(),
        "amount": amount,
        "method": method,
    }

# routes.py에서 import하는 별칭
def svc_test_purchase_with_payment(conn, **kwargs):
    return test_purchase_with_payment(conn, **kwargs)
