# passes/dao.py
from typing import Any, Dict, List, Optional, Tuple

# ---------- Plans ----------
def list_plans(conn, q: Optional[str], page: int, page_size: int) -> Tuple[List[Dict[str, Any]], int]:
    offset = (page - 1) * page_size
    where, params = "", []
    if q:
        where = "WHERE name LIKE %s OR description LIKE %s"
        like = f"%{q}%"
        params.extend([like, like])

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS cnt FROM pass_plans {where}", params)
        total = cur.fetchone()["cnt"]
        cur.execute(
            f"""
            SELECT id, name, description, price, duration_days, is_active, created_at, updated_at
            FROM pass_plans
            {where}
            ORDER BY is_active DESC, created_at DESC
            LIMIT %s OFFSET %s
            """,
            params + [page_size, offset],
        )
        rows = cur.fetchall()
    return rows, total

def get_plan(conn, pid: int) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
          SELECT id, name, description, price, duration_days, is_active, created_at, updated_at
          FROM pass_plans WHERE id=%s
        """, (pid,))
        return cur.fetchone()

def insert_plan(conn, name: str, description: Optional[str], price: float, duration_days: int, is_active: bool) -> int:
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO pass_plans (name, description, price, duration_days, is_active)
          VALUES (%s, %s, %s, %s, %s)
        """, (name, description, price, duration_days, 1 if is_active else 0))
        conn.commit()
        return cur.lastrowid

def update_plan(conn, pid: int, u: Dict[str, Any]) -> bool:
    if not u:
        return False
    fields, params = [], []
    for k, v in u.items():
        if k not in ("name", "description", "price", "duration_days", "is_active"):
            continue
        fields.append(f"{k}=%s")
        if k == "is_active":
            params.append(1 if bool(v) else 0)
        else:
            params.append(v)
    if not fields:
        return False
    params.append(pid)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE pass_plans SET {', '.join(fields)} WHERE id=%s", params)
        conn.commit()
        return cur.rowcount > 0

def delete_plan(conn, pid: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM pass_plans WHERE id=%s", (pid,))
        conn.commit()
        return cur.rowcount > 0

# ---------- Passes (subscriptions) ----------
def list_passes(conn, q: Optional[str], page: int, page_size: int, status: Optional[str]) -> Tuple[List[Dict[str, Any]], int]:
    offset = (page - 1) * page_size
    where, params, conds = "", [], []
    if q:
        conds.append("(u.username LIKE %s OR p2.name LIKE %s)")
        like = f"%{q}%"
        params.extend([like, like])
    if status:
        conds.append("p.status = %s")
        params.append(status)
    if conds:
        where = "WHERE " + " AND ".join(conds)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM passes p
            JOIN user u ON u.user_id = p.user_id
            JOIN pass_plans p2 ON p2.id = p.plan_id
            {where}
            """,
            params
        )
        total = cur.fetchone()["cnt"]

        cur.execute(
            f"""
            SELECT p.id, p.user_id, p.car_id, p.plan_id, p.start_date, p.end_date,
                   p.auto_renew, p.status, p.created_by, p.created_at, p.updated_at,
                   u.username, p2.name AS plan_name, p2.price, p2.duration_days
            FROM passes p
            JOIN user u ON u.user_id = p.user_id
            JOIN pass_plans p2 ON p2.id = p.plan_id
            {where}
            ORDER BY p.status='active' DESC, p.end_date DESC, p.created_at DESC
            LIMIT %s OFFSET %s
            """,
            params + [page_size, offset]
        )
        rows = cur.fetchall()
    return rows, total

def get_pass(conn, sid: int) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.user_id, p.car_id, p.plan_id, p.start_date, p.end_date,
                   p.auto_renew, p.status, p.created_by, p.created_at, p.updated_at
            FROM passes p WHERE p.id=%s
            """, (sid,)
        )
        return cur.fetchone()

def insert_pass(conn, user_id: int, car_id: Optional[int], plan_id: int, start_date, end_date, auto_renew: bool, created_by: Optional[int]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO passes (user_id, car_id, plan_id, start_date, end_date, auto_renew, status, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, 'active', %s)
            """,
            (user_id, car_id, plan_id, start_date, end_date, 1 if auto_renew else 0, created_by)
        )
        conn.commit()
        return cur.lastrowid

def update_pass(conn, sid: int, u: Dict[str, Any]) -> bool:
    if not u:
        return False
    fields, params = [], []
    for k, v in u.items():
        if k not in ("user_id", "car_id", "plan_id", "start_date", "end_date", "auto_renew", "status"):
            continue
        if k == "auto_renew":
            fields.append("auto_renew=%s")
            params.append(1 if bool(v) else 0)
        else:
            fields.append(f"{k}=%s")
            params.append(v)
    if not fields:
        return False
    params.append(sid)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE passes SET {', '.join(fields)} WHERE id=%s", params)
        conn.commit()
        return cur.rowcount > 0

def delete_pass(conn, sid: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM passes WHERE id=%s", (sid,))
        conn.commit()
        return cur.rowcount > 0

def is_plate_active(conn, plate_number: str):
    """
    차량 번호판(license_plate)으로 현재 유효한 정기권 존재 여부 체크.
    우선순위:
      1) passes.car_id ↔ car.car_id 조인 (가장 정확/빠름)
      2) passes.car_id IS NULL 인 경우 user_id ↔ car.user_id 로 보조 매칭
    반환: dict(row) 또는 None
    """
    with conn.cursor() as cur:
        # 1) car_id를 통한 직접 매칭 (권장 경로)
        cur.execute(
            """
            SELECT p.id, p.user_id, p.car_id, p.plan_id,
                   p.start_date, p.end_date, p.status
            FROM passes p
            JOIN car c ON c.car_id = p.car_id
            WHERE c.license_plate = %s
              AND p.status = 'active'
              AND CURDATE() BETWEEN p.start_date AND p.end_date
            ORDER BY p.end_date DESC
            LIMIT 1
            """,    
            (plate_number,),
        )
        row = cur.fetchone()
        if row:
            return row

        # 2) passes.car_id 가 NULL일 때의 보조 경로 (user_id 경유)
        cur.execute(
            """
            SELECT p.id, p.user_id, p.car_id, p.plan_id,
                   p.start_date, p.end_date, p.status
            FROM passes p
            JOIN car c ON c.user_id = p.user_id
            WHERE p.car_id IS NULL
              AND c.license_plate = %s
              AND p.status = 'active'
              AND CURDATE() BETWEEN p.start_date AND p.end_date
            ORDER BY p.end_date DESC
            LIMIT 1
            """,
            (plate_number,),
        )
        return cur.fetchone()

