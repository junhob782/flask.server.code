from typing import Dict, Any, Optional, Tuple, List

def list_notices(conn, q: Optional[str], page: int, page_size: int) -> Tuple[List[Dict[str, Any]], int]:
    offset = (page - 1) * page_size
    where = ""
    params: List[Any] = []
    if q:
        where = "WHERE title LIKE %s OR content LIKE %s"
        like = f"%{q}%"
        params.extend([like, like])

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS cnt FROM notices {where}", params)
        total = cur.fetchone()["cnt"]

        cur.execute(
            f"""
            SELECT id, title, content, is_pinned, created_by, created_at, updated_at
            FROM notices
            {where}
            ORDER BY is_pinned DESC, created_at DESC
            LIMIT %s OFFSET %s
            """,
            params + [page_size, offset],
        )
        rows = cur.fetchall()

    return rows, total

def get_notice(conn, nid: int) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, title, content, is_pinned, created_by, created_at, updated_at
            FROM notices WHERE id=%s
        """, (nid,))
        return cur.fetchone()

def insert_notice(conn, title: str, content: str, is_pinned: bool, created_by: Optional[int]) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO notices (title, content, is_pinned, created_by)
            VALUES (%s, %s, %s, %s)
        """, (title, content, 1 if is_pinned else 0, created_by))
        conn.commit()
        return cur.lastrowid

def update_notice(conn, nid: int, u: Dict[str, Any]) -> bool:
    if not u:
        return False
    fields = []
    params: List[Any] = []
    for k, v in u.items():
        if k not in ("title", "content", "is_pinned"):
            continue
        fields.append(f"{k}=%s")
        params.append(1 if (k == "is_pinned" and bool(v)) else v)

    if not fields:
        return False

    params.append(nid)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE notices SET {', '.join(fields)} WHERE id=%s", params)
        conn.commit()
        return cur.rowcount > 0

def delete_notice(conn, nid: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM notices WHERE id=%s", (nid,))
        conn.commit()
        return cur.rowcount > 0
