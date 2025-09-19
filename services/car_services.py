# services/car_services.py
from typing import Optional, Tuple
import pymysql

# 중앙에서 재사용하고 있다면 import 하세요:
# from connection import get_db
# 여기서는 간단히 직접 연결(당신 환경으로 맞추세요)
def _get_db():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='123456',
        database='lotbotsystem',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )

def upsert_car_and_get_id(license_plate: str, user_id: Optional[int] = None) -> Tuple[int, bool]:
    """
    car(license_plate UNIQUE)에 '없으면 INSERT, 있으면 기존 행'을 반환.
    return: (car_id, created_new)
    """
    # user_id가 NOT NULL 스키마라면 기본(게스트) 유저 아이디를 준비
    GUEST_USER_ID = 1 if user_id is None else user_id

    conn = _get_db()
    try:
        with conn.cursor() as cur:
            # MySQL upsert + LAST_INSERT_ID 트릭
            sql = """
            INSERT INTO car (license_plate, user_id)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE
                car_id = LAST_INSERT_ID(car_id),
                user_id = COALESCE(VALUES(user_id), user_id);
            """
            cur.execute(sql, (license_plate, GUEST_USER_ID))
            conn.commit()

            car_id = cur.lastrowid
            # rowcount==1이면 '삽입', 2이면 '업데이트'가 되는 경우가 많지만,
            # 안전하게는 재조회로 created_new를 판정해도 됩니다.
            created_new = (cur.rowcount == 1)
            return car_id, created_new
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
