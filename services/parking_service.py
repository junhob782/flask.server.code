# services/parking_service.py
from DB.connection import get_db
from utils.ocr import recognize_plate as default_recognize_plate
from utils.fee_calc import calculate_fee
import datetime
from typing import Callable, Optional

def _normalize_plate(s: str) -> str:
    return (s or "").strip().replace(" ", "")

def handle_entry(
    image_bytes: bytes,
    allow_duplicate: bool = False,
    gate: Optional[str] = None,
    source: Optional[str] = None,
    image_path: Optional[str] = None,
    crop_path: Optional[str] = None,
    ocr_confidence: Optional[float] = None,
    ocr_func: Optional[Callable[[bytes], str]] = None,   # 👈 OCR 함수 주입 가능
):
    """
    1) OCR → plate
    2) 같은 번호판 '열린 이벤트(미출차)' 있으면 중복 방지 (allow_duplicate=False)
    3) car upsert
    4) parkingevent INSERT(entry_time=NOW(), recognized=1)
    """
    # ---- OCR 호출 부분 (ocr_library 제거) ----
    ocr_func = ocr_func or default_recognize_plate
    plate = _normalize_plate(ocr_func(image_bytes))
    if not plate:
        raise ValueError("OCR 실패 – 번호판을 인식할 수 없습니다.")

    now = datetime.datetime.now()
    db = get_db()
    with db.cursor() as cur:
        # 열린 이벤트(미출차) 조회
        cur.execute("""
            SELECT e.event_id, e.car_id, e.entry_time
            FROM parkingevent e
            JOIN car c ON c.car_id = e.car_id
            WHERE c.license_plate = %s
              AND e.exit_time IS NULL
            ORDER BY e.entry_time DESC
            LIMIT 1
        """, (plate,))
        opened = cur.fetchone()

        if opened and not allow_duplicate:
            return {
                "duplicated": True,
                "car_id": opened["car_id"],
                "plate": plate,
                "event_id": opened["event_id"],
                "entry_time": opened["entry_time"].isoformat() if opened["entry_time"] else None,
                "gate": gate, "source": source,
                "image_path": image_path, "crop_path": crop_path,
                "ocr_confidence": ocr_confidence,
            }

        # car upsert
        cur.execute("SELECT car_id FROM car WHERE license_plate=%s", (plate,))
        car = cur.fetchone()
        if not car:
            cur.execute("INSERT INTO car (license_plate) VALUES (%s)", (plate,))
            car_id = cur.lastrowid
        else:
            car_id = car["car_id"]

        # 주차 이벤트 생성
        cur.execute("""
             INSERT INTO parkingevent (car_id, space_id, gate, license_plate, entry_time, recognized)
    VALUES (%s, %s, %s, %s, %s, 1)
        """, (car_id, None, gate, plate, now))
        event_id = cur.lastrowid
        db.commit()

    return {
    "duplicated": False,
    "car_id": car_id,
    "plate": plate,
    "event_id": event_id,
    "entry_time": now.isoformat(),
    "gate": gate,
    "space_id": None,
    "image_path": image_path, "crop_path": crop_path,
    "ocr_confidence": ocr_confidence,
}

def handle_exit(
    image_bytes: bytes,
    payment_method: str = "card",
    payment_success: bool = True,
    zero_fee_for_pass: bool = True,
    ocr_func: Optional[Callable[[bytes], str]] = None,   # 👈 동일하게 주입 가능
):
    ocr_func = ocr_func or default_recognize_plate
    plate = _normalize_plate(ocr_func(image_bytes))
    if not plate:
        raise ValueError("OCR 실패 – 번호판을 인식할 수 없습니다.")

    now = datetime.datetime.now()
    db = get_db()
    with db.cursor() as cur:
        # 미출차 이벤트 찾기
        cur.execute("""
            SELECT e.*, u.user_type, c.car_id
            FROM parkingevent e
            JOIN car c ON e.car_id = c.car_id
            LEFT JOIN user u ON c.user_id = u.user_id
            WHERE c.license_plate=%s AND e.exit_time IS NULL
            ORDER BY e.entry_time DESC
            LIMIT 1
        """, (plate,))
        event = cur.fetchone()
        if not event:
            raise ValueError("입차 기록이 없거나 이미 출차 처리됨.")

        # 출차 처리
        cur.execute("UPDATE parkingevent SET exit_time=%s WHERE event_id=%s", (now, event['event_id']))

        # 요금 계산
        fee = 0
        if not zero_fee_for_pass or (event.get('user_type') or 'non_member') == 'non_member':
            fee = calculate_fee(event['entry_time'], now, event.get('user_type') or 'non_member')

        # 결제 기록 (성공 가정)
        cur.execute("""
            INSERT INTO payment (event_id, amount, payment_time, payment_method, success)
            VALUES (%s, %s, %s, %s, %s)
        """, (event['event_id'], fee, now, payment_method, 1 if payment_success else 0))
        db.commit()

    return {"fee": float(fee), "exit_time": now.isoformat(), "plate": plate}
