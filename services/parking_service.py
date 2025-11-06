from DB.connection import get_db
from utils.ocr import recognize_plate as default_recognize_plate
from utils.fee_calc import calculate_fee
import datetime
from typing import Callable, Optional

def _normalize_plate(s: str) -> str:
    """공백 등을 제거하여 번호판 문자열을 정규화합니다."""
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
    차량 입차 과정을 처리합니다.
    1) OCR로 이미지에서 번호판을 인식합니다.
    2) 동일 차량의 미출차 기록이 있으면 중복으로 처리합니다 (allow_duplicate=False인 경우).
    3) 'car' 테이블에 차량 정보가 없으면 새로 추가합니다 (upsert).
    4) 'parkingevent' 테이블에 입차 기록을 새로 생성합니다.
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

        # 중복 입차 처리
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

        # car 테이블에 차량 정보가 없으면 추가 (upsert 로직)
        cur.execute("SELECT car_id FROM car WHERE license_plate=%s", (plate,))
        car = cur.fetchone()
        if not car:
            cur.execute("INSERT INTO car (license_plate) VALUES (%s)", (plate,))
            car_id = cur.lastrowid
        else:
            car_id = car["car_id"]

        # 새로운 주차 이벤트 생성
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
    """
    차량 출차 과정을 처리하고 요금을 계산합니다.
    """
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

        # 출차 처리: exit_time 업데이트
        cur.execute("UPDATE parkingevent SET exit_time=%s WHERE event_id=%s", (now, event['event_id']))

        # 요금 계산
        fee = 0
        # 정기권 사용자가 아니거나, 정기권 사용자도 요금을 부과하는 옵션일 경우
        if not zero_fee_for_pass or (event.get('user_type') or 'non_member') == 'non_member':
            fee = calculate_fee(event['entry_time'], now, event.get('user_type') or 'non_member')

        # 결제 기록 생성
        cur.execute("""
            INSERT INTO payment (event_id, amount, payment_time, payment_method, success)
            VALUES (%s, %s, %s, %s, %s)
        """, (event['event_id'], fee, now, payment_method, 1 if payment_success else 0))
        db.commit()

    return {"fee": float(fee), "exit_time": now.isoformat(), "plate": plate}

def handle_exit_by_plate(
    license_plate: str,
    payment_method: str = "card", 
    payment_success: bool = True,
    zero_fee_for_pass: bool = True,
):
    """
    차량 번호만으로 출차를 처리합니다. (Flutter JSON API용)
    """
    plate = _normalize_plate(license_plate)
    if not plate:
        raise ValueError("차량 번호가 유효하지 않습니다.")

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

        # 출차 처리: exit_time 업데이트
        cur.execute("UPDATE parkingevent SET exit_time=%s WHERE event_id=%s", (now, event['event_id']))

        # 요금 계산
        fee = 0
        if not zero_fee_for_pass or (event.get('user_type') or 'non_member') == 'non_member':
            fee = calculate_fee(event['entry_time'], now, event.get('user_type') or 'non_member')

        # 결제 기록 생성
        cur.execute("""
            INSERT INTO payment (event_id, amount, payment_time, payment_method, success)
            VALUES (%s, %s, %s, %s, %s)
        """, (event['event_id'], fee, now, payment_method, 1 if payment_success else 0))
        db.commit()

    return {"fee": float(fee), "exit_time": now.isoformat(), "plate": plate}

# 차량 번호 받아서 DB검색하는 함수
def get_guest_parking_details(license_plate: str):
    """
    (비회원용) 차량 번호로 현재 주차 중인 차량의 상세 정보
    (입차시간, 이용시간, 현재요금)를 조회합니다.
    """
    plate = _normalize_plate(license_plate)
    if not plate:
        raise ValueError("차량 번호가 유효하지 않습니다.")

    now = datetime.datetime.now()
    db = get_db()
    with db.cursor() as cur:
        # 미출차 이벤트 찾기 (라이센스 플레이트로 직접 조회)
        cur.execute("""
            SELECT 
                e.entry_time,
                'non_member' as user_type
            FROM parkingevent e
            WHERE e.license_plate = %s AND e.exit_time IS NULL
            ORDER BY e.entry_time DESC
            LIMIT 1
        """, (plate,))
        event = cur.fetchone()

        if not event:
            raise ValueError("현재 주차 중인 차량이 아니거나, 입차 기록이 없습니다.")

        entry_time = event['entry_time']
        duration = now - entry_time
        # 차량에 연결된 유저가 있으면 해당 유저타입을, 없으면 'non_member'를 기본값으로 사용
        user_type = event.get('user_type') or 'non_member'

        # 요금 계산 유틸리티 호출
        current_fee = calculate_fee(entry_time, now, user_type)

        # 앱(Flutter)으로 보낼 최종 데이터 구성
        return {
            "license_plate": plate,
            "entry_time": entry_time.isoformat(),
            "duration_seconds": int(duration.total_seconds()),
            "current_fee": float(current_fee)
        }