##서비스 계층 로직

from DB.connection import get_db
from utils.ocr import recognize_plate
from utils.fee_calc import calculate_fee
import datetime

def handle_entry(image_bytes):
    plate = recognize_plate(image_bytes)
    if not plate:
        raise ValueError("OCR 실패 – 번호판을 인식할 수 없습니다.")
    now = datetime.datetime.now()
    db = get_db()
    with db.cursor() as cur:
        # 차량 정보 업서트
        cur.execute("SELECT car_id FROM car WHERE license_plate=%s", (plate,))
        car = cur.fetchone()
        if not car:
            cur.execute("INSERT INTO car (license_plate) VALUES (%s)", (plate,))
            car_id = cur.lastrowid
        else:
            car_id = car['car_id']
        # 입차 이벤트 기록
        cur.execute("INSERT INTO parkingevent (car_id, entry_time, recognized) VALUES (%s, %s, 1)", (car_id, now))
        db.commit()
    return {'car_id': car_id, 'entry_time': now.isoformat()}

def handle_exit(image_bytes):
    plate = recognize_plate(image_bytes)
    if not plate:
        raise ValueError("OCR 실패 – 번호판을 인식할 수 없습니다.")
    now = datetime.datetime.now()
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT e.*, u.user_type
            FROM parkingevent e
            LEFT JOIN car c ON e.car_id=c.car_id
            LEFT JOIN user u ON c.user_id=u.user_id
            WHERE c.license_plate=%s AND e.exit_time IS NULL
            ORDER BY entry_time DESC LIMIT 1
        """, (plate,))
        event = cur.fetchone()
        if not event:
            raise ValueError("입차 기록이 없거나 이미 출차 처리됨.")
        cur.execute("UPDATE parkingevent SET exit_time=%s WHERE event_id=%s", (now, event['event_id']))
        fee = calculate_fee(event['entry_time'], now, event['user_type'] or 'non_member')
        cur.execute("""
            INSERT INTO payment (event_id, amount, payment_time, payment_method, success)
            VALUES (%s, %s, %s, %s, %s)
        """, (event['event_id'], fee, now, 'card', 1))
        db.commit()
    return {'fee': fee, 'exit_time': now.isoformat()}
