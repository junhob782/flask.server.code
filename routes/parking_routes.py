# routes/parking_routes.py

import numpy as np
import cv2
from flask import Blueprint, request, jsonify
from services.parking_service import handle_entry, handle_exit
from utils.response import make_response, error_response
from utils.validation import validate_image_file
import logging
from datetime import datetime

from services.vision_service import analyze_parking_slots
from config import SLOT_ROIS

# --- 데이터베이스 모델 및 SQLAlchemy 함수 import ---
from models import db, ParkingSpace, ParkingEvent, Car
from sqlalchemy import func, case

bp = Blueprint('parking', __name__)

# 공통 예외처리
@bp.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Unhandled Exception in parking route")
    return error_response("서버 내부 오류", 500)

# --- 이미지 파일 기반 입차 API ---
@bp.route('/entry', methods=['POST'])
def entry_with_image(): # 함수 이름을 명확하게 변경
    """(이미지 기반) 차량 입차를 처리합니다."""
    image_file = request.files.get('image')
    if not validate_image_file(image_file):
        return error_response("이미지 파일이 필요합니다.", 400)
    try:
        result = handle_entry(image_file.read())
        return make_response(result, 201)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logging.exception("Entry Error")
        return error_response("예상치 못한 오류", 500)

# --- 표준 출차 API (JSON 기반) ---
@bp.route('/exit', methods=['POST'])
def handle_car_exit():
    """
    (JSON 기반) 차량 번호를 받아 출차를 처리하고,
    주차 시간과 계산된 요금을 반환합니다. (표준 API)
    """
    data = request.get_json()
    if not data or 'license_plate' not in data:
        return jsonify({'error': '요청 본문에 차량 번호(license_plate)가 필요합니다.'}), 400

    plate = data['license_plate']

    try:
        # services 레이어의 표준 handle_exit 함수를 호출합니다.
        result = handle_exit(license_plate=plate)
        return jsonify(result), 200
    except ValueError as e:
        # handle_exit에서 주차 중인 차를 못 찾으면 ValueError가 발생합니다.
        return jsonify({'error': str(e)}), 404 # 404 Not Found
    except Exception as e:
        # 그 외 모든 예외는 500 서버 오류로 처리합니다.
        logging.exception("Exit Error")
        return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

# --- 조회 관련 API (수정 없음) ---

@bp.route('/status')
def get_parking_status():
    """건물(구역)별 전체 주차 공간 수와 빈 공간 수를 반환합니다."""
    try:
        status_query = db.session.query(
            ParkingSpace.location_desc,
            func.count(ParkingSpace.space_id).label('total_spaces'),
            func.sum(case((ParkingSpace.is_occupied == False, 1), else_=0)).label('free_spaces')
        ).group_by(ParkingSpace.location_desc).all()

        result = {
            row.location_desc if row.location_desc else 'default': {
                'total': row.total_spaces,
                'free': row.free_spaces
            } for row in status_query
        }
        
        if not result:
            return jsonify({})

        return jsonify(result)
        
    except Exception as e:
        print(f"Error in get_parking_status: {e}")
        return jsonify({'error': '서버 내부 오류가 발생했습니다.', 'success': False}), 500

@bp.route('/history/<string:license_plate>')
def get_parking_history(license_plate):
    """특정 차량 번호의 모든 입출차 기록을 최신순으로 반환합니다."""
    try:
        events = ParkingEvent.query.filter_by(license_plate=license_plate).order_by(ParkingEvent.entry_time.desc()).all()

        result = []
        for event in events:
            result.append({
                'entry_time': event.entry_time.isoformat() if event.entry_time else None,
                'exit_time': event.exit_time.isoformat() if event.exit_time else None,
                'gate': event.gate,
            })

        return jsonify(result)

    except Exception as e:
        print(f"Error in get_parking_history: {e}")
        return jsonify({'error': '서버 내부 오류가 발생했습니다.', 'success': False}), 500

@bp.route('/current/<string:license_plate>')
def get_current_parking_status(license_plate):
    """특정 차량의 현재 진행 중인 주차 기록(출차되지 않은)을 반환합니다."""
    try:
        current_event = ParkingEvent.query.filter_by(
            license_plate=license_plate, 
            exit_time=None
        ).order_by(ParkingEvent.entry_time.desc()).first()

        if not current_event:
            return jsonify({'message': '현재 주차 중인 차량이 아닙니다.'}), 404

        result = {
            'license_plate': current_event.license_plate,
            'entry_time': current_event.entry_time.isoformat(),
            'gate': current_event.gate
        }
        return jsonify(result)

    except Exception as e:
        print(f"Error in get_current_parking_status: {e}")
        return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

# --- JSON 기반 수동 입차 API (유지) ---

@bp.route('/entry/manual', methods=['POST'])
def handle_car_entry_manual():
    """(JSON 기반) 새로운 차량의 입차를 수동으로 기록합니다."""
    data = request.get_json()
    if not data or 'license_plate' not in data:
        return jsonify({'error': '차량 번호(license_plate)가 필요합니다.'}), 400

    plate = data['license_plate']

    try:
        existing_event = ParkingEvent.query.filter_by(license_plate=plate, exit_time=None).first()
        if existing_event:
            return jsonify({'error': '이미 주차 중인 차량입니다.'}), 409

        car = Car.query.filter_by(license_plate=plate).first()
        if not car:
            car = Car(license_plate=plate)
            db.session.add(car)
            db.session.commit()

        new_event = ParkingEvent(
            car_id=car.car_id,
            license_plate=plate,
            entry_time=datetime.now(),
            status='in',
            exit_time=None
        )
        db.session.add(new_event)
        db.session.commit()

        return jsonify({'message': f'{plate} 차량 입차가 기록되었습니다.'}), 201

    except Exception as e:
        db.session.rollback()
        print(f"Error in handle_car_entry: {e}")
        return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500
    
    
@bp.route('/update_cctv_status', methods=['POST'])
def update_cctv_status():
    """
    CCTV 프레임 이미지 전체를 받아 각 주차 공간의 상태를 분석하고 DB를 업데이트합니다.
    """
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': '이미지 파일이 필요합니다.'}), 400

    try:
        # 1. 전송된 이미지 파일을 OpenCV가 읽을 수 있는 형식으로 변환
        filestr = image_file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        full_frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # 2. 설정된 ROI 좌표에 따라 전체 이미지에서 각 주차 공간을 잘라냄
        cropped_images = []
        space_ids = []
        for space_id, ((x1, y1), (x2, y2)) in SLOT_ROIS.items():
            cropped_images.append(full_frame[y1:y2, x1:x2])
            space_ids.append(space_id)

        # 3. 잘라낸 이미지들을 AI 두뇌(vision_service)에게 보내 분석 요청
        predictions = analyze_parking_slots(cropped_images)

        # 4. 분석 결과를 바탕으로 데이터베이스 업데이트
        updated_count = 0
        for i, space_id in enumerate(space_ids):
            is_now_occupied = (predictions[i] == 'occupied')
            
            # DB에서 해당 주차 공간을 찾아서 is_occupied 상태를 업데이트
            space = ParkingSpace.query.get(space_id)
            if space and space.is_occupied != is_now_occupied:
                space.is_occupied = is_now_occupied
                updated_count += 1
        
        db.session.commit()

        return jsonify({
            'message': '주차 공간 상태 분석 및 업데이트 완료',
            'updated_count': updated_count,
            'details': dict(zip(space_ids, predictions)) # {1: 'occupied', 2: 'empty', ...}
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error in update_cctv_status: {e}")
        return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500