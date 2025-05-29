#BluePrint+응답통일 유틸 + 예외처리 미들웨어 적용

from flask import Blueprint, request
from services.parking_service import (
    handle_entry, handle_exit
)
from utils.response import make_response, error_response
from utils.validation import validate_image_file
import logging

bp = Blueprint('parking', __name__, url_prefix='/api/parking')

# 공통 예외처리 (Flask 2.x 이상)
@bp.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Unhandled Exception in parking route")
    return error_response("서버 내부 오류", 500)

@bp.route('/entry', methods=['POST'])
def entry():
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

@bp.route('/exit', methods=['POST'])
def exit():
    image_file = request.files.get('image')
    if not validate_image_file(image_file):
        return error_response("이미지 파일이 필요합니다.", 400)
    try:
        result = handle_exit(image_file.read())
        return make_response(result, 200)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logging.exception("Exit Error")
        return error_response("예상치 못한 오류", 500)

