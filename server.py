# server.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
import os, sys

# ==============================
# 0) 스타트업 진단 로그
# ==============================
print("=== STARTUP DIAG ===")
print("CWD        :", os.getcwd())
print("SERVER_FILE:", __file__)
print("PYTHON     :", sys.executable)

# notices 블루프린트 import 진단
try:
    from notices import notices_bp as _probe_bp  # notices/__init__.py에서 routes.bp 를 re-export 해야 함
    print("NOTICES_BP : imported OK")
except Exception as e:
    print("NOTICES_BP : import FAILED ->", repr(e))
    _probe_bp = None

# passes 블루프린트 import 진단
try:
    from passes import passes_bp as _passes_bp
    print("PASSES_BP  : imported OK")
except Exception as e:
    print("PASSES_BP  : import FAILED ->", repr(e))
    _passes_bp = None

# 내부 유틸
from utils.db import close_db

# OCR 엔진
from utils.OCR_engines.ocr_googlevision import GoogleVisionPlate
from services.car_services import upsert_car_and_get_id  # 파일명/경로 확인

# 주차 서비스 (입차/출차)
from services.parking_service import (
    handle_entry as svc_parking_entry,
    handle_exit as svc_parking_exit,
)

# ==============================
# 1) 환경 변수 / 앱 생성
# ==============================
# Vision API 키 필수
api_key = os.getenv("GOOGLE_VISION_API_KEY")
if not api_key:
    raise RuntimeError("환경변수 GOOGLE_VISION_API_KEY가 설정되지 않았습니다.")

# Flask 앱
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
# 업로드 크기 제한(예: 10MB) — 필요시 조정
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# 요청 종료 시 DB 커넥션 정리
app.teardown_appcontext(close_db)

# CORS (개발 단계 전역 허용; 배포 시 origins 화이트리스트로 좁히기)
CORS(app)  # CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://yourapp.com"]}})

# ==============================
# 2) 블루프린트 등록 + URL 맵 출력
# ==============================
if _probe_bp is not None:
    app.register_blueprint(_probe_bp)
    print("REGISTER   : notices_bp registered")
else:
    print("REGISTER   : SKIPPED (notices import failed)")

if '_passes_bp' in globals() and _passes_bp is not None:
    app.register_blueprint(_passes_bp)
    print("REGISTER   : passes_bp registered")
else:
    print("REGISTER   : SKIPPED (passes import failed)")

print("== URL MAP (before adding debug routes) ==")
for r in app.url_map.iter_rules():
    print(r)

# ==============================
# 3) 디버그/점검용 라우트
# ==============================
@app.get("/__routes")
def __routes():
    """현재 앱에 등록된 모든 URL Rule 목록을 반환하여 라우트 등록 여부를 즉시 확인."""
    return jsonify(sorted([str(r) for r in app.url_map.iter_rules()]))

@app.get("/api/notices_probe")
def notices_probe():
    """블루프린트 문제가 있을 때도 server.py 자체 라우트가 살아있는지 확인하는 프로브."""
    return jsonify({"ok": True, "msg": "server.py route is alive"})

# ==============================
# 4) DB 연결 (전역 커넥션; 회원/로그인 등에서 사용)
# ==============================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "123456")
DB_NAME = os.getenv("DB_NAME", "lotbotsystem")

db = pymysql.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    cursorclass=pymysql.cursors.DictCursor,
    charset="utf8mb4",
    autocommit=False
)

# ==============================
# 5) OCR 엔진
# ==============================
ocr_engine = GoogleVisionPlate(api_key=api_key)

def _normalize_plate(s: str) -> str:
    return (s or "").strip().replace(" ", "")

# ---------- OCR 라우트 ----------
@app.post('/api/ocr/license_plate')
def ocr_license_plate():
    """
    multipart/form-data 로 'image' 파일과 (선택) user_id를 받음.
    인식 성공 시 car 테이블 upsert 후 car_id 반환
    """
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is missing'}), 400

    user_id = request.form.get('user_id', type=int)
    image_bytes = request.files['image'].read()

    plate_number = _normalize_plate(ocr_engine.recognize_plate(image_bytes))
    if not plate_number:
        return jsonify({'error': 'No license plate detected'}), 404

    try:
        car_id, created_new = upsert_car_and_get_id(plate_number, user_id=user_id)
        return jsonify({
            'message': 'ok',
            'plate_number': plate_number,
            'car_id': car_id,
            'created_new': created_new
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==============================
# 6) 주차 이벤트(입차/출차) 라우트
#   - services.parking_service 의 handle_entry/handle_exit 호출
#   - parkingevent/payment 테이블을 사용 (현재 스키마 유지)
# ==============================
@app.post("/api/parking/entry")
def api_parking_entry():
    """
    multipart/form-data:
      - image: 파일(필수)
      - allow_duplicate: "1"/"true"/"yes" 면 열린 이벤트 있어도 새로 생성 (기본 False)
      - gate/source/image_path/crop_path/ocr_confidence: 현재 스키마에 컬럼은 없지만 향후 확장 대비
    """
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is missing'}), 400

    image_bytes = request.files['image'].read()
    allow_duplicate = str(request.form.get("allow_duplicate", "")).strip().lower() in ("1", "true", "yes")

    try:
        res = svc_parking_entry(
            image_bytes,
            allow_duplicate=allow_duplicate,
            gate=request.form.get("gate"),
            source="gate",
            image_path=None,
            crop_path=None,
            ocr_confidence=None
        )
        return jsonify({"message": "ok", **res}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/api/parking/exit")
def api_parking_exit():
    """
    multipart/form-data:
      - image: 파일(필수)
      - payment_method: 기본 'card'
      - payment_success: 기본 true(1)
      - zero_fee_for_pass: 정기권이면 0원 처리 (기본 true)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is missing'}), 400

    image_bytes = request.files['image'].read()
    payment_method = request.form.get("payment_method", "card")
    payment_success = str(request.form.get("payment_success", "1")).strip().lower() in ("1", "true", "yes")
    zero_fee_for_pass = str(request.form.get("zero_fee_for_pass", "1")).strip().lower() in ("1", "true", "yes")

    try:
        res = svc_parking_exit(
            image_bytes,
            payment_method=payment_method,
            payment_success=payment_success,
            zero_fee_for_pass=zero_fee_for_pass
        )
        return jsonify({"message": "ok", **res}), 200
    except ValueError as ve:
        # 입차 기록 없음/이미 출차 등
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# 7) 기존 회원/로그인 API
# ==============================
@app.get('/hello')
def hello():
    return jsonify({"message": "Hello from lotbotserver!!"})

@app.get('/')
def index():
    return jsonify({"message": "Welcome to the Flask API"})

@app.post('/register')
def register():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    hashed_pw = generate_password_hash(password)
    try:
        db.ping(reconnect=True)
        with db.cursor() as cursor:
            cursor.execute("SELECT 1 FROM user WHERE username = %s", (username,))
            if cursor.fetchone():
                return jsonify({'error': 'Username already exists'}), 409
            cursor.execute("""
                INSERT INTO user (username, password_hash, email)
                VALUES (%s, %s, %s)
            """, (username, hashed_pw, email))
            db.commit()
            return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500

@app.post('/login')
def login():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')
    try:
        db.ping(reconnect=True)
        with db.cursor() as cursor:
            cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
            user = cursor.fetchone()
            if user and check_password_hash(user['password_hash'], password):
                return jsonify({
                    'message': 'Login successful',
                    'user': {
                        'user_id': user['user_id'],
                        'username': user['username'],
                        'user_role': user.get('user_role'),
                        'user_type': user.get('user_type')
                    }
                }), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.get('/user/<username>')
def get_user(username):
    try:
        db.ping(reconnect=True)
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT user_id, username, user_role, user_type, email,
                       membership_start_date, membership_end_date
                FROM user WHERE username = %s
            """, (username,))
            user = cursor.fetchone()
            if user:
                return jsonify({'user': user})
            else:
                return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==============================
# 8) 메인
# ==============================
if __name__ == '__main__':
    print("== FINAL URL MAP ==")
    for r in app.url_map.iter_rules():
        print(r)

    app.run(host='0.0.0.0', port=5000, debug=True)
