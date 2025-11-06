from flask import Flask, jsonify, request
from flask_cors import CORS
import os, sys
import pymysql
from dotenv import load_dotenv

# ==============================
# 스타트업 진단 로그
# ==============================
print("=== STARTUP DIAG ===")
print("CWD        :", os.getcwd())
print("SERVER_FILE:", __file__)
print("PYTHON     :", sys.executable)

# ==============================
# 블루프린트 임포트 진단
# ==============================
try:
    from notices import notices_bp as _notices_bp
    print("NOTICES_BP : imported OK")
except Exception as e:
    print("NOTICES_BP : import FAILED ->", repr(e))
    _notices_bp = None

try:
    from passes import passes_bp as _passes_bp
    print("PASSES_BP  : imported OK")
except Exception as e:
    print("PASSES_BP  : import FAILED ->", repr(e))
    _passes_bp = None

from routes.auth_routes import auth_bp
from routes.user_routes import user_bp
from routes.payment_route import payment_bp
from routes.parking_routes import bp as parking_bp

from utils.db import close_db
from utils.OCR_engines.ocr_googlevision import GoogleVisionPlate
from services.car_services import upsert_car_and_get_id
from services.parking_service import handle_entry as svc_parking_entry, handle_exit as svc_parking_exit

# ✅ payment_service에서 expire_expired_memberships 가져오기
from services.payment_service import expire_expired_memberships, run_scheduler_in_background

# ✅ SQLAlchemy 전역 db (models.py)
from models import db
from sqlalchemy import text

# ==============================
# 환경 변수 / 앱 생성
# ==============================
load_dotenv()

api_key = os.getenv("GOOGLE_VISION_API_KEY")
if not api_key:
    raise RuntimeError("환경변수 GOOGLE_VISION_API_KEY가 설정되지 않았습니다.")

TOSS_SECRET_KEY = os.getenv("TOSS_SECRET_KEY")
TOSS_CONFIRM_URL = "https://api.tosspayments.com/v1/payments/confirm"

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.teardown_appcontext(close_db)
CORS(app)

# ==============================
# DB 연결 설정 (SQLAlchemy + PyMySQL 병행)
# ==============================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "123456")
DB_NAME = os.getenv("DB_NAME", "lotbotsystem")

# ★ SQLAlchemy 설정 + 바인딩
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# (개발 편의) 최초 실행 시 연결 점검 (Flask-SQLAlchemy 3.x: bind가 None일 수 있으므로 실제 쿼리로 확인)
with app.app_context():
    # 필요 시 최초 생성
    # db.create_all()
    try:
        db.session.execute(text("SELECT 1"))
        print("[OK] SQLAlchemy connected & app context active")
    except Exception as e:
        print("[FATAL] SQLAlchemy connect failed:", repr(e))
        raise

# (선택) PyMySQL 로우 커넥션 필요 시 유지 (이름 충돌 방지)
pymysql_conn = pymysql.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASS,
    database=DB_NAME,
    cursorclass=pymysql.cursors.DictCursor,
    charset="utf8mb4",
    autocommit=False
)

# ==============================
# 블루프린트 등록 (prefix 없음 → /status 그대로 사용)
# ==============================
if _notices_bp: app.register_blueprint(_notices_bp)
if '_passes_bp' in globals() and _passes_bp: app.register_blueprint(_passes_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(payment_bp)
app.register_blueprint(parking_bp)

# ==============================
# OCR 엔진
# ==============================
ocr_engine = GoogleVisionPlate(api_key=api_key)

def _normalize_plate(s: str) -> str:
    return (s or "").strip().replace(" ", "")

@app.post('/api/ocr/license_plate')
def ocr_license_plate():
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
# 주차 이벤트 라우트 (게이트 API)
# ==============================
@app.post("/api/parking/entry")
def api_parking_entry():
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
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# 디버그/점검용 라우트
# ==============================
@app.get("/__routes")
def __routes():
    return jsonify(sorted([str(r) for r in app.url_map.iter_rules()]))

@app.get("/api/notices_probe")
def notices_probe():
    return jsonify({"ok": True, "msg": "server.py route is alive"})

@app.get('/hello')
def hello():
    return jsonify({"message": "Hello from lotbotserver!!"})

@app.get('/')
def index():
    return jsonify({"message": "Welcome to the Flask API"})

# ==============================
# 메인
# ==============================
if __name__ == '__main__':
    print("== FINAL URL MAP ==")
    for r in app.url_map.iter_rules():
        print(r)

    # 🔥 결제 서비스 만료 스케줄러 백그라운드 실행
    run_scheduler_in_background()

    app.run(host='0.0.0.0', port=3000, debug=True)