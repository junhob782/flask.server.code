import pymysql
from DB.connection import get_db
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
import random
from dotenv import load_dotenv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

MAIL_USER = os.getenv("MAIL_USER")
MAIL_PASS = os.getenv("MAIL_PASS")
MAIL_PORT = int(os.getenv("MAIL_PORT", 465))
MAIL_HOST = os.getenv("MAIL_HOST")

SECRET_KEY = "your_jwt_secret_key"

def get_user(user_id):
    db = get_db()
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            # 1) user 기본 정보 조회
            cursor.execute("""
                SELECT user_id, name, birth_date, phone_number, email, car_number, subscribe_membership, marketing_opt_in
                FROM user
                WHERE user_id=%s
            """, (user_id,))
            user = cursor.fetchone()
            if not user:
                return {"error": "사용자 없음"}, 404

            # 2) membership_user에서 해당 유저의 최신(종료일 기준) 멤버십 한 건 조회
            cursor.execute("""
                SELECT membership_start, membership_end
                FROM membership_user
                WHERE user_id=%s
                ORDER BY membership_end DESC
                LIMIT 1
            """, (user_id,))
            membership = cursor.fetchone()

            # 3) 날짜 포맷 처리 (있으면 문자열로)
            if user.get("birth_date"):
                # birth_date는 DATE 타입이라면 date 객체일 것
                user["birth_date"] = user["birth_date"].strftime("%Y-%m-%d")

            if membership:
                # membership_start / membership_end 도 DATE 타입이라면 포맷
                if membership.get("membership_start"):
                    user["membership_start"] = membership["membership_start"].strftime("%Y-%m-%d")
                else:
                    user["membership_start"] = None

                if membership.get("membership_end"):
                    user["membership_end"] = membership["membership_end"].strftime("%Y-%m-%d")
                else:
                    user["membership_end"] = None
            else:
                user["membership_start"] = None
                user["membership_end"] = None

            return {"user": user}, 200
    except Exception as e:
        return {"error": str(e)}, 500

# 회원가입
def register_user(data):
    name = data.get("name")
    birth_date = data.get("birth_date")  # YYYYMMDD
    phone = data.get("phone_number")
    email = data.get("email")
    password = data.get("password")
    car_number = data.get("car_number")
    marketing_opt_in = data.get("marketing_opt_in", False)

    if not name or not email or not password:
        return {"error": "필수 항목 누락"}, 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db = get_db()

    # 기본값 설정
    user_role = "user"
    subscribe_membership = False  # 이전 user_type → 이제 멤버십 여부를 나타내는 불린
    kakao_auth = False            # 카카오 연동 여부

    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT user_id FROM user WHERE email=%s", (email,))
            if cursor.fetchone():
                return {"error": "이미 존재하는 이메일"}, 409

            cursor.execute("""
                INSERT INTO user 
                (marketing_opt_in, name, birth_date, phone_number, email, password_hash, car_number,
                user_role, subscribe_membership, kakao_auth)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                marketing_opt_in, 
                name, 
                datetime.strptime(birth_date, "%Y%m%d"), 
                phone, 
                email, 
                hashed_pw, 
                car_number,
                user_role,
                subscribe_membership,
                kakao_auth,
            ))
            db.commit()
        return {"message": "회원가입 완료"}, 201
    except Exception as e:
        return {"error": str(e)}, 500

# 로그인
def login_user(data):
    email = data.get("email")
    password = data.get("password")

    db = get_db()
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("SELECT * FROM user WHERE email=%s", (email,))
            user = cursor.fetchone()

            if not user:
                return {"error": "이메일 또는 비밀번호 불일치"}, 401

            # ✅ bcrypt로 비밀번호 검증
            if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
                return {"error": "이메일 또는 비밀번호 불일치"}, 401

            # JWT 발급
            token = jwt.encode({
                "user_id": user["user_id"],
                "exp": datetime.utcnow() + timedelta(hours=6)
            }, SECRET_KEY, algorithm="HS256")

            return {"message": "로그인 성공", "token": token}, 200

    except Exception as e:
        return {"error": str(e)}, 500

# 로그아웃 (JWT는 클라이언트에서 삭제 처리)
def logout_user():
    return {"message": "로그아웃 성공"}, 200

# 회원 정보 수정
def update_user(user_id, data):
    db = get_db()
    fields = []
    values = []

    for col in ["name", "birth_date", "phone_number", "email", "password", "car_number"]:
        if data.get(col):
            if col == "password":
                fields.append("password_hash=%s")
                hashed_pw = bcrypt.hashpw(data[col].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                values.append(hashed_pw)
            elif col == "birth_date":
                fields.append("birth_date=%s")
                values.append(datetime.strptime(data[col], "%Y%m%d"))
            else:
                fields.append(f"{col}=%s")
                values.append(data[col])

    if not fields:
        return {"error": "수정할 항목이 없습니다"}, 400

    values.append(user_id)
    sql = f"UPDATE user SET {', '.join(fields)} WHERE user_id=%s"

    try:
        with db.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            db.commit()
        return {"message": "회원정보 수정 완료"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

# 회원 탈퇴
def delete_user(user_id):
    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("DELETE FROM user WHERE user_id=%s", (user_id,))
            db.commit()
        return {"message": "회원 탈퇴 완료"}, 200
    except Exception as e:
        return {"error": str(e)}, 500
    
def send_email(to_email, code):
    subject = "이메일 인증 코드"
    body = f"인증 코드: {code}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = MAIL_USER
    msg['To'] = to_email

    try:
        print(">>> SMTP 연결 시도:", MAIL_HOST, MAIL_PORT, MAIL_USER)
        with smtplib.SMTP_SSL(MAIL_HOST, MAIL_PORT) as server:
            server.login(MAIL_USER, MAIL_PASS)
            server.send_message(msg)
        print(">>> 메일 발송 성공")
    except Exception as e:
        print(f">>> 메일 발송 실패: {type(e).__name__} - {e}")
        raise e

def send_verification_code(email):
    code = str(random.randint(100000, 999999))
    expire_time = datetime.utcnow() + timedelta(minutes=10)

    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute("""
                INSERT INTO email_verification (email, code, expire_at)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE code=%s, expire_at=%s
            """, (email, code, expire_time, code, expire_time))
            db.commit()

        send_email(email, code)
        return {"message": "인증 코드 발송 완료"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

def verify_email_code(email, code):
    db = get_db()
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT code, expire_at FROM email_verification
                WHERE email=%s
            """, (email,))
            row = cursor.fetchone()
            if not row:
                return {"error": "인증 코드 없음"}, 404

            if row["expire_at"] < datetime.utcnow():
                return {"error": "인증 코드 만료"}, 400

            if row["code"] != code:
                return {"error": "인증 코드 불일치"}, 400

            return {"message": "이메일 인증 완료"}, 200
    except Exception as e:
        return {"error": str(e)}, 500
    
# 사용자 존재 여부 확인 (이메일 + 전화번호)
def check_user_exists(email, phone_number):
    db = get_db()
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT user_id, name, email, phone_number
                FROM user
                WHERE email=%s AND phone_number=%s
            """, (email, phone_number))
            user = cursor.fetchone()

            if user:
                # 사용자 존재
                return {"message": "사용자 존재", "user": user}, 200
            else:
                # 사용자 없음
                return {"error": "사용자 없음"}, 404
    except Exception as e:
        return {"error": str(e)}, 500
    
import bcrypt

def reset_user_password(email, new_password):
    try:
        from DB.connection import get_db
        conn = get_db()
        cursor = conn.cursor()

        # 이메일로 사용자 확인
        cursor.execute("SELECT user_id FROM user WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            return {"error": "해당 이메일의 사용자가 존재하지 않습니다."}, 404

        # ✅ bcrypt 해싱
        hashed_pw = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # DB 업데이트
        cursor.execute("UPDATE user SET password_hash = %s WHERE email = %s", (hashed_pw, email))
        conn.commit()

        return {"success": True, "message": "비밀번호 재설정 완료"}, 200
    except Exception as e:
        return {"error": str(e)}, 500