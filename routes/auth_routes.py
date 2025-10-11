from flask import Blueprint, request, jsonify
from services.auth_service import (
    register_user, login_user, logout_user,
    update_user, delete_user, get_user,
    send_verification_code, verify_email_code
)
import jwt

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def get_token_from_header():
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None, jsonify({"error": "Authorization header missing"}), 401
    token = auth_header.split(" ")[1] if " " in auth_header else auth_header
    return token, None, None


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    print("Register 요청 도달:", data)
    response, status = register_user(data)
    return jsonify(response), status


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    print("로그인 요청 도달:", data, "헤더:", request.headers)
    response, status = login_user(data)
    return jsonify(response), status


@auth_bp.route("/logout", methods=["POST"])
def logout():
    response, status = logout_user()
    return jsonify(response), status


@auth_bp.route("/update/<int:user_id>", methods=["PUT"])
def update(user_id):
    data = request.get_json()
    print("Update 요청 도달:", data)
    response, status = update_user(user_id, data)
    return jsonify(response), status

@auth_bp.route("/check_user", methods=["POST"])
def check_user():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415

        data = request.get_json()
        email = data.get("email")
        phone_number = data.get("phone_number")

        if not email or not phone_number:
            return jsonify({"error": "이메일 또는 전화번호 누락"}), 400

        from services.auth_service import check_user_exists
        response, status = check_user_exists(email, phone_number)
        return jsonify(response), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/delete/<int:user_id>", methods=["DELETE"])
def delete(user_id):
    response, status = delete_user(user_id)
    return jsonify(response), status


@auth_bp.route("/user", methods=["GET"])
def get_user_info():
    try:
        token, err_resp, err_status = get_token_from_header()
        if err_resp:
            return err_resp, err_status

        # 배포 시 verify_signature=True와 비밀키 필요
        decoded = jwt.decode(token, options={"verify_signature": False})
        user_id = decoded.get("user_id")
        if not user_id:
            return jsonify({"error": "Invalid token"}), 401

        response, status = get_user(user_id)
        return jsonify(response), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/send_verification", methods=["POST"])
def send_verification():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400
        email = data.get("email")
        if not email:
            return jsonify({"error": "이메일 누락"}), 400

        response, status = send_verification_code(email)
        return jsonify(response), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/verify_code", methods=["POST"])
def verify_code():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400
        email = data.get("email")
        code = data.get("code")
        if not email:
            return jsonify({"error": "이메일 누락"}), 400
        if not code:
            return jsonify({"error": "코드 누락"}), 400

        response, status = verify_email_code(email, code)
        return jsonify(response), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@auth_bp.route("/reset_password", methods=["POST"])
def reset_password():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415

        data = request.get_json()
        email = data.get("email")
        new_password = data.get("password")

        if not email or not new_password:
            return jsonify({"error": "이메일 또는 비밀번호 누락"}), 400

        # 실제 비밀번호 재설정 로직 (DB 업데이트)
        from services.auth_service import reset_user_password
        response, status = reset_user_password(email, new_password)
        return jsonify(response), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# auth_routes.py

@auth_bp.route("/kakao_link", methods=["POST"])
def kakao_link():
    """
    Flutter에서 카카오 로그인 후 사용자 정보를 받아 DB에 kakao_auth 저장
    """
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        kakao_id = data.get("kakao_id")

        if not user_id or not kakao_id:
            return jsonify({"error": "user_id 또는 kakao_id 필요"}), 400

        from services.auth_service import link_kakao_account
        response, status = link_kakao_account(user_id, kakao_id)
        return jsonify(response), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/kakao_unlink", methods=["POST"])
def kakao_unlink():
    """
    Flutter에서 카카오 연동 해제 시 호출
    """
    try:
        data = request.get_json()
        user_id = data.get("user_id")

        if not user_id:
            return jsonify({"error": "user_id 필요"}), 400

        from services.auth_service import unlink_kakao_account
        response, status = unlink_kakao_account(user_id)
        return jsonify(response), status

    except Exception as e:
        return jsonify({"error": str(e)}, 500)
    
@auth_bp.route("/login/kakao", methods=["POST"])
def login_kakao():
    """
    Flutter에서 카카오 로그인 후 kakao_id를 받아
    로컬 DB 사용자와 매칭 후 JWT 발급
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415

        data = request.get_json()
        kakao_id = data.get("kakao_id")

        if not kakao_id:
            return jsonify({"error": "kakao_id 누락"}), 400

        from services.auth_service import login_with_kakao
        response, status = login_with_kakao(kakao_id)
        return jsonify(response), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500