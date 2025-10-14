from flask import Blueprint, request, jsonify
from services.payment_service import confirm_subscription_payment
from services.payment_service import (confirm_subscription_payment, get_user_payment_history, record_leave_payment)

payment_bp = Blueprint('payment', __name__, url_prefix="/api/payment")

@payment_bp.route('/history/<int:user_id>', methods=['GET'])
def get_payment_history(user_id):
    payments = get_user_payment_history(user_id)
    return jsonify({
        "user_id": user_id,
        "payments": payments
    })

@payment_bp.route('/subscribe', methods=['POST'])
def confirm_subscription():
    data = request.json
    user_id = data.get('user_id')
    duration_days = data.get('duration_days')
    print("[DEBUG] 받은 데이터:", data)

    if not user_id:
        return jsonify({"error": "user_id가 누락되었습니다."}), 400
    if not duration_days:
        return jsonify({"error": "duration_days가 누락되었습니다."}), 400

    result = confirm_subscription_payment(user_id, duration_days)
    return jsonify(result)


@payment_bp.route('/leave', methods=['POST'])
def confirm_leave_payment():
    """
    출차 결제 성공 시 내역 등록 (회원/비회원 공용)
    """
    data = request.json
    user_id = data.get('user_id')  # 회원이면 int, 비회원이면 None
    amount = data.get('amount')
    duration = data.get('duration')

    print("[DEBUG] 출차 결제 데이터:", data)

    if not amount:
        return jsonify({"error": "amount가 누락되었습니다."}), 400

    # ✅ user_id가 없으면 None으로 처리 (비회원)
    result = record_leave_payment(user_id, amount)
    return jsonify(result), 200