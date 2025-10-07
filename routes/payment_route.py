from flask import Blueprint, request, jsonify
from services.payment_service import confirm_subscription_payment

payment_bp = Blueprint('payment', __name__, url_prefix="/api/payment")

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