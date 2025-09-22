from flask import Blueprint, request, jsonify
from services.payment_service import confirm_payment

payment_bp = Blueprint('payment', __name__)

@payment_bp.route('/api/payment/confirm', methods=['POST'])
def payment_confirm():
    data = request.json
    payment_key = data.get('paymentKey')
    order_id = data.get('orderId')
    amount = data.get('amount')
    if not all([payment_key, order_id, amount]):
        return jsonify({'error': '필수 파라미터 누락'}), 400
    try:
        result = confirm_payment(payment_key, order_id, amount)
        # TODO: 결제 결과 DB 저장 등 추가 가능
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400