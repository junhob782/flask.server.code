from flask import Flask, request
from flask_cors import CORS
from routes.auth_routes import auth_bp
from routes.user_routes import user_bp
from routes.payment_route import payment_bp
from routes.parking_routes import bp as parking_bp
from services.payment_service import confirm_payment
import requests
import os
from dotenv import load_dotenv
import base64

app = Flask(__name__)
CORS(app)

# Blueprint 등록
app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(parking_bp)
app.register_blueprint(payment_bp)

load_dotenv()

TOSS_SECRET_KEY = os.getenv("TOSS_SECRET_KEY")
TOSS_CONFIRM_URL = "https://api.tosspayments.com/v1/payments/confirm"

def confirm_payment(payment_key, order_id, amount):
    key = TOSS_SECRET_KEY + ":"
    encoded_key = base64.b64encode(key.encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {encoded_key}",
        "Content-Type": "application/json"
    }
    data = {
        "paymentKey": payment_key,
        "orderId": order_id,
        "amount": amount
    }
    response = requests.post(TOSS_CONFIRM_URL, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

@app.route('/hello', methods=['GET'])
def hello():
    return {"message": "Hello from Flask!"}

@app.route('/')
def index():
    return {"message": "Welcome to the Flask API"}

@app.route('/success', methods=['GET'])
def success():
    payment_key = request.args.get('paymentKey')
    order_id = request.args.get('orderId')
    amount = request.args.get('amount')

    # 결제 승인 API 호출
    try:
        result = confirm_payment(payment_key, order_id, amount)
        return f"결제 승인 성공! {result}"
    except Exception as e:
        return f"결제 승인 실패: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
