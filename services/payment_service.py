import requests
import os
from dotenv import load_dotenv
import base64

load_dotenv()

TOSS_SECRET_KEY = os.getenv("TOSS_SECRET_KEY")
TOSS_CONFIRM_URL = "https://api.tosspayments.com/v1/payments/confirm"

def confirm_payment(payment_key, order_id, amount):
    # 시크릿 키를 base64로 인코딩
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
    
    # 응답 데이터 출력
    print("Response:", response.json())
    print("Authorization Header:", headers["Authorization"])
    
    return response.json()