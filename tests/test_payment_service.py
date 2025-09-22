import unittest
from services.payment_service import confirm_payment

class TestPaymentService(unittest.TestCase):
    def test_confirm_payment_real(self):
        payment_key = "여기에 paymentKey 입력"
        order_id = "여기에 orderId 입력"
        amount = 1000  # 결제 금액

        try:
            result = confirm_payment(payment_key, order_id, amount)
            print("결제 승인 결과:", result)
            self.assertEqual(result["status"], "DONE")
            self.assertEqual(result["orderId"], order_id)
            self.assertEqual(result["amount"], amount)
        except Exception as e:
            self.fail(f"결제 승인 실패: {e}")

if __name__ == '__main__':
    unittest.main()