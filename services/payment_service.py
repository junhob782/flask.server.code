from DB.connection import get_db
from datetime import datetime, timedelta

def confirm_subscription_payment(user_id, duration_days):
    db = get_db()
    cursor = db.cursor()

    try:
        # 1️⃣ user 구독 상태 변경
        cursor.execute(
            "UPDATE user SET subscribe_membership = TRUE WHERE user_id = %s",
            (user_id,)
        )

        # 2️⃣ membership_user INSERT
        membership_start = datetime.now().date()
        membership_end = membership_start + timedelta(days=int(duration_days))

        cursor.execute("""
            INSERT INTO membership_user (user_id, membership_start, membership_end)
            VALUES (%s, %s, %s)
        """, (user_id, membership_start, membership_end))

        db.commit()

        return {
            "message": "정기권 결제가 완료되었습니다.",
            "user_id": user_id,
            "membership_start": str(membership_start),
            "membership_end": str(membership_end),
            "duration_days": duration_days
        }

    finally:
        cursor.close()
        db.close()