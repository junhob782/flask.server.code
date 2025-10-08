from DB.connection import get_db
from datetime import datetime, timedelta
import schedule
import time
import threading

def expire_expired_memberships(cursor=None):
    """
    만료된 정기권 자동 해제 (membership_end <= 오늘)
    - membership_user 삭제
    - user 테이블의 subscribe_membership FALSE로 갱신
    cursor가 None이면 DB 연결을 새로 생성
    """
    own_connection = False
    if cursor is None:
        db = get_db()
        cursor = db.cursor()
        own_connection = True

    today = datetime.now().date()

    cursor.execute("""
        SELECT membership_id, user_id
        FROM membership_user
        WHERE membership_end <= %s
    """, (today,))
    expired_users = cursor.fetchall()

    if expired_users:
        for row in expired_users:
            if isinstance(row, dict):
                membership_id = row["membership_id"]
                user_id = row["user_id"]
            else:
                membership_id = row[0]
                user_id = row[1]

            cursor.execute("DELETE FROM membership_user WHERE membership_id = %s", (membership_id,))
            cursor.execute("UPDATE user SET subscribe_membership = FALSE WHERE user_id = %s", (user_id,))

        print(f"[{datetime.now()}] 만료된 정기권 {len(expired_users)}건 자동 해제 완료")

    if own_connection:
        db.commit()
        cursor.close()
        db.close()


def confirm_subscription_payment(user_id, duration_days):
    db = get_db()
    cursor = db.cursor()

    try:
        # 기존 구독 조회
        cursor.execute("""
            SELECT membership_id, membership_start, membership_end
            FROM membership_user
            WHERE user_id = %s
            ORDER BY membership_end DESC
            LIMIT 1
        """, (user_id,))
        existing = cursor.fetchone()

        today = datetime.now().date()
        membership_start = today

        if not existing:
            membership_end = today + timedelta(days=int(duration_days))
            cursor.execute("""
                INSERT INTO membership_user (user_id, membership_start, membership_end)
                VALUES (%s, %s, %s)
            """, (user_id, membership_start, membership_end))
            cursor.execute("UPDATE user SET subscribe_membership = TRUE WHERE user_id = %s", (user_id,))
            status = "new"

        else:
            if isinstance(existing, dict):
                membership_id = existing["membership_id"]
                current_end = existing["membership_end"]
            else:
                membership_id = existing[0]
                current_end = existing[2]

            if current_end == today:
                cursor.execute("DELETE FROM membership_user WHERE membership_id = %s", (membership_id,))
                cursor.execute("UPDATE user SET subscribe_membership = FALSE WHERE user_id = %s", (user_id,))
                membership_end = None
                status = "expired_today"
            elif current_end > today:
                membership_end = current_end + timedelta(days=int(duration_days))
                cursor.execute("""
                    UPDATE membership_user
                    SET membership_end = %s
                    WHERE membership_id = %s
                """, (membership_end, membership_id))
                cursor.execute("UPDATE user SET subscribe_membership = TRUE WHERE user_id = %s", (user_id,))
                status = "extended"
            else:
                membership_end = today + timedelta(days=int(duration_days))
                cursor.execute("""
                    UPDATE membership_user
                    SET membership_start = %s, membership_end = %s
                    WHERE membership_id = %s
                """, (membership_start, membership_end, membership_id))
                cursor.execute("UPDATE user SET subscribe_membership = TRUE WHERE user_id = %s", (user_id,))
                status = "reactivated"

        db.commit()

        return {
            "message": "정기권 결제 처리 완료",
            "user_id": user_id,
            "membership_start": str(membership_start),
            "membership_end": str(membership_end) if membership_end else None,
            "duration_days": duration_days,
            "status": status
        }

    finally:
        cursor.close()
        db.close()


# --------------------------------------------
# 스케줄러: 매일 자정 자동 만료
# --------------------------------------------
def start_membership_expiration_scheduler():
    schedule.every(5).seconds.do(expire_expired_memberships)
    print("⏰ 테스트용 정기권 자동 만료 스케줄러 실행 중... (5초마다)")

    while True:
        schedule.run_pending()
        time.sleep(1)  # 1초마다 스케줄 체크

# 별도 스레드로 스케줄러 실행
def run_scheduler_in_background():
    scheduler_thread = threading.Thread(target=start_membership_expiration_scheduler, daemon=True)
    scheduler_thread.start()