import pymysql
import os

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": False,  # 트랜잭션 제어
}

DB_NAME = "lotbotsystem"

def get_db():
    # DB 연결
    conn = pymysql.connect(**DB_CONFIG)

    # 데이터베이스 생성
    with conn.cursor() as cursor:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    conn.select_db(DB_NAME)

    # user 테이블 생성 확인
    with conn.cursor() as cursor:
        cursor.execute("SHOW TABLES LIKE 'user'")
        result = cursor.fetchone()
        if not result:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema_user.sql')
            with open(schema_path, 'r', encoding='utf8') as f:
                schema_sql = f.read()
            for statement in schema_sql.split(';'):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt)
            conn.commit()

    # membership_user 테이블 생성 확인
    with conn.cursor() as cursor:
        cursor.execute("SHOW TABLES LIKE 'membership_user'")
        result = cursor.fetchone()
        if not result:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema_MembershipUser.sql')
            with open(schema_path, 'r', encoding='utf8') as f:
                schema_sql = f.read()
            for statement in schema_sql.split(';'):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt)
            conn.commit()

    # payment_breakdown 테이블 생성 확인
    with conn.cursor() as cursor:
        cursor.execute("SHOW TABLES LIKE 'payment_breakdown'")
        result = cursor.fetchone()
        if not result:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema_payment_breakdown.sql')
            with open(schema_path, 'r', encoding='utf8') as f:
                schema_sql = f.read()
            for statement in schema_sql.split(';'):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt)
            conn.commit()

    # email_verification 테이블 생성 확인
    with conn.cursor() as cursor:
        cursor.execute("SHOW TABLES LIKE 'email_verification'")
        result = cursor.fetchone()
        if not result:
            cursor.execute("""
                CREATE TABLE email_verification (
                    email VARCHAR(255) NOT NULL PRIMARY KEY,
                    code VARCHAR(6) NOT NULL,
                    expire_at DATETIME NOT NULL
                )
            """)
            conn.commit()

    return conn