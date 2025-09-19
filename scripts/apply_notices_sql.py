import os, pymysql, sys

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "123456")
DB_NAME = os.getenv("DB_NAME", "lotbotsystem")

sql_path = os.path.join(os.path.dirname(__file__), "..", "DB", "notices.sql")
sql_path = os.path.abspath(sql_path)

print(f"Applying SQL: {sql_path}")

conn = pymysql.connect(
    host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME,
    cursorclass=pymysql.cursors.DictCursor, autocommit=False
)
try:
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            cur.execute(stmt)
    conn.commit()
    print("✅ notices.sql 적용 완료")
except Exception as e:
    conn.rollback()
    print("❌ 적용 실패:", e)
    sys.exit(1)
finally:
    conn.close()
    