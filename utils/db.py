# utils/db.py
from flask import g
import os
import pymysql

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "123456")
DB_NAME = os.getenv("DB_NAME", "lotbotsystem")

def get_db():
    """
    요청 단위로 MySQL 커넥션을 생성해 g에 저장.
    호출자는 with cursor() 사용 후 커밋/롤백을 수행.
    """
    if "db" not in g:
        g.db = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",     # ← 추가
            autocommit=False,      # 보통 False 유지
        )
        # 선택: 안정성을 위해 한 번 더 보정
        # with g.db.cursor() as cur:
        #     cur.execute("SET NAMES utf8mb4")
    return g.db

def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()
