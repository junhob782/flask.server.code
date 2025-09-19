import pymysql

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "lotbotsystem",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": False,  # 트랜잭션 제어를 위해
}

def get_db():
    return pymysql.connect(**DB_CONFIG)