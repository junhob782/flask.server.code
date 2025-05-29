import pymysql
from config import DB_CONFIG

def get_db():
    return pymysql.connect(
        host=DB_CONFIG['localhost'],
        user=DB_CONFIG['root'],
        password=DB_CONFIG['123456'],
        database=DB_CONFIG['lotbotsystem'],
        cursorclass=pymysql.cursors.DictCursor
    )