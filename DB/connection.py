#DB 연결 모듈

import pymysql
from config import DB_CONFIG

def get_db():   
    #DB접속하는 커넥션 반환 -> 입차 기록 조회, 
    # 출차 시간 업데이트, 
    #결제(payment) 테이블 삽입 등을 수행
    
    
    return pymysql.connect(
        host=DB_CONFIG['localhost'],
        user=DB_CONFIG['root'],
        password=DB_CONFIG['123456'],
        database=DB_CONFIG['lotbotsystem'],
        cursorclass=pymysql.cursors.DictCursor
    )