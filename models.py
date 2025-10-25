from flask_sqlalchemy import SQLAlchemy
# 'db' 객체는 app.py에서 초기화된 후 사용됩니다.
db = SQLAlchemy()

# 'user' 테이블 모델 (Car 모델이 참조하므로 기본 형태 추가)
class User(db.Model):
    __tablename__ = 'user'
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    # ... (username, password 등 다른 컬럼들)
    
    # User와 Car의 관계 설정
    cars = db.relationship('Car', backref='owner', lazy=True)

class ParkingSpace(db.Model):
    __tablename__ = 'parkingspace'
    space_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    location_desc = db.Column(db.String(100))
    is_occupied = db.Column(db.Boolean, default=False)
    
    # ParkingSpace와 ParkingSpaceEvent의 관계 설정
    events = db.relationship('ParkingSpaceEvent', backref='space', lazy=True)

class Car(db.Model):
    __tablename__ = 'car'
    car_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    license_plate = db.Column(db.String(20), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'))
    
    # Car와 ParkingEvent, ParkingSpaceEvent의 관계 설정
    parking_events = db.relationship('ParkingEvent', backref='car', lazy=True)
    space_events = db.relationship('ParkingSpaceEvent', backref='car', lazy=True)

class ParkingEvent(db.Model):
    __tablename__ = 'parking_event'
    event_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    car_id = db.Column(db.Integer, db.ForeignKey('car.car_id'))
    license_plate = db.Column(db.String(20), nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime)
    gate = db.Column(db.String(50))
    image_path = db.Column(db.String(255))
    
    # ✨ --- 누락된 컬럼들 추가 --- ✨
    crop_path = db.Column(db.String(255))
    ocr_confidence = db.Column(db.Numeric(5, 2)) # Decimal은 Numeric으로 매핑
    source = db.Column(db.String(10)) # enum은 String으로 처리
    status = db.Column(db.String(10))
    notes = db.Column(db.String(255))
    # ✨ ------------------------- ✨
    
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())
    updated_at = db.Column(db.TIMESTAMP, server_default=db.func.now(), onupdate=db.func.now())
    
class ParkingSpaceEvent(db.Model):
    __tablename__ = 'parkingevent'
    event_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    car_id = db.Column(db.Integer, db.ForeignKey('car.car_id'))
    space_id = db.Column(db.Integer, db.ForeignKey('parkingspace.space_id'))
    license_plate = db.Column(db.String(20))
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime)