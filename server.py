from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql

app = Flask(__name__)
CORS(app)

# ✅ MySQL 연결
db = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    database='lotbotsystem',
    cursorclass=pymysql.cursors.DictCursor
)

# ✅ 핑테스트
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Flask!"})

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Flask API"})

# ✅ 회원가입 라우트
@app.route('/register', methods=['POST'])
def register():
    return handle_register()

# ✅ 로그인 라우트
@app.route('/login', methods=['POST'])
def login():
    return handle_login()

# ✅ 사용자 조회 라우트
@app.route('/user/<username>', methods=['GET'])
def get_user(username):
    return handle_get_user(username)

# ✅ 회원가입 로직 함수
def handle_register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    hashed_pw = generate_password_hash(password)

    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
            if cursor.fetchone():
                return jsonify({'error': 'Username already exists'}), 409

            cursor.execute("""
                INSERT INTO user (username, password_hash, email)
                VALUES (%s, %s, %s)
            """, (username, hashed_pw, email))
            db.commit()
            return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ 로그인 로직 함수
def handle_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT * FROM user WHERE username = %s", (username,))
            user = cursor.fetchone()

            if user and check_password_hash(user['password_hash'], password):
                return jsonify({
                    'message': 'Login successful',
                    'user': {
                        'user_id': user['user_id'],
                        'username': user['username'],
                        'user_role': user['user_role'],
                        'user_type': user['user_type']
                    }
                }), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ 사용자 조회 로직 함수
def handle_get_user(username):
    try:
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT user_id, username, user_role, user_type, email,
                       membership_start_date, membership_end_date
                FROM user WHERE username = %s
            """, (username,))
            user = cursor.fetchone()

            if user:
                return jsonify({'user': user})
            else:
                return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
