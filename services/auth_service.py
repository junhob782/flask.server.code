from flask import jsonify
from db.connection import get_db
from werkzeug.security import generate_password_hash, check_password_hash

def register_user(data):
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    hashed_pw = generate_password_hash(password)

    db = get_db()
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


def login_user(data):
    username = data.get('username')
    password = data.get('password')

    db = get_db()
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


def get_user(username):
    db = get_db()
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