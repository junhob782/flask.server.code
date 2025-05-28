from flask import Flask
from flask_cors import CORS
from routes.auth_routes import auth_bp
from routes.user_routes import user_bp

app = Flask(__name__)
CORS(app)

# Blueprint 등록
app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)

@app.route('/hello', methods=['GET'])
def hello():
    return {"message": "Hello from Flask!"}

@app.route('/')
def index():
    return {"message": "Welcome to the Flask API"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    