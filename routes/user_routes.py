from flask import Blueprint
from services.auth_service import get_user

user_bp = Blueprint('user', __name__)

@user_bp.route('/user/<username>', methods=['GET'])
def user_info(username):
    return get_user(username)