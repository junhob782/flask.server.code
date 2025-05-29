##응답 유틸(response통일)

from flask import jsonify

def make_response(data, status=200):
    return jsonify({"success": True, "data": data}), status

def error_response(message, status=400):
    return jsonify({"success": False, "error": message}), status
