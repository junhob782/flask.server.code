import datetime
from utils.fee_calc import calculate_fee

def test_non_member_free():
    entry = datetime.datetime(2024, 5, 29, 10, 0, 0)
    exit = datetime.datetime(2024, 5, 29, 10, 25, 0)
    assert calculate_fee(entry, exit, 'non_member') == 0

def test_non_member_paid():
    entry = datetime.datetime(2024, 5, 29, 10, 0, 0)
    exit = datetime.datetime(2024, 5, 29, 11, 0, 0)
    # 60분 - 30분 무료 = 30분 × 100원 = 3000원
    assert calculate_fee(entry, exit, 'non_member') == 3000

def test_member_regular():
    entry = datetime.datetime(2024, 5, 29, 9, 0, 0)
    exit = datetime.datetime(2024, 5, 29, 10, 0, 0)
    # 60분 × 50원 = 3000원
    assert calculate_fee(entry, exit, 'member_regular') == 3000

def test_member_subscriber():
    entry = datetime.datetime(2024, 5, 29, 9, 0, 0)
    exit = datetime.datetime(2024, 5, 29, 18, 0, 0)
    # 월정액
    assert calculate_fee(entry, exit, 'member_subscriber') == 10000
