# utils/fee_calc.py
import math

def calculate_fee(entry_time, exit_time, user_type):
    duration_min = math.ceil((exit_time - entry_time).total_seconds() / 60)
    if user_type == "non_member":
        if duration_min <= 30:
            return 0
        else:
            return (duration_min - 30) * 100
    elif user_type == "member_regular":
        return duration_min * 50
    else:
        return 10000  # 월정액
