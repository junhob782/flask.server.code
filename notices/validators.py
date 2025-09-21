#밸리데이션

def nonempty_string(s: str, field: str):
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"{field} is required.")
    return s.strip()

def ensure_bool(v, field: str):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.lower().strip()
        if s in ("true", "1", "yes", "y"): return True
        if s in ("false", "0", "no", "n"): return False
    if isinstance(v, (int, float)):
        return bool(v)
    raise ValueError(f"{field} must be boolean.")