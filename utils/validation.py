## 입력 검증 유틸

def validate_image_file(image_file):
    if not image_file:
        return False
    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        return False
    # 파일 크기 제한 등 추가 가능
    return True
