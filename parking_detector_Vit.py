# parking_detector_ViT.py

import cv2
import requests # API 호출을 위한 라이브러리
import io       # 메모리 내에서 파일을 다루기 위한 라이브러리
import time     # 지연 시간을 주기 위한 라이브러리

# --- 사용자 설정 ---
# 분석할 원본 비디오 파일의 경로
VIDEO_SOURCE = r"C:\Users\hanhw\capstonedesign\lotbot_server\videos\1.mp4" 

# Flask API 서버의 주소
API_URL = "http://127.0.0.1:5000/api/parking/update_cctv_status"

# --- 메인 로직 ---
def analyze_and_update_db():
    """
    비디오 파일을 프레임 단위로 읽어, 주기적으로 Flask API 서버로 전송하여
    데이터베이스의 주차 상태를 업데이트합니다.
    """
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {VIDEO_SOURCE}")

    print("\n📹 비디오 분석 및 DB 업데이트를 시작합니다...")
    frame_idx = 0
    while True:
        # 1. 비디오에서 프레임 1장을 읽어옵니다.
        ok, frame = cap.read()
        
        # 비디오가 끝나면 루프를 종료합니다.
        if not ok:
            print("\n✅ 비디오의 끝에 도달했습니다. 모든 프레임 처리를 완료했습니다.")
            break

        # ✨ --- 프레임 건너뛰기 로직 (40 프레임마다 한 번씩) --- ✨
        # 현재 프레임 번호(frame_idx)가 40의 배수가 아니면,
        # 아래 분석 로직을 건너뛰고 다음 프레임으로 바로 넘어갑니다.
        if frame_idx % 40 != 0:
            frame_idx += 1
            continue # 다음 루프로 즉시 이동
        # ✨ -------------------------------------------------- ✨

        # 2. 현재 프레임을 이미지 파일처럼 메모리에 인코딩합니다.
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            print(f"[{frame_idx}] 프레임 인코딩 실패. 이 프레임을 건너뜁니다.")
            frame_idx += 1
            continue
            
        image_bytes = io.BytesIO(buffer)

        # 3. Flask API 서버로 현재 프레임(이미지)을 전송합니다.
        try:
            # Postman의 form-data와 동일한 역할을 합니다.
            files = {'image': ('current_frame.jpg', image_bytes, 'image/jpeg')}
            
            # API에 POST 요청을 보냅니다.
            response = requests.post(API_URL, files=files, timeout=10) # 10초 타임아웃 설정

            # API 응답 결과를 터미널에 출력합니다.
            if response.status_code == 200:
                print(f"[{frame_idx}] 프레임 분석 및 DB 업데이트 성공: {response.json()}")
            else:
                print(f"[{frame_idx}] 👎 API 호출 실패: {response.status_code} - {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"📡 API 서버에 연결할 수 없습니다: {e}")
            print("Flask 서버(app.py)가 실행 중인지 먼저 확인하세요.")
            break # 서버가 꺼져있으면 더 이상 진행할 수 없으므로 중단합니다.

        frame_idx += 1
        time.sleep(1) # 다음 분석까지 1초 대기 (선택 사항)

    # --- 마무리 ---
    cap.release()
    print("\n분석 프로그램이 종료되었습니다.")

if __name__ == '__main__':
    # 'requests' 라이브러리가 설치되어 있는지 확인합니다.
    try:
        import requests
    except ImportError:
        print("🛑 오류: 'requests' 라이브러리가 설치되지 않았습니다.")
        print("터미널에 'pip install requests'를 입력하여 설치해주세요.")
    else:
        analyze_and_update_db()