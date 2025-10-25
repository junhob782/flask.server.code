#실시간으로 분석하여 최신화 해주는 코드 (스캐쥴러)

# scheduler.py

import cv2
import requests
import io
from apscheduler.schedulers.background import BackgroundScheduler
from config import CCTV_SOURCE

# Flask 서버가 로컬에서 실행 중인 주소
API_URL = "http://127.0.0.1:5000/api/parking/update_cctv_status"

def analyze_cctv_frame():
    """
    CCTV 소스에서 프레임 하나를 캡처하여 분석 API로 전송하는 작업.
    이 함수가 5초마다 반복 실행됩니다.
    """
    print("[Scheduler] CCTV 프레임 분석 작업을 시작합니다...")
    
    cap = None  # 비디오 캡처 객체를 초기화
    try:
        # 1. 비디오 소스에서 프레임 한 장 캡처
        cap = cv2.VideoCapture(CCTV_SOURCE)
        if not cap.isOpened():
            print(f"[Scheduler Error] 비디오 소스를 열 수 없습니다: {CCTV_SOURCE}")
            return

        ret, frame = cap.read()
        if not ret:
            print("[Scheduler Info] 비디오의 끝에 도달했습니다. (파일 영상의 경우 정상)")
            # 영상 파일의 경우, 처음으로 되돌아가게 할 수도 있습니다.
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            return

        # 2. 캡처한 프레임(이미지)을 파일처럼 메모리에 변환
        # requests로 파일을 보내려면 실제 파일이거나, 파일처럼 보이는 메모리 객체여야 합니다.
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            print("[Scheduler Error] 프레임을 이미지 버퍼로 인코딩하는 데 실패했습니다.")
            return
        
        image_bytes = io.BytesIO(buffer)

        # 3. Postman이 하던 일을 코드가 대신 수행 (API 호출)
        # 'files' 파라미터는 form-data로 파일을 보내는 것과 동일합니다.
        files = {'image': ('cctv_frame.jpg', image_bytes, 'image/jpeg')}
        
        response = requests.post(API_URL, files=files)
        
        # 4. API 호출 결과 확인
        if response.status_code == 200:
            print(f"[Scheduler] 분석 API 호출 성공: {response.json()}")
        else:
            print(f"[Scheduler Error] 분석 API 호출 실패: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"[Scheduler Critical Error] 작업 실행 중 예외 발생: {e}")
    finally:
        if cap is not None:
            cap.release() # 작업이 끝나면 비디오 소스 연결을 반드시 해제


# --- 스케줄러 설정 및 시작 ---
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(analyze_cctv_frame, 'interval', seconds=5) # 5초 간격으로 `analyze_cctv_frame` 함수 실행

def start_scheduler():
    try:
        scheduler.start()
        print("백그라운드 스케줄러가 시작되었습니다. 5초마다 CCTV를 분석합니다.")
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("스케줄러가 종료되었습니다.")
        
