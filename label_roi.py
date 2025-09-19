import cv2

# 1) 영상 소스 경로 (spot/config.py 의 VIDEO_SOURCE 와 동일하게 설정)
VIDEO_SOURCE = r'C:\Users\hanhw\capstonedesign\lotbot_server\videos\1.mp4'

# 2) 클릭된 좌표를 저장할 리스트
coords = []

# 3) 마우스 콜백 함수: 왼쪽 버튼 클릭 시 (x,y)를 coords에 추가하고 출력
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        print(f"Clicked: {(x, y)}")

def main():
    # 비디오 캡처 후 첫 프레임 읽기
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("첫 프레임을 읽어올 수 없습니다.")
        return

    # 윈도우 생성 및 콜백 등록
    window_name = "Click to get ROI corners (press 'q' to finish)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("▶ ROI 좌표를 차례로 클릭하세요. (예: 왼쪽위→오른쪽아래 순으로 두 번 클릭 → 다음 슬롯 …)")
    print("▶ 완료되면 창에서 'q' 키를 누르세요.")

    while True:
        disp = frame.copy()
        # 클릭된 점들 시각화
        for (x, y) in coords:
            cv2.circle(disp, (x, y), 5, (0,255,0), -1)
        cv2.imshow(window_name, disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n최종 좌표 리스트:")
    for i, (x, y) in enumerate(coords):
        print(f"  {i}: ({x}, {y})")

if __name__ == "__main__":
    main()
