import time
import cv2
from typing import Optional
from spot.utils.parking_tracker import get_slot_status
from spot.config import VIDEO_SOURCE, SLOT_ROIS

class ParkingService:
    """
    비즈니스 로직: • 비디오 프레임 순차 처리 • 빈/점유 슬롯 어노테이션 • 상태 반환
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not self.cap.isOpened():
            raise RuntimeError(f"비디오 소스 열기 실패: {VIDEO_SOURCE}")

    def get_current_slot_status(self) -> dict[int, bool] | None:
        """
        프레임을 읽어 각 슬롯의 점유 상태를 계산
        • 비디오 종료 시 None 반환
        • 어노테이션된 프레임을 화면에 표시
        """
        ret, frame = self.cap.read()
        # 비디오가 끝난 경우
        if not ret:
            return None

        # 슬롯 상태 계산
        statuses: dict[int, bool] = get_slot_status(frame)

        # 프레임 어노테이션
        self._annotate_frame(frame, statuses)

        # 화면 출력 (GUI 불가 환경 무시)
        try:
            cv2.imshow("Parking Slot Detection", frame)
            if cv2.waitKey(1) == 27:  # ESC
                raise KeyboardInterrupt
        except cv2.error:
            pass

        return statuses

    def _annotate_frame(self, frame: cv2.Mat, statuses: dict[int, bool]) -> None:
   
        for idx, occupied in statuses.items():
        # 인덱스가 ROI 개수보다 크면 스킵
            if idx >= len(SLOT_ROIS):
                continue

        x1, y1, x2, y2 = SLOT_ROIS[idx]
        # 빈자리→green, 점유→red
        color = (0, 255, 0) if not occupied else (0, 0, 255)
        label = "Empty" if not occupied else "Occupied"

        # 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 그림자 텍스트(가독성 향상)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # 그림자
            3,
            cv2.LINE_AA
        )
        # 실제 컬러 텍스트
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA
        )


    def release(self):
        if self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == '__main__':
    service = ParkingService()
    try:
        while True:
            result = service.get_current_slot_status()
            if result is None:
                print("▶ 비디오 끝, 처리 종료")
                break
            # print(result)
    except KeyboardInterrupt:
        print("사용자에 의해 중단됩니다.")
    finally:
        service.release()
        print("리소스 해제 완료.")
