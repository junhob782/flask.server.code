# pixel_picker.py
# 좌표/ROI 픽셀 단위 선택 도구 (OpenCV)
# 사용 예:
#   python pixel_picker.py --video "C:\Users\hanhw\capstonedesign\lotbot_server\videos\1.mp4" --mode roi
#   python pixel_picker.py --video "C:\Users\hanhw\capstonedesign\lotbot_server\videos\1.mp4" --mode point

import cv2
import os, json, csv, glob, argparse

def parse_args():
    p = argparse.ArgumentParser(description="영상 프레임에서 픽셀 좌표/ROI를 클릭으로 수집")
    p.add_argument("--video", type=str, default=None,
                   help="비디오 파일 경로 (미지정 시 폴더에서 첫 영상 자동 탐색)")
    p.add_argument("--folder", type=str,
                   default=r"C:\Users\hanhw\capstonedesign\lotbot_server\videos\2.mp4",
                   help="비디오가 있는 폴더 경로")
    p.add_argument("--mode", type=str, choices=["point", "roi"], default="roi",
                   help="point=좌표 찍기, roi=사각형(두 점) 지정")
    p.add_argument("--output_dir", type=str, default=None,
                   help="결과 저장 폴더 (기본: 비디오와 동일 폴더)")
    return p.parse_args()

class Picker:
    def __init__(self, cap, mode, out_dir):
        self.cap = cap
        self.mode = mode
        self.out_dir = out_dir

        # 상태
        self.playing = False     # Space로 토글
        self.frame_idx = 0
        self.frame = None
        self.display = None

        # 수집 결과
        self.points = []   # [{frame,x,y}]
        self.rois = []     # [{id,frame,top_left:[x,y],bottom_right:[x,y]}]
        self.next_roi_id = 1

        # ROI 임시 상태(첫 클릭)
        self.first_pt = None
        self.mouse_xy = None

        # 저장 경로
        self.points_csv = os.path.join(self.out_dir, "points.csv")
        self.rois_json = os.path.join(self.out_dir, "slot_rois.json")

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_xy = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "point":
                self.points.append({"frame": self.frame_idx, "x": int(x), "y": int(y)})
                print(f"[POINT] frame={self.frame_idx}  (x={x}, y={y})  -> 저장됨")
            else:  # roi
                if self.first_pt is None:
                    self.first_pt = (x, y)  # 첫 점(좌상단 권장)
                    print(f"[ROI] 첫 점 선택: (x={x}, y={y})  두 번째 점을 클릭하세요(우하단).")
                else:
                    x1, y1 = self.first_pt
                    x2, y2 = x, y
                    # 좌상단/우하단 정렬
                    tl = (min(x1, x2), min(y1, y2))
                    br = (max(x1, x2), max(y1, y2))
                    record = {
                        "id": self.next_roi_id,
                        "frame": self.frame_idx,
                        "top_left": [int(tl[0]), int(tl[1])],
                        "bottom_right": [int(br[0]), int(br[1])]
                    }
                    self.rois.append(record)
                    print(f"[ROI] id={self.next_roi_id} 저장: TL={record['top_left']} BR={record['bottom_right']} (frame={self.frame_idx})")
                    self.next_roi_id += 1
                    self.first_pt = None  # 초기화

    def draw_overlay(self, img):
        # 안내 텍스트
        guide = [
            f"MODE: {self.mode.upper()}   frame: {self.frame_idx}",
            "LeftClick: 좌표 찍기 / ROI 두 점 찍기",
            "Space: 재생/일시정지 | n: 다음 프레임 | c: 현재 프레임 표시 초기화",
            "s: 저장 | q: 종료"
        ]
        y0 = 24
        for line in guide:
            cv2.putText(img, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            y0 += 24

        # 포인트 표시
        for p in self.points:
            if p["frame"] == self.frame_idx:
                cv2.circle(img, (p["x"], p["y"]), 4, (0, 255, 0), -1)
                cv2.putText(img, f"({p['x']},{p['y']})", (p["x"]+6, p["y"]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # ROI 표시
        for r in self.rois:
            if r["frame"] == self.frame_idx:
                tl = tuple(r["top_left"])
                br = tuple(r["bottom_right"])
                cv2.rectangle(img, tl, br, (0, 200, 255), 2)
                cv2.putText(img, f"ID {r['id']}", (tl[0], tl[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)

        # ROI 미리보기(첫 점 찍은 상태에서 마우스 움직임)
        if self.mode == "roi" and self.first_pt is not None and self.mouse_xy is not None:
            x1, y1 = self.first_pt
            x2, y2 = self.mouse_xy
            tl = (min(x1, x2), min(y1, y2))
            br = (max(x1, x2), max(y1, y2))
            cv2.rectangle(img, tl, br, (255, 0, 0), 1)
            cv2.circle(img, (x1, y1), 4, (255, 0, 0), -1)
            cv2.putText(img, f"TL ({tl[0]},{tl[1]})  BR ({br[0]},{br[1]})",
                        (tl[0], tl[1]-24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)

        # 현재 마우스 좌표 툴팁
        if self.mouse_xy is not None:
            mx, my = self.mouse_xy
            cv2.putText(img, f"({mx},{my})", (mx+10, my+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

    def save_results(self):
        if self.mode == "point":
            if len(self.points) == 0:
                print("[SAVE] 저장할 포인트가 없습니다.")
                return
            # CSV 저장
            with open(self.points_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["frame","x","y"])
                w.writeheader()
                for p in self.points:
                    w.writerow(p)
            print(f"[SAVE] 포인트 {len(self.points)}개 저장: {self.points_csv}")
        else:
            if len(self.rois) == 0:
                print("[SAVE] 저장할 ROI가 없습니다.")
                return
            with open(self.rois_json, "w", encoding="utf-8") as f:
                json.dump(self.rois, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] ROI {len(self.rois)}개 저장: {self.rois_json}")

    def run(self):
        cv2.namedWindow("Picker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Picker", self.on_mouse)

        # 첫 프레임 로드
        ok, self.frame = self.cap.read()
        if not ok:
            print("비디오를 읽을 수 없습니다.")
            return

        while True:
            if self.playing:
                ok, self.frame = self.cap.read()
                if not ok:
                    print("영상 끝에 도달했습니다.")
                    self.playing = False
                else:
                    self.frame_idx += 1

            self.display = self.frame.copy()
            self.draw_overlay(self.display)
            cv2.imshow("Picker", self.display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space
                self.playing = not self.playing
            elif key == ord('n'):  # next frame (단일 스텝)
                self.playing = False
                ok, self.frame = self.cap.read()
                if ok:
                    self.frame_idx += 1
                else:
                    print("다음 프레임이 없습니다.")
            elif key == ord('c'):  # clear temp on current frame
                self.first_pt = None
                print("[CLEAR] 현재 프레임의 임시 상태를 초기화했습니다.")
            elif key == ord('s'):
                self.save_results()

        cv2.destroyAllWindows()

def find_default_video(folder):
    exts = ("*.mp4","*.avi","*.mov","*.mkv")
    for ext in exts:
        files = glob.glob(os.path.join(folder, ext))
        if files:
            return files[0]
    return None

def main():
    args = parse_args()
    video_path = args.video
    if video_path is None:
        video_path = find_default_video(args.folder)
        if video_path is None:
            print("비디오를 찾지 못했습니다. --video 로 경로를 지정하세요.")
            return
        print(f"[AUTO] 첫 번째 비디오를 사용: {video_path}")

    out_dir = args.output_dir or os.path.dirname(video_path)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 열기 실패: {video_path}")
        return

    picker = Picker(cap, mode=args.mode, out_dir=out_dir)
    picker.run()
    cap.release()
    # 종료 시 자동 저장(선택)
    picker.save_results()

if __name__ == "__main__":
    main()
