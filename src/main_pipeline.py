"""
main_pipeline.py — v4
Pipeline phát hiện vi phạm giao thông từ góc nhìn dashcam.
Tích hợp: YOLOv11s + BYTETracker + LaneDetector + ViolationConfirmer.

FIXES v4 (so với v3):
════════════════════════════════════════════════════════════════

[DIST-1] STABILIZATION — Số khoảng cách nhảy loạn
  Vấn đề: buffer median size=7 quá nhỏ → 1 bbox nhiễu làm lệch median ngay.
  Fix:
    - Tăng dist_buffer maxlen 7 → 15
    - Thêm EMA (α=0.25) song song median: dist_ema = α*raw + (1-α)*prev_ema
    - Output cuối = 0.6*median + 0.4*ema → vừa robust vừa mượt
    - Clamp delta giữa 2 frame liên tiếp: max thay đổi = MAX_DIST_JUMP_M (5m/frame)
      → loại bỏ spike đột ngột do bbox bị detect sai kích thước 1 frame

[DIST-2] UNDERESTIMATE — Xe gần bị báo xa hơn thực
  Nguyên nhân gốc: bbox_h bị shrink khi xe không fill hết box (YOLO pad),
  và focal_factor=0.85 calibrate cho 640px nhưng ta đang dùng 480px imgsz.
  Fix:
    - Tính effective_focal từ imgsz thực tế thay vì hardcode frame_h*0.85:
        focal = (IMG_SIZE / frame_h) * frame_h * FOCAL_FACTOR
      = IMG_SIZE * FOCAL_FACTOR  (độc lập với frame_h, đúng hơn khi resize)
    - Thêm BBOX_H_SHRINK_FACTOR = 0.82: bbox_h thực = bbox_h * shrink
      (bù cho việc YOLO box thường to hơn thân xe ~18%)
    - Tách per-class FOCAL_FACTOR_OVERRIDE: Car/Moto khác Bus/Truck do tỉ lệ
      chiều cao/chiều dài khác nhau gây sai số systematic khác nhau

[DIST-3] FAR SPAM — Xe rõ vẫn hiện FAR
  Nguyên nhân: MIN_BBOX_H_PX=12 quá cao so với xe ở 30-40m (bbox_h ≈ 20-30px
  ở 1080p nhưng sau resize 480px chỉ còn ~10px).
  Fix:
    - MIN_BBOX_H_PX tính động theo tỉ lệ resize: = 12 * (IMG_SIZE/640)
    - Khi dist_ok=False nhưng có buffer cũ → hiện giá trị buffer thay vì FAR
      (chỉ hiện FAR khi buffer rỗng hoàn toàn)

[FPS-1] LANE DETECTOR BOTTLENECK — Nguyên nhân chính <10 FPS
  LaneDetector (UFLD) chạy mỗi heavy frame = ~40-80ms/frame trên GPU.
  Fix: LANE_SKIP_FACTOR = 3 — lane chỉ update mỗi 3 heavy frame.
  Giữa các lần update: dùng cache + linear extrapolation cho lane drift.
  Tiết kiệm ~60% thời gian lane detection.

[FPS-2] YOLO WARMUP — Cold start penalty
  Lần inference đầu tiên của YOLO tốn 3-5x so với steady state (CUDA JIT).
  Fix: chạy warmup dummy inference trước khi mở video.

[FPS-3] FRAME SKIP AGGRESSIVE — FRAME_SKIP 2 → 3
  Với camera 30fps, skip 1 frame vẫn xử lý 15fps heavy.
  Skip 2 frame → xử lý 10fps heavy, hoàn toàn đủ cho detection vi phạm.
  Cached result được giữ nguyên, drawing vẫn chạy 30fps.

[FPS-4] IMSHOW THROTTLE — imshow gây block khi queue dài
  cv2.imshow + waitKey tốn 2-5ms mỗi lần gọi do X11/GTK sync.
  Fix: chỉ imshow mỗi DISPLAY_SKIP_FACTOR frame (default=2).

[FPS-5] CONTIGUOUS NUMPY — BYTETracker tensor copy overhead
  torch.stack() → .cpu().numpy() tạo non-contiguous array trên một số build.
  Fix: thêm .ascontiguousarray() để đảm bảo contiguous trước khi truyền tracker.

[FPS-6] YOLO STREAM MODE — giảm overhead giữa các frame
  Dùng model.predict() với stream=True thay vì model() mỗi frame riêng lẻ
  → giảm Python object allocation overhead ~5-10%.

[FPS-7] PROFILING HOOK — đo từng stage
  Thêm StageTimer để in breakdown FPS theo stage sau khi xử lý xong:
  Lane / YOLO / Tracker / Logic / Draw — biết đúng bottleneck còn lại.
════════════════════════════════════════════════════════════════
"""

import os
import sys
import cv2
import time
import threading
import queue
import torch
import numpy as np
from collections import deque, defaultdict
from typing import Optional, Tuple, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from byte_tracker import BYTETracker
    from violation_classifier import ViolationConfirmer, ViolationType, CONFIRM_FRAMES
    from lane_detector import LaneDetector
except ImportError as e:
    print(f"[FATAL] Khong the import module noi bo: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("[FATAL] Chua cai ultralytics.")
    sys.exit(1)


# ══════════════════════════════════════════
# CẤU HÌNH TẬP TRUNG
# ══════════════════════════════════════════
VIDEO_PATH   = "demo_input2.mp4"
OUTPUT_VIDEO = os.path.join("output", "output_annotated.avi")

# [FPS-1] Xử lý 1 frame, skip N-1 frame (tăng tốc video playback)
FRAME_SKIP          = 3
# [FPS-4] imshow mỗi N frame để tránh sync overhead
DISPLAY_SKIP_FACTOR = 2
# [FPS-1] Lane chỉ update mỗi N heavy frame (tăng lên 5)
LANE_SKIP_FACTOR    = 5
# [FPS-8] Throttle: chỉ gửi frame cho lane worker mỗi N heavy frame.
# Lane worker (UFLD ~380ms/CPU) nếu chạy liên tục sẽ giành core với YOLO
# → YOLO chậm lại. Throttle giúp YOLO có những frame chạy "một mình" (nhanh),
# trong khi lane vẫn update ~1.5 lần/giây (đủ vì lane đã được EMA làm mượt).
LANE_SUBMIT_EVERY   = 10

# [RT-1] Realtime playback: pace vòng lặp hiển thị theo đồng hồ thực để video
# chạy đúng tốc độ thật (mượt ~vid_fps). Detection chạy async nền nên main loop
# nhẹ, đủ sức giữ nhịp realtime. Đặt False để xử lý nhanh nhất có thể (benchmark).
REALTIME_PLAYBACK   = True
# Gửi frame cho detector mỗi N frame hiển thị (1 = mỗi frame; worker tự drop frame cũ)
DETECT_SUBMIT_EVERY = 1

IMG_SIZE = 640   # Đã đổi lên 640 để khớp 100% với kích thước ảnh lúc train, giúp model nhìn rõ xe ở xa hơn

# Tốc độ giả định của xe (km/h) - Dùng để tính khoảng cách an toàn theo luật VN
EGO_SPEED_KMH = 60

# Luật giao thông Việt Nam (Thông tư 31/2019/TT-BGTVT):
if EGO_SPEED_KMH <= 60:
    SAFE_DISTANCE_M = 35.0
elif EGO_SPEED_KMH <= 80:
    SAFE_DISTANCE_M = 55.0
elif EGO_SPEED_KMH <= 100:
    SAFE_DISTANCE_M = 70.0
else:
    SAFE_DISTANCE_M = 100.0
UNSAFE_DISTANCE_DELTA   = 0.5
UNSAFE_LOOKBACK_FRAMES  = 3
YOLO_CONF               = 0.02   # Hạ cực thấp để bắt được cả xe bị mờ/biến dạng do đâm đụng
YOLO_IOU                = 0.45   # NMS IoU — thấp hơn mặc định 0.7 để gộp box trùng của cùng 1 xe
MAX_DET                 = 60     # Giới hạn số detection/frame (đủ cho cảnh giao thông)

# [LOWLIGHT] Tăng sáng/contrast cho ảnh đưa vào YOLO (KHÔNG ảnh hưởng hiển thị/output).
# Áp dụng CLAHE trên kênh L (LAB) + gamma nhẹ → giúp model nhìn rõ xe trong cảnh
# tối/backlight (vd dashcam ban đêm). Thêm ~2-3ms/lần infer trên CPU.
# Đặt False để tắt hoàn toàn.
LOWLIGHT_ENHANCE        = True
LOWLIGHT_CLAHE_CLIP     = 2.5    # 2.0=nhẹ, 3.0=mạnh (dễ nhiễu nếu cao quá)
LOWLIGHT_CLAHE_GRID     = 8      # kích thước ô CLAHE
LOWLIGHT_GAMMA          = 0.85   # <1.0 = sáng vùng tối; 1.0 = không đổi
# Camera
CAMERA_HEIGHT   = 1.6
FOCAL_FACTOR    = 0.85

# [DIST-2] Bù bbox padding của YOLO (box thường ~18% lớn hơn thân xe)
BBOX_H_SHRINK_FACTOR = 0.82

# [DIST-2] Per-class focal override (Car/Moto thường đúng hơn với base factor)
FOCAL_OVERRIDE: Dict[int, float] = {
    2: 0.85,   # Car
    3: 0.80,   # Moto — bbox YOLO thường overlap bánh, cần factor nhỏ hơn
    5: 0.88,   # Bus
    7: 0.88,   # Truck
}

# [DIST-3] Tính động theo resize ratio
MIN_BBOX_H_PX   = max(6, int(12 * IMG_SIZE / 640))
MIN_DIST_Y_REL  = 8.0

MAX_RELIABLE_DIST = 80.0

# [DIST-1] Giới hạn jump khoảng cách giữa 2 frame liên tiếp
MAX_DIST_JUMP_M  = 5.0
# [DIST-1] EMA alpha: nhỏ hơn = mượt hơn nhưng lag hơn
DIST_EMA_ALPHA   = 0.25
# [DIST-1] Blend weight: median vs EMA
DIST_MEDIAN_W    = 0.60
DIST_EMA_W       = 0.40

# Thêm logic tự động nhận diện model để map Class ID cho đúng
YOLO_MODEL   = "runs/detect/fast_adas_model/weights/best.pt"
# YOLO_MODEL   = "yolo11n.pt"


if "best.pt" in YOLO_MODEL or "fast_adas" in YOLO_MODEL:
    # Class IDs của Udacity Dataset mới train
    VEHICLE_REAL_HEIGHT: Dict[int, float] = {
        1: 1.45,  # Car
        10: 3.50, # Truck
        0: 1.10,  # Biker
    }
    VEHICLE_CLASSES = {1: "Car", 10: "Truck", 0: "Moto"}
else:
    # Class IDs của YOLO COCO mặc định
    VEHICLE_REAL_HEIGHT: Dict[int, float] = {
        2: 1.45, 3: 1.10, 5: 3.20, 7: 3.50,
    }
    VEHICLE_CLASSES = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}

HYBRID_NEAR_THRESH = 30.0
HYBRID_FAR_THRESH  = 50.0

CUT_IN_DELTA_X           = 50
CUT_IN_FRAMES            = 10
ENABLE_OUTSIDE_LANE      = True   # Bật tính năng phát hiện xe khác lấn làn/đè vạch
MIN_BBOX_AREA            = 5000
EGO_FRONT_ZONE_RATIO     = 0.35
EGO_LANE_CENTER_MARGIN   = 0.05
CUT_IN_DISTANCE_THRESH   = 12.0
CUT_IN_DISTANCE_DELTA    = 2.5
CUT_IN_LATERAL_SPEED     = 4.0
CUT_IN_ENTER_CONFIRM     = 3
CUT_IN_PREV_OUT_MIN      = 2
CUT_IN_CENTER_APPROACH   = 25

ENABLE_SIDE_INTRUSION       = True
SIDE_INTRUSION_DIST_THRESH  = 8.0
SIDE_INTRUSION_MIN_AREA     = 18000
SIDE_INTRUSION_MIN_Y2_RATIO = 0.52

SCENE_CHANGE_THRESH       = 0.02
SCENE_CHANGE_RESET_THRESH = SCENE_CHANGE_THRESH * 15

COLOR_DANGER  = (0,   0,   255)
COLOR_WARNING = (0,   165, 255)
COLOR_SAFE    = (0,   220,  60)
FLASH_FRAMES      = 30
TEXT_FLASH_FRAMES = 60


# ══════════════════════════════════════════
# STAGE TIMER — profiling từng bước  [FPS-7]
# ══════════════════════════════════════════
class StageTimer:
    def __init__(self, stages: List[str], report_every: int = 100):
        self._stages      = stages
        self._report_every = report_every
        self._acc: Dict[str, float] = {s: 0.0 for s in stages}
        self._count = 0
        self._t0: Dict[str, float] = {}

    def start(self, stage: str):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._t0[stage] = time.perf_counter()

    def stop(self, stage: str):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._acc[stage] += time.perf_counter() - self._t0.get(stage, time.perf_counter())

    def tick(self):
        self._count += 1
        if self._count % self._report_every == 0:
            total = max(sum(self._acc.values()), 1e-6)
            parts = "  ".join(
                f"{s}: {self._acc[s]/self._count*1000:.1f}ms"
                for s in self._stages
            )
            fps_heavy = self._count / max(total, 1e-6)
            print(f"[PROFILE] {parts}  | heavy_fps≈{fps_heavy:.1f}")
            self._acc = {s: 0.0 for s in self._stages}
            self._count = 0


class TrackerArgs:
    track_thresh = 0.05   # Ngưỡng cực thấp để re-activate track cho xe bị mờ lúc va chạm
    track_buffer = 45     # giữ track lâu hơn khi xe bị che tạm thời → ID ổn định hơn
    match_thresh = 0.85   # nới ngưỡng match để bám xe tốt hơn giữa các frame
    mot20        = False


# ══════════════════════════════════════════
# ASYNC VIDEO WRITER
# ══════════════════════════════════════════
class AsyncVideoWriter:
    def __init__(self, path: str, fourcc, fps: float, size: Tuple[int, int]):
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        self._q: queue.Queue = queue.Queue(maxsize=32)
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            self._writer.write(item)
            self._q.task_done()

    def write(self, frame: np.ndarray):
        try:
            self._q.put_nowait(frame.copy())
        except queue.Full:
            pass  # Drop frame thay vì block

    def release(self):
        self._q.put(None)
        self._t.join()
        self._writer.release()


# ══════════════════════════════════════════
# ASYNC LANE PROCESSOR  [FPS-8]
# ══════════════════════════════════════════
class AsyncLaneProcessor:
    """
    [FPS-8] Chạy LaneDetector trên thread riêng để KHÔNG block main loop.

    Vấn đề cũ: UFLD inference trên CPU tốn ~380ms/lần. Dù đã skip
    (LANE_SKIP_FACTOR=5), mỗi lần lane update vẫn làm main loop đứng hình
    ~0.4s → giật định kỳ + tụt FPS trung bình.

    Fix: worker thread luôn xử lý frame MỚI NHẤT (queue maxlen=1, drop frame
    cũ). Main loop chỉ submit (copy frame ~1ms) + đọc kết quả cache gần nhất
    → main loop chạy ở tốc độ YOLO (~20 heavy fps) thay vì bị lane kéo xuống 8.
    Lane vẫn tự cập nhật ~2-3 lần/giây trên frame mới nhất → đủ mượt cho UI.
    """

    def __init__(self, lane_detector, vid_h: int):
        self._ld              = lane_detector
        self._default_horizon = int(vid_h * 0.45)
        self._lock            = threading.Lock()
        self._latest: Tuple   = (None, None, self._default_horizon, [])
        self._q: queue.Queue  = queue.Queue(maxsize=1)
        self._reset_flag      = False
        self._stop            = False
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def submit(self, frame: np.ndarray, frame_id: int):
        """Đẩy frame mới nhất vào worker; drop frame cũ chưa xử lý."""
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put_nowait((frame.copy(), frame_id))
        except queue.Full:
            pass

    def get_latest(self) -> Tuple:
        """Đọc kết quả lane gần nhất (ll, rl, horizon_y, lanes_all)."""
        with self._lock:
            return self._latest

    def reset(self):
        """Báo worker reset smoother (gọi khi scene change lớn)."""
        self._reset_flag = True

    def _worker(self):
        while not self._stop:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            frame, frame_id = item
            if frame is None:
                break
            if self._reset_flag:
                self._reset_flag = False
                try:
                    self._ld.reset_smoother()
                except Exception:
                    pass
            try:
                ll, rl    = self._ld.get_ego_lane_boundaries(frame, frame_id=frame_id)
                lanes_all = self._ld.detect(frame, frame_id=frame_id)
                hy        = self._ld.get_vanishing_point(lanes_all)
                horizon   = hy if hy > 0 else self._default_horizon
                with self._lock:
                    self._latest = (ll, rl, horizon, lanes_all)
            except Exception as e:
                print(f"[WARN] Async lane frame {frame_id}: {e}")

    def stop(self):
        self._stop = True
        try:
            self._q.put_nowait((None, -1))
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait((None, -1))
            except Exception:
                pass
        self._t.join(timeout=2.0)


# ══════════════════════════════════════════
# ASYNC DETECTOR (YOLO + BYTETracker)  [RT-1]
# ══════════════════════════════════════════
class AsyncDetector:
    """
    [RT-1] Chạy YOLO + BYTETracker trên thread riêng để main loop KHÔNG bị
    block bởi YOLO (~55ms/CPU). Nhờ đó main loop chỉ còn việc nhẹ (vẽ +
    hiển thị ~10ms) và có thể chạy mượt ở tốc độ realtime của video.

    Worker luôn xử lý frame MỚI NHẤT (queue maxlen=1, drop frame cũ).
    Mỗi lần xong tăng version → main loop biết khi nào có detection mới để
    chạy lại logic vi phạm (giữ nhịp xác nhận đúng, không tính theo display fps).
    """

    def __init__(self, model, tracker, device):
        self._model   = model
        self._tracker = tracker
        self._device  = device
        self._lock    = threading.Lock()
        self._latest: List = []
        self._version = 0
        self._infer_count = 0
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop    = False
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def submit(self, frame: np.ndarray):
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_latest(self) -> Tuple[List, int]:
        with self._lock:
            return self._latest, self._version

    @property
    def infer_count(self) -> int:
        return self._infer_count

    def _worker(self):
        while not self._stop:
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break
            tracked_raw = self._infer(frame)
            with self._lock:
                self._latest  = tracked_raw
                self._version += 1
                self._infer_count += 1

    def _infer(self, frame: np.ndarray) -> List:
        detections    = []
        det_class_ids = []
        try:
            with torch.no_grad():
                results = self._model(
                    frame,
                    imgsz=IMG_SIZE,
                    verbose=False,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    agnostic_nms=True,
                    max_det=MAX_DET,
                    classes=list(VEHICLE_CLASSES.keys()),
                )[0]
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)
                
                # Giữ lại nếu conf >= 0.15, HOẶC xe lớn/gần (area > 5000 hoặc y2 sát đáy) thì chấp nhận mọi conf (>= YOLO_CONF)
                if conf < 0.15 and area < 5000 and y2 < IMG_SIZE * 0.5:
                    continue
                    
                detections.append(torch.tensor([x1, y1, x2, y2, conf]))
                det_class_ids.append(int(box.cls[0]))
        except Exception as e:
            print(f"[WARN] YOLO: {e}")
            return []

        tracked_raw = []
        if detections:
            try:
                det_np = np.ascontiguousarray(torch.stack(detections).cpu().numpy())
                tracked = self._tracker.update(det_np, frame.shape[:2], frame.shape[:2])
                det_boxes = det_np[:, :4]
                for t in tracked:
                    x1, y1, x2, y2 = t.tlbr
                    tid = t.track_id
                    tb  = np.array([x1, y1, x2, y2])
                    if len(det_boxes) > 0:
                        ious = _box_iou_batch(det_boxes, tb)
                        best = int(np.argmax(ious))
                        cid  = det_class_ids[best] if ious[best] > 0.3 else _track_class.get(tid, 1)
                        conf = float(det_np[best, 4]) if ious[best] > 0.3 else _track_conf.get(tid, 0.0)
                    else:
                        cid  = _track_class.get(tid, 1)
                        conf = _track_conf.get(tid, 0.0)
                    _track_class[tid] = cid
                    _track_conf[tid]  = conf
                    tracked_raw.append((float(x1), float(y1), float(x2), float(y2), tid, cid, conf))
            except Exception as e:
                print(f"[WARN] Tracker: {e}")
        return tracked_raw

    def stop(self):
        self._stop = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(None)
            except Exception:
                pass
        self._t.join(timeout=2.0)


# ══════════════════════════════════════════
# STATE GLOBAL
# ══════════════════════════════════════════
# [DIST-1] Buffer lớn hơn (15) + EMA state
_dist_buffers: Dict[int, deque]  = defaultdict(lambda: deque(maxlen=15))
_dist_ema:     Dict[int, float]  = {}
_dist_prev:    Dict[int, float]  = {}   # giá trị output frame trước (clamp jump)
_track_class:  Dict[int, int]    = {}
_track_conf:   Dict[int, float]  = {}

def reset_global_runtime_state(lane_detector=None) -> None:
    _dist_buffers.clear()
    _dist_ema.clear()
    _dist_prev.clear()
    _track_class.clear()
    _track_conf.clear()
    if lane_detector is not None:
        try:
            lane_detector.reset_smoother()
        except Exception:
            pass


# ══════════════════════════════════════════
# KHOẢNG CÁCH — v4 với stabilization đầy đủ
# ══════════════════════════════════════════
def calc_distance(
    track_id: int,
    y1: float, y2: float,
    horizon_y: float,
    frame_h: int,
    class_id: int = 2,
) -> Tuple[float, bool]:
    """
    Trả về (distance_m, dist_ok).

    Pipeline:
    1. Tính dist_A (bbox_h method) và dist_B (horizon method)
    2. Hybrid blend theo reference distance
    3. Median filter (buffer=15) + EMA (α=0.25) → blend 60/40
    4. Clamp jump tối đa MAX_DIST_JUMP_M so với frame trước
    5. Nếu không đo được → trả giá trị buffer/ema cũ, dist_ok=False
       (không hiện FAR nếu có history)
    """
    _track_class[track_id] = class_id

    # [DIST-2] focal theo imgsz thực tế + per-class override
    focal_f = FOCAL_OVERRIDE.get(class_id, FOCAL_FACTOR)
    focal   = IMG_SIZE * focal_f   # độc lập frame_h, đúng hơn khi YOLO resize

    real_h  = VEHICLE_REAL_HEIGHT.get(class_id, 1.5)
    # [DIST-2] Shrink bbox_h để bù YOLO padding
    bbox_h  = float(y2 - y1) * BBOX_H_SHRINK_FACTOR
    y_rel   = float(y2 - horizon_y)

    dist_A_ok = bbox_h >= MIN_BBOX_H_PX
    dist_B_ok = y_rel  >= MIN_DIST_Y_REL
    dist_A: Optional[float] = None
    dist_B: Optional[float] = None

    if dist_A_ok:
        raw_A = (real_h * focal) / bbox_h
        if np.isfinite(raw_A) and 1.5 <= raw_A <= MAX_RELIABLE_DIST:
            dist_A = float(raw_A)
        else:
            dist_A_ok = False

    if dist_B_ok:
        raw_B = (CAMERA_HEIGHT * focal) / y_rel
        if np.isfinite(raw_B) and 1.5 <= raw_B <= MAX_RELIABLE_DIST:
            dist_B = float(raw_B)
        else:
            dist_B_ok = False

    # Trả fallback nếu không đo được
    if not dist_A_ok and not dist_B_ok:
        fallback = _dist_prev.get(track_id)
        if fallback is None:
            fallback = float(np.median(_dist_buffers[track_id])) \
                       if _dist_buffers[track_id] else MAX_RELIABLE_DIST
        # dist_ok=False nhưng trả giá trị cũ thay vì MAX để tránh FAR spam
        return fallback, False

    # Hybrid blend
    if dist_A_ok and dist_B_ok:
        ref = dist_A
        if ref <= HYBRID_NEAR_THRESH:
            alpha = 1.0
        elif ref >= HYBRID_FAR_THRESH:
            alpha = 0.0
        else:
            alpha = 1.0 - (ref - HYBRID_NEAR_THRESH) / (HYBRID_FAR_THRESH - HYBRID_NEAR_THRESH)
        raw_dist = alpha * dist_A + (1.0 - alpha) * dist_B
    elif dist_A_ok:
        raw_dist = dist_A
    else:
        raw_dist = dist_B

    raw_dist = float(np.clip(raw_dist, 1.5, MAX_RELIABLE_DIST))

    # [DIST-1] Clamp jump trước khi đưa vào buffer
    prev = _dist_prev.get(track_id)
    if prev is not None:
        raw_dist = float(np.clip(raw_dist, prev - MAX_DIST_JUMP_M, prev + MAX_DIST_JUMP_M))

    # [DIST-1] Median filter
    _dist_buffers[track_id].append(raw_dist)
    med = float(np.median(_dist_buffers[track_id]))

    # [DIST-1] EMA
    prev_ema = _dist_ema.get(track_id, med)
    ema = DIST_EMA_ALPHA * raw_dist + (1.0 - DIST_EMA_ALPHA) * prev_ema
    _dist_ema[track_id] = ema

    # [DIST-1] Blend median + EMA
    result = DIST_MEDIAN_W * med + DIST_EMA_W * ema
    result = float(np.clip(result, 1.5, MAX_RELIABLE_DIST))
    _dist_prev[track_id] = result

    return result, True


# ══════════════════════════════════════════
# LANE HELPERS (chuẩn hóa, giữ nguyên logic v3)
# ══════════════════════════════════════════
def _normalize_lane(lane):
    if lane is None:
        return None
    if isinstance(lane, (tuple, list)) and len(lane) == 2:
        p0, p1 = lane[0], lane[1]
        if isinstance(p0, (tuple, list)) and len(p0) == 2:
            return (p0, p1)
    if isinstance(lane, (list, np.ndarray)) and len(lane) >= 2:
        pts = sorted(lane, key=lambda p: p[1], reverse=True)
        return (pts[0], pts[-1])
    return None

def _interp_x(nl, y: float) -> float:
    (xb, yb), (xt, yt) = nl
    if abs(yb - yt) < 1e-3:
        return float(xb)
    return float(xt + np.clip((y - yt) / (yb - yt), 0.0, 1.0) * (xb - xt))

def is_in_ego_lane(x1, y1, x2, y2, ll, rl) -> bool:
    ln, rn = _normalize_lane(ll), _normalize_lane(rl)
    if ln is None or rn is None:
        return False
    mg = (x2 - x1) * 0.2
    ex1, ex2 = x1 + mg, x2 - mg
    lx, rx = _interp_x(ln, y2), _interp_x(rn, y2)
    bw = max(ex2 - ex1, 1.0)
    ov = max(0.0, min(ex2, rx) - max(ex1, lx))
    return (ov / bw) >= 0.60

def is_center_in_ego_lane(x1, y1, x2, y2, ll, rl, mr=EGO_LANE_CENTER_MARGIN) -> bool:
    ln, rn = _normalize_lane(ll), _normalize_lane(rl)
    if ln is None or rn is None:
        return False
    cx = (x1 + x2) / 2.0
    lx, rx = _interp_x(ln, y2), _interp_x(rn, y2)
    if rx < lx: lx, rx = rx, lx
    w = max(rx - lx, 1.0)
    return (lx + w * mr) <= cx <= (rx - w * mr)

def is_front_center(mid_x: float, fw: int, tol: float = EGO_FRONT_ZONE_RATIO) -> bool:
    return abs(mid_x - fw / 2.0) / max(fw, 1) <= tol

def is_in_any_lane(mx: float, ll, rl, y2: float) -> bool:
    ln, rn = _normalize_lane(ll), _normalize_lane(rl)
    if ln is None or rn is None:
        return True
    return _interp_x(ln, y2) <= mx <= _interp_x(rn, y2)

def _lane_bottom_x(lane):
    try:
        if isinstance(lane, (tuple, list)) and len(lane) == 2:
            p0 = lane[0]
            if isinstance(p0, (tuple, list)):
                return float(p0[0])
        if isinstance(lane, (list, np.ndarray)):
            return float(sorted(lane, key=lambda p: p[1], reverse=True)[0][0])
    except Exception:
        pass
    return None

def is_solid_line(frame: np.ndarray, lane_pts) -> bool:
    """Phân tích ảnh để xác định vạch kẻ đường là nét liền hay nét đứt."""
    lane = _normalize_lane(lane_pts)
    if lane is None or frame is None:
        return True # Mặc định là vạch liền nếu không xác định được
    (x1, y1), (x2, y2) = lane
    h, w = frame.shape[:2]
    
    # Tạo mask dạng đường thẳng trùng với lane
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=8)
    
    # Chuyển sang ảnh xám và tìm các pixel sáng (vạch kẻ đường)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    line_pixels = cv2.bitwise_and(bright, mask)
    
    bright_count = cv2.countNonZero(line_pixels)
    mask_area = cv2.countNonZero(mask)
    
    # Mật độ pixel sáng: Vạch liền > 0.35, vạch đứt < 0.35
    density = bright_count / max(1.0, mask_area)
    return density > 0.35



# ══════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════
def draw_lanes(frame: np.ndarray, ll, rl, lanes_all=[]) -> np.ndarray:
    overlay = frame.copy()

    def _pts(lane):
        if lane is None: return None
        if isinstance(lane, (tuple, list)) and len(lane) == 2:
            p0, p1 = lane[0], lane[1]
            if isinstance(p0, (tuple, list)):
                return [tuple(map(int, p0)), tuple(map(int, p1))]
        if isinstance(lane, (list, np.ndarray)):
            return [tuple(map(int, p)) for p in lane if p[0] > 0]
        return None

    # Draw all detected lanes first
    for lane_pts in lanes_all:
        if len(lane_pts) > 1:
            pts_array = np.array(lane_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_array], isClosed=False, color=(255, 100, 255), thickness=4)

    lp, rp = _pts(ll), _pts(rl)
    for pts in [lp, rp]:
        if pts and len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], (255, 150, 0), 3) # Blue color
    if lp and rp and len(lp) >= 2 and len(rp) >= 2:
        poly = np.array([lp[0], lp[-1]] + list(reversed(rp)), dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (0, 200, 100)) # Thay đổi màu xanh lục nhạt
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame) # Tăng độ đậm lên 0.45 để dễ nhìn
    return frame

def draw_hud(frame, fps, n_veh, n_viol, stage_info=""):
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (560, 40), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    txt = f"FPS:{fps:.1f}  Veh:{n_veh}  Viol:{n_viol}"
    if stage_info:
        txt += f"  [{stage_info}]"
    cv2.putText(frame, txt, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2, cv2.LINE_AA)

def draw_flash(frame, vtype_str, frames_left):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
    if frames_left > 0:
        ov = frame.copy()
        txt_y = h // 2
        sz, _ = cv2.getTextSize(vtype_str, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
        tx = (w - sz[0]) // 2
        cv2.rectangle(ov, (tx - 10, txt_y - 40), (tx + sz[0] + 10, txt_y + 15), (0,0,0), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, vtype_str, (tx, txt_y),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 80, 255), 3, cv2.LINE_AA)


# ══════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════
def _box_iou_batch(boxes: np.ndarray, q: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(boxes[:, 0], q[0]); iy1 = np.maximum(boxes[:, 1], q[1])
    ix2 = np.minimum(boxes[:, 2], q[2]); iy2 = np.minimum(boxes[:, 3], q[3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    aa = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ab = (q[2] - q[0]) * (q[3] - q[1])
    union = aa + ab - inter
    return np.where(union > 0, inter / union, 0.0)

def _cleanup_track(tid, confirmer, *dicts):
    confirmer.reset(tid)
    for d in (_dist_buffers, _dist_ema, _dist_prev, _track_class, _track_conf):
        d.pop(tid, None)
    for d in dicts:
        d.pop(tid, None)


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
    os.makedirs(
        os.path.dirname(OUTPUT_VIDEO) if os.path.dirname(OUTPUT_VIDEO) else "output",
        exist_ok=True,
    )
    reset_global_runtime_state()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # ── YOLO load + warmup ──────────────────────────────────────
    print(f"[*] Load YOLO: {YOLO_MODEL}")
    try:
        model = YOLO(YOLO_MODEL).to(device)
        if device.type == "cuda":
            model.model.half()
            print("[*] YOLO: FP16 enabled")
    except Exception as e:
        print(f"[FATAL] {e}"); sys.exit(1)

    # [FPS-2] Warmup: 3 lần inference dummy để CUDA JIT compile xong
    print("[*] YOLO warmup...")
    _dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE,
                         dtype=torch.float16 if device.type == "cuda" else torch.float32
                         ).to(device)
    with torch.no_grad():
        for _ in range(3):
            model.model(_dummy)
    del _dummy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[*] Warmup done.")

    # ── Lane detector ───────────────────────────────────────────
    print("[*] Load LaneDetector...")
    try:
        lane_detector = LaneDetector(device=str(device))
    except Exception as e:
        print(f"[FATAL] {e}"); sys.exit(1)

    tracker   = BYTETracker(TrackerArgs())
    confirmer = ViolationConfirmer()

    # ── Video I/O ───────────────────────────────────────────────
    print(f"[*] Mo video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[FATAL] Khong the mo: {VIDEO_PATH}"); sys.exit(1)

    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    LANE_ROI_Y = int(vid_h * 0.35)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = AsyncVideoWriter(OUTPUT_VIDEO, fourcc, vid_fps, (vid_w, vid_h))

    # [FPS-8] Lane detection chạy trên thread riêng, không block main loop
    lane_proc = AsyncLaneProcessor(lane_detector, vid_h)
    # [RT-1] YOLO + tracker cũng chạy trên thread riêng → main loop nhẹ, realtime
    detector  = AsyncDetector(model, tracker, device)

    # [FPS-4] Tạo cửa sổ hiển thị 1 lần (tránh namedWindow/resizeWindow mỗi frame)
    cv2.namedWindow("ADAS v4", cv2.WINDOW_NORMAL)
    if vid_h > vid_w:
        cv2.resizeWindow("ADAS v4", int(800 * vid_w / vid_h), 800)
    else:
        cv2.resizeWindow("ADAS v4", 1280, 720)

    print(f"[*] Output: {OUTPUT_VIDEO}  ({vid_w}x{vid_h} @ {vid_fps:.0f}fps)")
    print(f"[*] Config: REALTIME={REALTIME_PLAYBACK}  DETECT=async  LANE=async  "
          f"IMG_SIZE={IMG_SIZE}  MIN_BBOX_H={MIN_BBOX_H_PX}px")
    print("[*] Bat dau xu ly... Nhan 'q' de thoat.\n")

    # ── State ───────────────────────────────────────────────────
    frame_count      = 0
    fps_buf          = deque(maxlen=60)
    t_start          = time.perf_counter()

    prev_gray              = None
    cached_ll              = None
    cached_rl              = None
    cached_horizon_y       = int(vid_h * 0.45)
    cached_lanes_all       = []
    prev_track_ids         = set()
    last_det_version       = -1
    cached_tracked_raw     = []
    cached_draw_items      = []     # [(x1,y1,x2,y2,color,label), ...] để vẽ mỗi frame
    n_vehicles             = 0
    lost_track_frames      = {}     # Lưu số frame track bị mất liên lạc tạm thời

    cx_hist:   Dict[int, deque] = defaultdict(lambda: deque(maxlen=CUT_IN_FRAMES + 2))
    d_hist:    Dict[int, deque] = defaultdict(lambda: deque(maxlen=CUT_IN_FRAMES + 2))
    ls_hist:   Dict[int, deque] = defaultdict(lambda: deque(maxlen=CUT_IN_FRAMES + 2))
    cie_cache: Dict[int, bool]  = {}
    dv_cache:  Dict[int, float] = {}
    dok_cache: Dict[int, bool]  = {}

    prev_stats:  Dict[str, int]  = defaultdict(int)
    flash_queue: Dict[str, Tuple[int, str]] = {}
    stats_now: Dict[str, int]    = defaultdict(int)

    # Profiling realtime
    prof_t0          = time.perf_counter()
    prof_frames      = 0
    prof_det0        = 0

    # ── Main loop (REALTIME paced) ──────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ── Scene change (để reset lane khi cắt cảnh) ──────────
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
            sc   = max(float(np.mean(diff)) / 255.0, float(np.std(diff)) / 255.0)
        else:
            sc = 1.0
        prev_gray = curr_gray
        if sc >= SCENE_CHANGE_RESET_THRESH:
            lane_proc.reset()

        # ── Gửi frame cho các worker (async, drop frame cũ) ────
        if frame_count % DETECT_SUBMIT_EVERY == 0:
            detector.submit(frame)
        if frame_count % LANE_SUBMIT_EVERY == 0:
            lane_proc.submit(frame, frame_count)

        # ── Đọc kết quả mới nhất từ worker ─────────────────────
        cached_ll, cached_rl, cached_horizon_y, cached_lanes_all = lane_proc.get_latest()
        ll, rl, horizon_y, lanes_all = cached_ll, cached_rl, cached_horizon_y, cached_lanes_all

        tracked_raw, det_version = detector.get_latest()
        new_detection = det_version != last_det_version

        # ══ LOGIC: chỉ chạy khi có detection MỚI (giữ nhịp xác nhận) ══
        if new_detection:
            last_det_version   = det_version
            cached_tracked_raw = tracked_raw
            n_vehicles         = len(tracked_raw)

            # Cleanup track đã mất (Grace period = 12 frames để không reset đếm nhầm do YOLO/Tracker drop tạm thời)
            current_tids = {o[4] for o in tracked_raw}
            for tid in prev_track_ids - current_tids:
                if tid not in lost_track_frames:
                    lost_track_frames[tid] = 0
                lost_track_frames[tid] += 1
                if lost_track_frames[tid] > 12:
                    _cleanup_track(tid, confirmer, cx_hist, d_hist, ls_hist,
                                   cie_cache, dv_cache, dok_cache)
                    del lost_track_frames[tid]
            
            for tid in current_tids:
                if tid in lost_track_frames:
                    del lost_track_frames[tid]
                    
            prev_track_ids = prev_track_ids.union(current_tids)
            prev_track_ids = {t for t in prev_track_ids if t not in lost_track_frames or lost_track_frames[t] <= 12}

            # Vạch liền hay đứt (cvtColor) — tính theo nhịp detection
            left_solid  = is_solid_line(frame, ll)
            right_solid = is_solid_line(frame, rl)

            # ── EGO DEPARTURE ──────────────────────────────────
            ego_dep = False
            if ll is not None and rl is not None:
                cx_ = vid_w // 2
                lbx = _lane_bottom_x(ll)
                rbx = _lane_bottom_x(rl)
                if lbx is not None and cx_ < lbx + 15 and left_solid:  ego_dep = True
                if rbx is not None and cx_ > rbx - 15 and right_solid: ego_dep = True
            if ego_dep:
                confirmer.update(-1, ViolationType.EGO_LANE_DEPARTURE, frame_count, frame,
                                 {"distance_m": 0.0})
                flash_queue.setdefault("ego", (TEXT_FLASH_FRAMES, "[VIOLATION] EGO LANE DEPARTURE"))
            else:
                confirmer.reset(-1)

            # ── VEHICLES ───────────────────────────────────────
            instant_warn = set()
            for x1, y1, x2, y2, tid, cid, conf in tracked_raw:
                mx = (x1 + x2) / 2.0
                dist, dok      = calc_distance(tid, y1, y2, horizon_y, vid_h, cid)
                dv_cache[tid]  = dist
                dok_cache[tid] = dok

                in_ego = is_in_ego_lane(x1, y1, x2, y2, ll, rl)
                cie    = is_center_in_ego_lane(x1, y1, x2, y2, ll, rl)
                cie_cache[tid] = cie
                fc     = is_front_center(mx, vid_w)
                area   = max(0.0, (x2 - x1) * (y2 - y1))

                cx_hist[tid].append((frame_count, mx))
                if dok:
                    d_hist[tid].append((frame_count, dist))
                ls_hist[tid].append(cie)

                # e1. UNSAFE_DISTANCE
                hd = list(d_hist[tid])
                ud = 0.0
                if dok and len(hd) >= UNSAFE_LOOKBACK_FRAMES:
                    ud = hd[-UNSAFE_LOOKBACK_FRAMES][1] - dist
                
                # Nếu xe cực gần (<15m) và lấn làn (in_ego), 
                # HOẶC xe siêu gần (<12m), ngay trước mặt (fc) và cực lớn (area > 15000) -> cảnh báo đâm đụng
                is_danger = cie or (in_ego and dist < 15.0) or (fc and dist < 12.0 and area > 15000)
                
                if dok and is_danger and fc and area >= MIN_BBOX_AREA and dist < SAFE_DISTANCE_M and ud > -1.0:
                    new_conf = confirmer.update(tid, ViolationType.UNSAFE_DISTANCE, frame_count, frame,
                                     {"distance_m": dist, "bbox": (x1, y1, x2, y2), "dist_delta": ud})
                    if new_conf:
                        flash_queue[f"{tid}_{ViolationType.UNSAFE_DISTANCE.value}"] = (TEXT_FLASH_FRAMES, f"[VIOLATION] UNSAFE DISTANCE ID:{tid}")

                # e1.5 SIDE INTRUSION
                if (dok and ENABLE_SIDE_INTRUSION and in_ego and not cie
                        and dist < SIDE_INTRUSION_DIST_THRESH
                        and area >= SIDE_INTRUSION_MIN_AREA
                        and y2 > vid_h * SIDE_INTRUSION_MIN_Y2_RATIO):
                    instant_warn.add(tid)

                # e2. CUT-IN
                hx  = list(cx_hist[tid])
                hd2 = list(d_hist[tid])
                if dok and len(hx) >= CUT_IN_FRAMES and len(hd2) >= CUT_IN_FRAMES:
                    ofx, ox = hx[-CUT_IN_FRAMES]
                    _,   od = hd2[-CUT_IN_FRAMES]
                    dx = abs(mx - ox)
                    fe = max(frame_count - ofx, 1)
                    ls_ = ls_hist[tid]
                    ent = False
                    if len(ls_) >= CUT_IN_FRAMES:
                        pp = list(ls_)[:-CUT_IN_ENTER_CONFIRM]
                        cp = list(ls_)[-CUT_IN_ENTER_CONFIRM:]
                        ent = sum(1 for v in pp if not v) >= CUT_IN_PREV_OUT_MIN and all(cp)
                    if (ent and cie and fc and area >= MIN_BBOX_AREA
                            and dist < CUT_IN_DISTANCE_THRESH
                            and od - dist > CUT_IN_DISTANCE_DELTA
                            and dx > CUT_IN_DELTA_X
                            and dx / fe > CUT_IN_LATERAL_SPEED
                            and (abs(ox - vid_w / 2) - abs(mx - vid_w / 2)) > CUT_IN_CENTER_APPROACH):
                        new_conf = confirmer.update(tid, ViolationType.DANGEROUS_CUT_IN, frame_count, frame,
                                         {"distance_m": dist, "bbox": (x1, y1, x2, y2)})
                        if new_conf:
                            flash_queue[f"{tid}_{ViolationType.DANGEROUS_CUT_IN.value}"] = (TEXT_FLASH_FRAMES, f"[VIOLATION] DANGEROUS CUT-IN ID:{tid}")

                # e3. OUTSIDE_LANE
                if ENABLE_OUTSIDE_LANE and not is_in_any_lane(mx, ll, rl, y2):
                    ln, rn = _normalize_lane(ll), _normalize_lane(rl)
                    lx = _interp_x(ln, y2) if ln else 0
                    rx = _interp_x(rn, y2) if rn else vid_w
                    crossed_solid = False
                    if mx < lx and left_solid:  crossed_solid = True
                    if mx > rx and right_solid: crossed_solid = True
                    if crossed_solid:
                        new_conf = confirmer.update(tid, ViolationType.OUTSIDE_LANE, frame_count, frame,
                                         {"distance_m": dist, "bbox": (x1, y1, x2, y2)})
                        if new_conf:
                            flash_queue[f"{tid}_{ViolationType.OUTSIDE_LANE.value}"] = (TEXT_FLASH_FRAMES, f"[VIOLATION] OUTSIDE LANE ID:{tid}")

            # ── KHÔNG CẦN DUYỆT active NỮA (vì đã xử lý flash trong lúc update) ──
            active = confirmer.get_active()

            stats_now = confirmer.get_stats()
            for vtype in ViolationType:
                cn, cp = int(stats_now.get(vtype.value, 0)), int(prev_stats.get(vtype.value, 0))
                # Không add text tĩnh chung chung nếu đã có flash ID chi tiết ở trên
                prev_stats[vtype.value] = cn

            # ── Build draw items (màu + nhãn) để vẽ lại mỗi frame ──
            cached_draw_items = []
            for x1, y1, x2, y2, tid, cid, conf in tracked_raw:
                dist = dv_cache.get(tid, MAX_RELIABLE_DIST)
                dok  = dok_cache.get(tid, False)
                if dok:
                    dlabel = f"{dist:.1f}m"
                else:
                    hist = _dist_buffers.get(tid)
                    dlabel = f"~{dist:.0f}m?" if (hist and len(hist) > 0) else "FAR"
                mx   = (x1 + x2) / 2.0
                cie  = cie_cache.get(tid, False)
                fc   = is_front_center(mx, vid_w)
                area = max(0.0, (x2 - x1) * (y2 - y1))
                is_w = any(k[0] == tid for k in active.keys())
                if dok and cie and fc and area >= MIN_BBOX_AREA and dist < SAFE_DISTANCE_M:
                    color = COLOR_DANGER
                elif is_w or tid in instant_warn:
                    color = COLOR_WARNING
                else:
                    color = COLOR_SAFE
                cls_name = VEHICLE_CLASSES.get(cid, "Veh")
                label = f"ID:{tid}[{cls_name} {conf:.2f}]{dlabel}"
                cached_draw_items.append((int(x1), int(y1), int(x2), int(y2), color, label))

        # ══ DRAW (mỗi frame — nhẹ) ═════════════════════════════
        frame = draw_lanes(frame, ll, rl, lanes_all)
        for x1, y1, x2, y2, color, label in cached_draw_items:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        # Flash
        for fk in list(flash_queue):
            fl, vstr = flash_queue[fk]
            if fl > 0:
                if fl > TEXT_FLASH_FRAMES - FLASH_FRAMES:
                    draw_flash(frame, vstr, fl)
                else:
                    cv2.rectangle(frame, (0, 0), (vid_w - 1, vid_h - 1), (0, 0, 255), 10)
                flash_queue[fk] = (fl - 1, vstr)
            else:
                del flash_queue[fk]

        # HUD (FPS hiển thị)
        cur_fps = float(np.mean(fps_buf)) if fps_buf else vid_fps
        draw_hud(frame, cur_fps, n_vehicles, stats_now.get("TOTAL", 0))

        writer.write(frame)
        cv2.imshow("ADAS v4", frame)

        # ── REALTIME PACING ────────────────────────────────────
        # Ngủ đến đúng mốc thời gian của frame theo vid_fps → video chạy
        # đúng tốc độ thật, mượt. Nếu trễ thì bỏ qua sleep để tự bắt kịp.
        if REALTIME_PLAYBACK:
            target_t = t_start + frame_count / vid_fps
            now      = time.perf_counter()
            sleep_s  = target_t - now
            if sleep_s > 0:
                wait_ms = max(1, int(sleep_s * 1000))
            else:
                wait_ms = 1
        else:
            wait_ms = 1

        # FPS thực tế của vòng hiển thị
        prof_frames += 1
        elapsed = time.perf_counter() - prof_t0
        if elapsed >= 1.0:
            disp_fps = prof_frames / elapsed
            det_fps  = (detector.infer_count - prof_det0) / elapsed
            fps_buf.append(disp_fps)
            print(f"[PROFILE] display_fps≈{disp_fps:.1f}  detect_fps≈{det_fps:.1f}  "
                  f"veh={n_vehicles}")
            prof_t0     = time.perf_counter()
            prof_frames = 0
            prof_det0   = detector.infer_count

        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("[*] Thoat."); break

    cap.release()
    writer.release()
    lane_proc.stop()
    detector.stop()
    cv2.destroyAllWindows()
    reset_global_runtime_state(lane_detector)

    total = time.perf_counter() - t_start
    print("\n" + "═"*60)
    print("  TONG KET  (v4)")
    print("═"*60)
    print(f"  Frames       : {frame_count}")
    print(f"  Thoi gian    : {total:.1f}s")
    print(f"  FPS TB       : {frame_count/max(total,1e-6):.1f}")
    print(f"  Output       : {OUTPUT_VIDEO}")
    print()
    stats = confirmer.get_stats()
    for vt in ViolationType:
        print(f"    {vt.value:<25}: {stats.get(vt.value, 0)}")
    print(f"    {'TOTAL':<25}: {stats.get('TOTAL', 0)}")
    print("═"*60)


def process_video(video_path: str, output_video: Optional[str] = None) -> None:
    global VIDEO_PATH, OUTPUT_VIDEO
    VIDEO_PATH = video_path
    if output_video is not None:
        OUTPUT_VIDEO = output_video
    main()


if __name__ == "__main__":
    clips = ["demo_input3.mp4", "demo_input2.mp4", "demo_input1.mp4"] 
    for i, clip in enumerate(clips, 1):
        process_video(clip, f"output/output_clip_{i}.avi")  