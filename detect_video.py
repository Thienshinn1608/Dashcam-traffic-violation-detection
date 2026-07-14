"""
detect_video.py
───────────────
Chạy nhận diện làn đường từ model UFLD đã train trên CurveLanes,
áp dụng trực tiếp lên video (.mp4) hoặc webcam (--source 0).

Sử dụng:
    python detect_video.py --source video.mp4 --model models/curvelanes_finetuned_best.pth
    python detect_video.py --source 0          (webcam)
    python detect_video.py --source video.mp4 --no-save  (chỉ hiển thị, không lưu)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Thêm thư mục src vào sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from lane_detector import (
    UFLDNet,
    UFLD_IMG_H,
    UFLD_IMG_W,
    UFLD_GRIDING_NUM,
    UFLD_NUM_LANES,
    UFLD_ROW_ANCHOR,
)

# Màu sắc cho từng làn đường (BGR)
LANE_COLORS = [
    (255, 100,  50),   # Lane 0: Left-Outer  — Xanh dương nhạt
    ( 50, 255, 100),   # Lane 1: Left-Inner  — Xanh lá
    ( 50, 100, 255),   # Lane 2: Right-Inner — Đỏ cam
    (255, 255,  50),   # Lane 3: Right-Outer — Vàng
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(model_path: str, device: str) -> UFLDNet:
    print(f"[INFO] Dang load model tu: {model_path}")
    model = UFLDNet().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model", ckpt)
    # Loại bỏ prefix "module." nếu có (DataParallel)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print("[INFO] Load model thanh cong!")
    return model


@torch.no_grad()
def predict_lanes(model, frame_bgr: np.ndarray, device: str, threshold: float = 0.40):
    """
    Trả về danh sách [(lane_id, [(x_orig, y_orig), ...]), ...]
    cho những làn đường được phát hiện.
    """
    orig_h, orig_w = frame_bgr.shape[:2]

    # Tiền xử lý
    img = cv2.resize(frame_bgr, (UFLD_IMG_W, UFLD_IMG_H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Suy diễn
    outputs = model(tensor)                           # [1, 101, 56, 4]
    probs   = torch.softmax(outputs, dim=1).cpu()

    bg_conf   = probs[0, UFLD_GRIDING_NUM, :, :]     # [56, 4]
    lane_conf = 1.0 - bg_conf                         # [56, 4]
    pred_grid = probs[0, :UFLD_GRIDING_NUM, :, :].argmax(dim=0)  # [56, 4]

    lanes = []
    for lane_id in range(UFLD_NUM_LANES):
        pts = []
        for r_idx, y_res in enumerate(UFLD_ROW_ANCHOR):
            conf = lane_conf[r_idx, lane_id].item()
            grid = pred_grid[r_idx, lane_id].item()
            if conf > threshold:
                x_orig = int((grid + 0.5) / UFLD_GRIDING_NUM * orig_w)
                y_orig = int(y_res / UFLD_IMG_H * orig_h)
                pts.append((x_orig, y_orig))
        if len(pts) >= 2:
            lanes.append((lane_id, pts))
    return lanes


def draw_lanes(frame: np.ndarray, lanes: list, thickness: int = 4, dot_radius: int = 5) -> np.ndarray:
    """Vẽ làn đường lên frame với màu sắc khác nhau cho từng làn."""
    overlay = frame.copy()
    for lane_id, pts in lanes:
        color = LANE_COLORS[lane_id % len(LANE_COLORS)]
        # Vẽ đường nối các điểm
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)
        # Vẽ điểm tròn tại mỗi anchor
        for pt in pts:
            cv2.circle(overlay, pt, dot_radius, color, -1, cv2.LINE_AA)
    # Pha trộn overlay để hiệu ứng trong suốt nhẹ
    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


def draw_hud(frame: np.ndarray, fps: float, num_lanes: int) -> np.ndarray:
    """Vẽ thông tin HUD góc trên trái."""
    h, w = frame.shape[:2]
    bar_h = 50
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Lanes: {num_lanes}/4", (160, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 50), 2, cv2.LINE_AA)
    cv2.putText(frame, "UFLD - CurveLanes", (w - 330, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1, cv2.LINE_AA)

    # Chú thích màu làn đường
    lane_names = ["L-Outer", "L-Inner", "R-Inner", "R-Outer"]
    for i, (name, color) in enumerate(zip(lane_names, LANE_COLORS)):
        x = 10 + i * 155
        cv2.rectangle(frame, (x, bar_h + 5), (x + 20, bar_h + 20), color, -1)
        cv2.putText(frame, name, (x + 25, bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Nhan dien lan duong tren video bang mo hinh UFLD da train.")
    parser.add_argument("--source",    default="0",
                        help="Duong dan video (.mp4 v.v.) hoac '0' cho webcam")
    parser.add_argument("--model",     default="models/curvelanes_finetuned_best.pth",
                        help="File checkpoint .pth cua mo hinh")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Nguong confidence de hien thi lan duong (mac dinh: 0.40)")
    parser.add_argument("--output",    default="output_lanes.mp4",
                        help="Ten file video ket qua xuat ra (mac dinh: output_lanes.mp4)")
    parser.add_argument("--no-save",   action="store_true",
                        help="Chi hien thi, khong luu file video ket qua")
    parser.add_argument("--no-show",   action="store_true",
                        help="Chi luu file, khong mo cua so hien thi")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Su dung thiet bi: {device.upper()}")

    # Load model
    model = load_model(args.model, device)

    # Mở nguồn video
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Khong the mo nguon video: {args.source}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Nguon video: {args.source}  |  {orig_w}x{orig_h}  |  {fps_src:.1f} FPS  |  {total_frames} frames")

    # Khởi tạo VideoWriter
    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (orig_w, orig_h))
        print(f"[INFO] Ket qua se duoc luu vao: {args.output}")

    frame_idx = 0
    fps_display = 0.0
    t_prev = time.time()

    print("[INFO] Bat dau xu ly video... Nhan 'q' de thoat.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Het video.")
            break

        frame_idx += 1

        # Dự đoán làn đường
        lanes = predict_lanes(model, frame, device, threshold=args.threshold)

        # Vẽ làn đường
        frame = draw_lanes(frame, lanes)

        # Tính FPS hiển thị
        t_now = time.time()
        fps_display = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # Vẽ HUD
        frame = draw_hud(frame, fps_display, len(lanes))

        # In tiến độ ra terminal mỗi 30 frame
        if frame_idx % 30 == 0:
            pct = f"{frame_idx/total_frames*100:.1f}%" if total_frames > 0 else "N/A"
            print(f"[{frame_idx}/{total_frames if total_frames > 0 else '?'}] ({pct}) | FPS: {fps_display:.1f} | Lanes: {len(lanes)}")

        # Lưu frame
        if writer:
            writer.write(frame)

        # Hiển thị
        if not args.no_show:
            cv2.imshow("UFLD Lane Detection - CurveLanes", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Nguoi dung yeu cau dung.")
                break

    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Da luu video ket qua tai: {args.output}")
    cv2.destroyAllWindows()
    print("[INFO] Hoan thanh!")


if __name__ == "__main__":
    main()
