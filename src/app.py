"""
app.py — Streamlit Web App cho Pipeline Phát Hiện Vi Phạm Giao Thông (Dashcam)
═══════════════════════════════════════════════════════════════════════════════
Bản nâng cấp: Tích hợp xem ảnh bằng chứng trực tiếp tại Sidebar thời gian thực.
"""

import os
import sys
import time
import tempfile
import glob

import cv2
import numpy as np
import torch
import streamlit as st
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple

# ── Thêm thư mục hiện tại vào sys.path để import module nội bộ ──────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)   # thư mục cha của src/
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── Import các module nội bộ ─────────────────────────────────────────────────
try:
    from byte_tracker import BYTETracker
    from violation_classifier import ViolationConfirmer, ViolationType, CONFIRM_FRAMES
    from lane_detector import LaneDetector
    from main_pipeline import (
        calc_distance,
        reset_global_runtime_state,
        is_in_ego_lane,
        is_center_in_ego_lane,
        is_front_center,
        is_in_any_lane,
        is_solid_line,
        draw_lanes,
        draw_flash,
        draw_hud,
        _box_iou_batch,
        _cleanup_track,
        _normalize_lane,
        _lane_bottom_x,
        VEHICLE_CLASSES,
        VEHICLE_REAL_HEIGHT,
        FRAME_SKIP,
        LANE_SKIP_FACTOR,
        IMG_SIZE,
        COLOR_DANGER,
        COLOR_WARNING,
        COLOR_SAFE,
        FLASH_FRAMES,
        TEXT_FLASH_FRAMES,
        MIN_BBOX_AREA,
        EGO_FRONT_ZONE_RATIO,
        EGO_LANE_CENTER_MARGIN,
        UNSAFE_DISTANCE_DELTA,
        UNSAFE_LOOKBACK_FRAMES,
        CUT_IN_DELTA_X,
        CUT_IN_FRAMES,
        ENABLE_OUTSIDE_LANE,
        ENABLE_SIDE_INTRUSION,
        SIDE_INTRUSION_DIST_THRESH,
        SIDE_INTRUSION_MIN_AREA,
        SIDE_INTRUSION_MIN_Y2_RATIO,
        CUT_IN_DISTANCE_THRESH,
        CUT_IN_DISTANCE_DELTA,
        CUT_IN_LATERAL_SPEED,
        CUT_IN_ENTER_CONFIRM,
        CUT_IN_PREV_OUT_MIN,
        CUT_IN_CENTER_APPROACH,
        SCENE_CHANGE_THRESH,
        SCENE_CHANGE_RESET_THRESH,
        MAX_RELIABLE_DIST,
        _dist_buffers,
        _dist_ema,
        _dist_prev,
        _track_class,
        TrackerArgs,
    )
except ImportError as e:
    st.error(f"❌ Không thể import module nội bộ: {e}")
    st.stop()

try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ Chưa cài `ultralytics`.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HIỂN THỊ KHUNG HÌNH CỐ ĐỊNH — fit vừa cả width VÀ height
# Video dọc (điện thoại 9:16) sẽ được thu nhỏ để không bị dài ngoẵng.
# ══════════════════════════════════════════════════════════════════════════════
MAX_DISPLAY_W = 720   # px — chiều rộng tối đa
MAX_DISPLAY_H = 540   # px — chiều cao tối đa (giới hạn video dọc)

def fit_frame(frame_bgr: np.ndarray,
              max_w: int = MAX_DISPLAY_W,
              max_h: int = MAX_DISPLAY_H) -> np.ndarray:
    """
    Scale frame xuống trước khi convert màu để tăng tốc xử lý (BGR -> RGB chỉ làm với ảnh nhỏ).
    """
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH TRANG STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🚦 Traffic Violation Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tùy chỉnh giao diện siêu cao cấp (Premium Glassmorphism & Animations)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Metric Cards - Glassmorphism */
    [data-testid="metric-container"] {
        background: rgba(25, 25, 30, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s ease-in-out;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 201, 255, 0.2);
        border: 1px solid rgba(0, 201, 255, 0.3);
    }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    div[data-testid="stMetricValue"] > div { 
        font-weight: 800; 
        font-size: 2.4rem;
        color: #ffffff;
        text-shadow: 0px 0px 10px rgba(255,255,255,0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        color: white !important;
        border: none;
        border-radius: 30px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6);
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 12, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #aaa;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-bottom: 3px solid #00C9FF !important;
    }

    .evidence-box {
        border: 1px solid rgba(255, 75, 75, 0.5);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
        background: rgba(255, 75, 75, 0.1);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    .evidence-box:hover {
        transform: scale(1.02);
        background: rgba(255, 75, 75, 0.2);
    }
</style>
""", unsafe_allow_html=True)


def calc_safe_distance(speed_kmh: float) -> float:
    if speed_kmh <= 60: return 35.0
    elif speed_kmh <= 80: return 55.0
    elif speed_kmh <= 100: return 70.0
    else: return 100.0


# ── CACHE MODEL — Không reload khi tương tác UI ──────────────────────────────
@st.cache_resource(show_spinner="🔄 Đang tải mô hình YOLO...")
def load_yolo_model(model_path: str) -> Optional[YOLO]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = YOLO(model_path)
        model.to(device)
        if device.type == "cuda":
            try: model.model.half()
            except Exception: pass
        return model
    except Exception as e:
        st.error(f"❌ Lỗi tải YOLO model: {e}")
        return None

@st.cache_resource(show_spinner="🔄 Đang tải LaneDetector...")
def load_lane_detector() -> Optional[LaneDetector]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try: return LaneDetector(device=str(device))
    except Exception: return LaneDetector(device="cpu")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Cấu hình & Khu vực Xem nhanh bằng chứng
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Cấu Hình & Giám Sát")
    
    # Tạo menu accordion cho gọn giao diện Sidebar
    with st.expander("🤖 Cấu hình tham số AI & Luật", expanded=False):
        yolo_model_path = st.text_input("File model (.pt)", value=os.path.join(PROJECT_ROOT, "runs", "detect", "fast_adas_model", "weights", "best.pt"))
        yolo_conf = st.slider("🎯 YOLO Confidence", 0.01, 1.00, 0.02, 0.01)
        ego_speed_kmh = st.slider("Tốc độ xe (km/h)", 10, 140, 60, 5)
        safe_dist_m = calc_safe_distance(ego_speed_kmh)
        st.info(f"📏 Khoảng cách an toàn luật định: `{safe_dist_m:.0f}m`")

    with st.expander("🖥️ Cấu hình hiển thị video", expanded=False):
        MAX_DISPLAY_W = st.slider(
            "Chiều rộng tối đa (px)", 320, 1280, 720, 80,
            help="Video sẽ được scale xuống không vượt quá giá trị này."
        )
        MAX_DISPLAY_H = st.slider(
            "Chiều cao tối đa (px)", 240, 960, 540, 60,
            help="Giới hạn chiều cao — quan trọng với video dọc (điện thoại 9:16)."
        )
        ratio_label = f"{MAX_DISPLAY_W} × {MAX_DISPLAY_H} px"
        st.caption(f"📐 Khung hiển thị: **{ratio_label}**")

    with st.expander("⚡ Tối ưu hiệu năng (Mượt mà)", expanded=True):
        frame_skip = st.slider(
            "YOLO Frame Skip", 1, 10, 3, 1,
            help="Chạy YOLO mỗi N frame. Giá trị càng lớn chạy càng nhanh, mượt."
        )
        lane_skip = st.slider(
            "Lane Detector Skip", 1, 20, 5, 1,
            help="Chạy Lane UFLD mỗi N heavy frame. Giảm tải đáng kể cho GPU/CPU."
        )
        ui_skip = st.slider(
            "UI Display Skip", 1, 10, 3, 1,
            help="Gửi ảnh lên Web mỗi N frame. Tối ưu hóa cực mạnh băng thông WebSocket để giao diện không bị giật."
        )

    st.markdown("---")
    
    # ✨ CHỨC NĂNG MỚI: XEM NHANH BẰNG CHỨNG VI PHẠM TRỰC TIẾP TẠI SIDEBAR ✨
    st.subheader("📸 Hồ Sơ Bằng Chứng Mới Nhất")
    sidebar_gallery_slot = st.empty()
    sidebar_gallery_slot.caption("Chưa có bằng chứng vi phạm nào được lập.")


# ══════════════════════════════════════════════════════════════════════════════
# TABS GIAO DIỆN CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
tab_video, tab_stats = st.tabs(["📺 Xử Lý Video Real-time", "📊 Thống Kê & Lịch Sử Toàn Bộ"])

# Luồng hiển thị Tab 1
with tab_video:
    st.header("📺 Phát Hiện Vi Phạm Giao Thông Real-time")

    # Tùy chọn nguồn video để test nhanh không cần chờ upload lâu
    video_source = st.radio("Chọn nguồn Video:", ["Video có sẵn (Test nhanh)", "Tự tải video lên (.mp4)..."], horizontal=True)
    
    video_path_to_run = None
    uploaded_file = None
    
    if video_source == "Video có sẵn (Test nhanh)":
        available_videos = ["demo_input2.mp4", "demo_input3.mp4"]
        selected_video = st.selectbox("Chọn video mẫu:", available_videos)
        video_path_to_run = selected_video
    else:
        uploaded_file = st.file_uploader("📂 Tải video dashcam lên (.mp4)", type=["mp4", "avi", "mov", "mkv"])

    col_video, col_metrics = st.columns([3, 1])

    with col_video:
        video_frame_slot = st.empty()
        video_frame_slot.info("⬆️ Chọn/Upload video và nhấn Bắt Đầu Xử Lý để phân tích dữ liệu dòng xe.")

    with col_metrics:
        st.markdown("### 📊 Chỉ Số Real-time")
        metric_fps = st.empty()
        metric_vehicles = st.empty()
        metric_violations = st.empty()

        metric_fps.metric("⚡ FPS", "—")
        metric_vehicles.metric("🚗 Xe hiện tại", "—")
        metric_violations.metric("🚨 Tổng vi phạm", "—")

        st.markdown("---")
        st.markdown("### 🔔 Thông Báo Gần Đây")
        recent_violations_slot = st.empty()
        recent_violations_slot.caption("Chưa có vi phạm nào.")

    btn_col1, btn_col2 = st.columns(2)
    
    is_ready = (video_path_to_run is not None) or (uploaded_file is not None)
    with btn_col1:
        btn_start = st.button("▶️ Bắt Đầu Xử Lý", type="primary", disabled=not is_ready)
    with btn_col2:
        btn_stop = st.button("⏹️ Dừng", type="secondary")

    # Luồng chạy khi kích hoạt xử lý video
    if btn_start and (video_path_to_run is not None or uploaded_file is not None):
        is_temp = False
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_f:
                tmp_f.write(uploaded_file.read())
                video_path_to_run = tmp_f.name
                is_temp = True

        os.makedirs(os.path.join(PROJECT_ROOT, "output", "violations"), exist_ok=True)

        import main_pipeline
        main_pipeline.YOLO_CONF = yolo_conf
        
        # Tự động nhận diện config class và chiều cao
        if "best.pt" in yolo_model_path or "fast_adas" in yolo_model_path:
            VEHICLE_CLASSES = {1: "Car", 10: "Truck", 0: "Moto"}
            VEHICLE_REAL_HEIGHT = {1: 1.45, 10: 3.50, 0: 1.10}
        else:
            VEHICLE_CLASSES = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}
            VEHICLE_REAL_HEIGHT = {2: 1.45, 3: 1.10, 5: 3.20, 7: 3.50}

        main_pipeline.VEHICLE_CLASSES = VEHICLE_CLASSES
        main_pipeline.VEHICLE_REAL_HEIGHT = VEHICLE_REAL_HEIGHT

        # Khởi tạo ASYNC WORKERS thay vì synchronous models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolo_model = load_yolo_model(yolo_model_path)
        lane_detector_obj = load_lane_detector()
        if yolo_model is None or lane_detector_obj is None:
            st.error("❌ Không thể khởi tạo mô hình.")
            st.stop()

        tracker = BYTETracker(main_pipeline.TrackerArgs())
        detector = main_pipeline.AsyncDetector(yolo_model, tracker, device)

        # ==========================================
        # ĐỒNG BỘ CẤU HÌNH & CHẠY PIPELINE
        # ==========================================
        import violation_classifier
        violation_classifier.CONFIRM_FRAMES = 8

        
        cap = cv2.VideoCapture(video_path_to_run)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps <= 0: vid_fps = 30.0
        frame_delay = 1.0 / vid_fps
        LANE_ROI_Y = int(vid_h * 0.35)

        lane_processor = main_pipeline.AsyncLaneProcessor(lane_detector_obj, vid_h)
        confirmer = ViolationConfirmer()

        frame_count = 0
        fps_buf = deque(maxlen=60)
        recent_viol_log = []
        evidence_log: list = []

        cached_horizon_y = int(vid_h * 0.45)
        tracked_raw = []
        last_det_version = -1
        prev_track_ids = set()
        lost_track_frames = {}

        cx_hist, d_hist, ls_hist = defaultdict(list), defaultdict(list), defaultdict(list)
        cie_cache, dv_cache, dok_cache = {}, {}, {}
        prev_stats = defaultdict(int)
        stats_now = defaultdict(int)
        n_vehicles = 0

        st.session_state["processing"] = True
        start_time = time.perf_counter()

        while st.session_state.get("processing", True):
            t_frame = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            violation_this_frame = False
            
            # Pacing video để khớp với FPS thực tế, giúp UI không bị dội quá tải Queue
            elapsed = time.perf_counter() - start_time
            target_time = frame_count * frame_delay
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

            # Đẩy vào Async Queue (Non-blocking)
            # LUÔN submit mỗi frame để đảm bảo Tracker không bị mất dấu xe đi nhanh.
            # (Worker Queue maxsize=1 sẽ tự động drop frame an toàn nếu GPU xử lý không kịp)
            detector.submit(frame)
                
            if frame_count % lane_skip == 0:
                lane_processor.submit(frame, frame_count)

            # Lấy kết quả mới nhất mà không phải đợi (Asynchronous Fetch)
            ll, rl, horizon_y, lanes_all = lane_processor.get_latest()

            tracked_raw_new, det_version = detector.get_latest()
            new_detection = det_version != last_det_version

            if new_detection:
                tracked_raw = tracked_raw_new
                current_tids = set(t[4] for t in tracked_raw)
                for tid in prev_track_ids - current_tids:
                    if tid not in lost_track_frames:
                        lost_track_frames[tid] = 0
                    lost_track_frames[tid] += 1
                    if lost_track_frames[tid] > 12:
                        _cleanup_track(tid, confirmer, cx_hist, d_hist, ls_hist, cie_cache, dv_cache, dok_cache)
                        del lost_track_frames[tid]
                
                for tid in current_tids:
                    if tid in lost_track_frames:
                        del lost_track_frames[tid]

                prev_track_ids = prev_track_ids.union(current_tids)
                prev_track_ids = {t for t in prev_track_ids if t not in lost_track_frames or lost_track_frames[t] <= 12}
                last_det_version = det_version

                # Kiểm tra luật vi phạm
                left_solid = is_solid_line(frame, ll)
                right_solid = is_solid_line(frame, rl)

                # Ego departure
                ego_dep = False
                if ll is not None and rl is not None:
                    cx_ = vid_w // 2
                    lbx = _lane_bottom_x(ll)
                    rbx = _lane_bottom_x(rl)
                    if lbx is not None and cx_ < lbx + 15 and left_solid: ego_dep = True
                    if rbx is not None and cx_ > rbx - 15 and right_solid: ego_dep = True
                if ego_dep:
                    confirmer.update(-1, ViolationType.EGO_LANE_DEPARTURE, frame_count, frame, {"distance_m": 0.0})
                else: confirmer.reset(-1)

                # Xe vệ tinh
                n_vehicles = len(tracked_raw)
                for x1, y1, x2, y2, tid, cid, conf in tracked_raw:
                    mx = (x1 + x2) / 2.0
                    dist, dok = calc_distance(tid, y1, y2, horizon_y, vid_h, cid)
                    dv_cache[tid], dok_cache[tid] = dist, dok
                    in_ego = is_in_ego_lane(x1, y1, x2, y2, ll, rl)
                    cie = is_center_in_ego_lane(x1, y1, x2, y2, ll, rl)
                    cie_cache[tid] = cie
                    fc = is_front_center(mx, vid_w)
                    area = max(0.0, (x2 - x1) * (y2 - y1))

                    if tid not in cx_hist: cx_hist[tid] = deque(maxlen=12)
                    if tid not in d_hist: d_hist[tid] = deque(maxlen=12)
                
                    cx_hist[tid].append((frame_count, mx))
                    if dok: d_hist[tid].append((frame_count, dist))

                    # Unsafe distance
                    hd = list(d_hist[tid])
                    is_danger = cie or (in_ego and dist < 15.0) or (fc and dist < 12.0 and area > 15000)
                    
                    if dok and is_danger and len(hd) >= UNSAFE_LOOKBACK_FRAMES and fc and area >= MIN_BBOX_AREA and dist < safe_dist_m:
                        confirmer.update(tid, ViolationType.UNSAFE_DISTANCE, frame_count, frame, {"distance_m": dist, "bbox": (x1, y1, x2, y2)})

                    # Outside lane
                    if ENABLE_OUTSIDE_LANE and not is_in_any_lane(mx, ll, rl, y2):
                        ln, rn = _normalize_lane(ll), _normalize_lane(rl)
                        from main_pipeline import _interp_x
                        lx = _interp_x(ln, y2) if ln else 0
                        rx = _interp_x(rn, y2) if rn else vid_w
                        crossed_solid = (mx < lx and left_solid) or (mx > rx and right_solid)
                        if crossed_solid:
                            confirmer.update(tid, ViolationType.OUTSIDE_LANE, frame_count, frame, {"distance_m": dist, "bbox": (x1, y1, x2, y2)})

                stats_now = confirmer.get_stats()
                
                # ── Phát hiện vi phạm mới → tích lũy evidence_log → render grid ──
                for vtype in ViolationType:
                    cn = int(stats_now.get(vtype.value, 0))
                    if cn > prev_stats[vtype.value]:
                        violation_this_frame = True
                        recent_viol_log.append(f"🔴 Frame {frame_count}: **{vtype.value}**")

                        all_files = sorted(
                            glob.glob(os.path.join(PROJECT_ROOT, "output", "violations", "*.jpg")),
                            key=os.path.getmtime
                        )
                        known_paths = {e["path"] for e in evidence_log}
                        for img_path in all_files:
                            if img_path not in known_paths:
                                evidence_log.append({
                                    "path": img_path,
                                    "label": f"F{frame_count} · {vtype.value}",
                                    "frame": frame_count,
                                })

                        with sidebar_gallery_slot.container():
                            st.caption(f"🗂️ **{len(evidence_log)} bằng chứng** đã ghi nhận")
                            for ev in reversed(evidence_log[-5:]):
                                img_bgr = cv2.imread(ev["path"])
                                if img_bgr is None: continue
                                target_w = 260
                                ih, iw = img_bgr.shape[:2]
                                target_h = int(target_w * ih / iw)
                                img_resized = cv2.resize(img_bgr, (target_w, target_h))
                                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                                fname   = os.path.basename(ev["path"])
                                st.image(img_rgb, caption=f"{ev['label']} · {fname[:28]}", width=target_w)
                                st.divider()

                    prev_stats[vtype.value] = cn

            # Render frame (LUÔN CHẠY TRÊN MỖI KHUNG HÌNH)
            frame = draw_lanes(frame, ll, rl, lanes_all)
            for x1, y1, x2, y2, tid, cid, conf in tracked_raw:
                dist = dv_cache.get(tid, MAX_RELIABLE_DIST)
                color = (0, 0, 255) if any(k[0] == tid for k in confirmer.get_active().keys()) else (0, 220, 60)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID:{tid} {dist:.1f}m", (int(x1), max(int(y1) - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Vẽ bù các xe đang vi phạm từ Confirmer (giúp giữ khung đỏ nhấp nháy ngay cả khi tạm mất track)
            for (tid, vtype), info in confirmer.get_active().items():
                metadata = info.get("metadata", {})
                if "bbox" in metadata:
                    x1, y1, x2, y2 = metadata["bbox"]
                    # Vẽ khung nháy đỏ/cam
                    color = (0, 165, 255) if int(time.time() * 4) % 2 == 0 else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    # Nền chữ
                    label = f"[VIOLATION] {vtype.value} ID:{tid}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (int(x1), int(y1)-lh-10), (int(x1)+lw, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Tối ưu hóa UI: Chỉ cập nhật hình ảnh mỗi ui_skip frame để giảm độ trễ Streamlit
            if frame_count % ui_skip == 0 or violation_this_frame:
                video_frame_slot.image(
                    fit_frame(frame, max_w=MAX_DISPLAY_W, max_h=MAX_DISPLAY_H),
                    channels="RGB",
                    width="content",
                )

            dt = time.perf_counter() - t_frame
            fps_buf.append(1.0 / max(dt, 1e-6))
            cur_fps = float(np.mean(fps_buf))

            # Tối ưu hóa UI: Chỉ cập nhật các Metrics mỗi 5 frame
            if frame_count % 5 == 0:
                metric_fps.metric("⚡ FPS", f"{cur_fps:.1f}")
                metric_vehicles.metric("🚗 Xe hiện tại", str(n_vehicles))
                metric_violations.metric("🚨 Tổng vi phạm", str(stats_now.get("TOTAL", 0)))

            if recent_viol_log and frame_count % 5 == 0:
                recent_violations_slot.markdown("\n\n".join(recent_viol_log[-4:]))

        cap.release()
        try:
            detector.stop()
            lane_processor.stop()
        except:
            pass
        if is_temp and os.path.exists(video_path_to_run):
            os.unlink(video_path_to_run)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — XEM LẠI TOÀN BỘ BẰNG CHỨNG & NHẬT KÝ VI PHẠM
# Tính năng:
#   - Xem TẤT CẢ ảnh bằng chứng đã lưu, không giới hạn số lượng
#   - Filter theo loại vi phạm
#   - Phân trang (pagination) để không load nặng khi nhiều ảnh
#   - Click ảnh → xem to (lightbox đơn giản dùng st.expander)
#   - Kết nối với CSV: click dòng log → highlight ảnh tương ứng
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.header("📊 Toàn Bộ Nhật Ký & Hồ Sơ Bằng Chứng")

    # Thanh công cụ trên cùng
    tool_col1, tool_col2, tool_col3, tool_col4 = st.columns([2, 1, 1, 1])
    with tool_col1:
        st.markdown("") # spacer
    with tool_col2:
        if st.button("🔄 Làm Mới", width="stretch"):
            st.rerun()
    with tool_col3:
        csv_path = os.path.join(PROJECT_ROOT, "output", "violation_log.csv")
        if st.button("🗑️ Xóa CSV", width="stretch"):
            if os.path.exists(csv_path):
                os.remove(csv_path)
                st.success("Đã xóa CSV.")
                st.rerun()
    with tool_col4:
        import shutil
        if st.button("🗑️ Xóa Ảnh", use_container_width=True):
            ev_dir = os.path.join("output", "violations")
            if os.path.isdir(ev_dir):
                shutil.rmtree(ev_dir)
                os.makedirs(ev_dir, exist_ok=True)
                st.success("Đã xóa ảnh.")
                st.rerun()

    st.markdown("---")

    # ── Đọc dữ liệu CSV ────────────────────────────────────────────────────────
    import pandas as pd
    df_log = None
    if os.path.exists(csv_path):
        try:
            df_log = pd.read_csv(csv_path)
        except Exception:
            df_log = None

    # ── Đọc danh sách ảnh bằng chứng ──────────────────────────────────────────
    evidence_dir = os.path.join(PROJECT_ROOT, "output", "violations")
    all_img_paths = []
    if os.path.isdir(evidence_dir):
        all_img_paths = sorted(
            glob.glob(os.path.join(evidence_dir, "*.jpg")) +
            glob.glob(os.path.join(evidence_dir, "*.png")),
            key=os.path.getmtime,
            reverse=True   # Mới nhất lên đầu
        )

    # Thống kê nhanh ở đầu trang
    total_viol = len(df_log) if df_log is not None else 0
    total_imgs = len(all_img_paths)
    m1, m2, m3 = st.columns(3)
    m1.metric("📋 Tổng bản ghi CSV", total_viol)
    m2.metric("🖼️ Tổng ảnh bằng chứng", total_imgs)
    if df_log is not None and "violation_type" in df_log.columns:
        m3.metric("🚨 Loại vi phạm", df_log["violation_type"].nunique())
    else:
        m3.metric("🚨 Loại vi phạm", "—")

    st.markdown("---")

    # ── Hai cột: CSV bên trái | Gallery bên phải ──────────────────────────────
    col_log, col_gallery = st.columns([1, 1])

    # ── CỘT TRÁI: Bảng CSV + filter ───────────────────────────────────────────
    with col_log:
        st.subheader("📋 Nhật Ký Vi Phạm")

        if df_log is not None and len(df_log) > 0:
            # Filter theo loại vi phạm
            vtype_options = ["Tất cả"] + sorted(df_log["violation_type"].unique().tolist()) \
                if "violation_type" in df_log.columns else ["Tất cả"]
            selected_vtype = st.selectbox(
                "🔍 Lọc theo loại vi phạm",
                options=vtype_options,
                key="csv_filter"
            )

            df_filtered = df_log.copy()
            if selected_vtype != "Tất cả" and "violation_type" in df_log.columns:
                df_filtered = df_filtered[df_filtered["violation_type"] == selected_vtype]

            st.caption(f"Hiển thị **{len(df_filtered)}** / {len(df_log)} bản ghi")

            # Tô màu hàng theo loại vi phạm
            def _highlight(row):
                vt = str(row.get("violation_type", "")).upper()
                if "UNSAFE" in vt:   return ["background-color: rgba(255,50,50,0.2)"]  * len(row)
                if "CUT_IN" in vt:   return ["background-color: rgba(255,165,0,0.2)"]  * len(row)
                if "LANE" in vt:     return ["background-color: rgba(255,255,0,0.12)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_filtered.style.apply(_highlight, axis=1),
                use_container_width=True,
                height=480,
            )
        else:
            st.info("📂 Chưa có dữ liệu CSV.\n\nChạy video để bắt đầu ghi nhận vi phạm.")

    # ── CỘT PHẢI: Gallery ảnh bằng chứng ─────────────────────────────────────
    with col_gallery:
        st.subheader("🖼️ Bằng Chứng Hình Ảnh")

        if not all_img_paths:
            st.info("📷 Chưa có ảnh bằng chứng nào.\n\nẢnh sẽ được lưu tự động khi vi phạm được xác nhận.")
        else:
            # ── Bộ lọc & phân trang ─────────────────────────────────────────
            filter_col, page_col = st.columns([2, 1])

            with filter_col:
                # Filter theo từ khóa tên file (chứa loại vi phạm, track_id, timestamp)
                search_kw = st.text_input(
                    "🔍 Tìm kiếm (tên file / loại lỗi)",
                    placeholder="VD: UNSAFE, CUT_IN, ID_5...",
                    key="img_search"
                )

            # Lọc ảnh theo keyword
            if search_kw.strip():
                kw = search_kw.strip().upper()
                filtered_paths = [p for p in all_img_paths if kw in os.path.basename(p).upper()]
            else:
                filtered_paths = all_img_paths

            PAGE_SIZE = 12   # Số ảnh mỗi trang
            total_pages = max(1, (len(filtered_paths) + PAGE_SIZE - 1) // PAGE_SIZE)

            with page_col:
                page_num = st.number_input(
                    f"Trang (/{total_pages})",
                    min_value=1, max_value=total_pages,
                    value=1, step=1,
                    key="gallery_page"
                )

            start_idx = (page_num - 1) * PAGE_SIZE
            page_paths = filtered_paths[start_idx : start_idx + PAGE_SIZE]

            st.caption(
                f"**{len(filtered_paths)}** ảnh"
                + (f" (lọc từ {len(all_img_paths)})" if search_kw.strip() else "")
                + f" — Trang {page_num}/{total_pages}"
            )

            # ── Render grid 2 cột ───────────────────────────────────────────
            if not page_paths:
                st.warning("Không tìm thấy ảnh nào khớp từ khóa.")
            else:
                gc1, gc2 = st.columns(2)
                for idx, img_path in enumerate(page_paths):
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        continue

                    fname = os.path.basename(img_path)
                    # Resize ảnh bằng chứng về max 400px để grid gọn
                    thumb = fit_frame(img_bgr, max_w=400, max_h=300)

                    target_col = gc1 if idx % 2 == 0 else gc2
                    with target_col:
                        st.image(thumb, caption=fname, use_container_width=False, width=340)

                        # Xem ảnh to khi cần — dùng expander làm lightbox đơn giản
                        with st.expander("🔍 Xem ảnh to"):
                            full_img = fit_frame(img_bgr, max_w=640, max_h=480)
                            st.image(full_img, caption=fname, use_container_width=False)

                        st.markdown("---")

            # Nút chuyển trang nhanh
            nav1, nav2, nav3 = st.columns([1, 2, 1])
            with nav1:
                if page_num > 1:
                    if st.button("◀ Trước", key="prev_page", use_container_width=True):
                        st.session_state["gallery_page"] = page_num - 1
                        st.rerun()
            with nav3:
                if page_num < total_pages:
                    if st.button("Sau ▶", key="next_page", use_container_width=True):
                        st.session_state["gallery_page"] = page_num + 1
                        st.rerun()