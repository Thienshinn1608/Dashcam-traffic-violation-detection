"""
lane_detector.py  — v5
Wrapper cho Ultra-Fast Lane Detection (UFLD) pretrained TUSimple.
Fallback tự động sang Hough Lines nếu UFLD load thất bại.

Author  : AI Engineer
Project : Traffic Violation Detection System (Đồ án tốt nghiệp)

FIXES APPLIED (v1 — original):
  #1  Backbone đúng: torchvision ResNet18
  #2  exist decode đúng: argmax thay vì threshold logit âm
  #3  ROW_ANCHOR 56 giá trị khớp TUSimple pretrained
  #4  Vanishing point: truyền List[List[Tuple]] đúng kiểu
  #5  Mirror tọa độ âm: clamp x vào [0, w-1]
  #6  Grayscale frame: convert sang BGR

FIXES APPLIED (v2 — Hough improvements):
  #7  Hough ROI mở rộng: int(h * 0.45) thay vì int(h/2)+80
  #8  Hough slope filter nới lỏng: 0.3→3.5
  #9  Hough left/right dùng centroid_x thay vì x1 cứng
  #10 Hough polyfit clamp x vào [0, w-1]
  #11 UFLD x_ratio filter: 0.005/0.995 thay vì 0.01/0.99
  #12 get_ego_lane_boundaries tách path UFLD/Hough
  #13 Virtual lane horizon_y đồng bộ HORIZON_Y_RATIO=0.62
  #14 _hough_average_lines guard y-range quá nhỏ

FIXES APPLIED (v3 — curve & smoothing):
  #15 Polynomial fit bậc 2: lane được fit curve thay vì chỉ đường thẳng
      → bắt được đường cong, ngã rẽ, highway on-ramp
  #16 Temporal smoothing: EMA (Exponential Moving Average) trên hệ số
      polynomial giữa các frame → lane không rung lắc theo frame
  #17 Adaptive color threshold: tự điều chỉnh ngưỡng L (lightness) dựa
      trên độ sáng trung bình ROI → hoạt động cả ban đêm, mưa, ngược sáng
  #18 Hough minLineLength tăng: 30→55 để lọc bỏ noise ngắn không phải vạch
  #19 LaneSmootherPoly class: quản lý EMA poly coefficients cho left/right
  #20 _poly_lane_to_boundary(): convert poly curve → 2 điểm boundary
      để tương thích ngược với main_pipeline.py (không cần sửa pipeline)

FIXES APPLIED (v4 — double-smoother & stability):
  #21 Slope upper bound hạ: 8.0→3.5 — loại bỏ cột điện/cạnh xe dựng đứng
      bị lọt vào do nới quá rộng ở fix trước
  #22 Yellow mask S_min giảm: 100→60 — bắt vạch vàng mờ khi mưa/sương
  #23 warnings.catch_warnings scope thu hẹp: chỉ bắt RankWarning, không
      ẩn hết warning trong _fit_poly2()
  #24 FIX CRITICAL: detect_lanes_hough() tách smoother=None default,
      smoother chỉ update 1 lần/frame — tránh double-update khi
      get_ego_lane_boundaries() + detect() cùng gọi trong 1 frame
  #25 FIX HIGH: UFLD path trong get_ego_lane_boundaries() áp dụng
      smoother sau khi convert pts → poly coeff để lane UFLD cũng
      được làm mượt qua EMA
  #26 FIX HIGH: reset_smoother() expose thêm flag _smoother_dirty
      để main_pipeline biết khi nào cần reset (gọi qua property)
  #27 _cached_hough: cache kết quả Hough trong frame hiện tại,
      get_ego_lane_boundaries() tái dùng thay vì chạy lại Hough lần 2
"""

import os
import sys
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

# ==========================================
# UFLD CONFIG
# ==========================================
UFLD_MODEL_PATH_PTH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "models", "curvelanes_finetuned_best.pth")
UFLD_MODEL_PATH_PT  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "models", "tusimple_18.pt")
UFLD_GRIDING_NUM = 100 
UFLD_NUM_LANES   = 4
UFLD_IMG_W       = 800 
UFLD_IMG_H       = 288
ROI_TOP_RATIO    = 0.40 
 
UFLD_CLS_NUM    = 56
# Ngưỡng tin cậy tối thiểu cho mỗi điểm lane (xác suất đỉnh trên 100 bin vị trí).
# Phân bố "phẳng" (model không chắc) có đỉnh ~0.01 → loại; điểm chắc chắn đỉnh cao hơn.
# Giữ thấp (0.10) để chỉ loại điểm thực sự nhiễu, không bỏ sót lane thật.
UFLD_POINT_CONF = 0.10
UFLD_ROW_ANCHOR = [
     64,  68,  72,  76,  80,  84,  88,  92,  96, 100,
    104, 108, 112, 116, 120, 124, 128, 132, 136, 140,
    144, 148, 152, 156, 160, 164, 168, 172, 176, 180,
    184, 188, 192, 196, 200, 204, 208, 212, 216, 220,
    224, 228, 232, 236, 240, 244, 248, 252, 256, 260,
    264, 268, 272, 276, 280, 284,
]
assert len(UFLD_ROW_ANCHOR) == UFLD_CLS_NUM

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# #13: đồng bộ với main_pipeline.py
HORIZON_Y_RATIO = 0.62

# #16: EMA alpha — 0.0=hoàn toàn dùng cũ, 1.0=hoàn toàn dùng mới
# Lane giờ update thưa hơn (async + throttle ~1.5 lần/giây) nên cần alpha CAO hơn
# để bám kịp đường cong, nếu để 0.05 thì lane sẽ rất trễ/lệch so với thực tế.
EMA_ALPHA = 0.30

# v5: Giới hạn vùng vẽ lane để không kéo thành chữ X lên tận trời.
# 0.62 nghĩa là điểm top của lane chỉ nằm ở ~62% chiều cao frame.
LANE_TOP_RATIO = 0.45
VIRTUAL_HALF_TOP_RATIO = 1 / 14
VIRTUAL_HALF_BOTTOM_RATIO = 1 / 2.5


# ==========================================
# BACKBONE (giữ nguyên v1/v2)
# ==========================================
try:
    from torchvision.models import resnet18 as _tv_resnet18
    _TORCHVISION_OK = True
except Exception:
    _TORCHVISION_OK = False


class UFLDNet(nn.Module):
    """
    Ultra-Fast Lane Detection network — khớp với tusimple_18.pth.

    Kiến trúc thật của PTH (khác với code cũ dùng avgpool + 1 Linear):
      backbone : ResNet18  (key prefix: "model.conv1", "model.layer1", ...)
      pool     : Conv2d(512, 8, 1x1)   → spatial pooling
      cls      : Sequential(Linear(1800, 2048), ReLU, Linear(2048, 101*56*4))

    Input  : (B, 3, 288, 800)
    Output : (B, 101, 56, 4)  — (griding_num+1, cls_num, num_lanes)

    FIX v5: Sửa lại đúng architecture sau khi phân tích state_dict của
            tusimple_18.pth — code cũ dùng avgpool+Linear(512) sẽ không
            load được weights (102 missing keys, 126 unexpected keys).
            Key mapping: model.conv1→model.0, model.layer1→model.4, ...
    """
    def __init__(self, griding_num=UFLD_GRIDING_NUM, cls_num=UFLD_CLS_NUM,
                 num_lanes=UFLD_NUM_LANES):
        super().__init__()
        self.griding_num = griding_num
        self.cls_num     = cls_num
        self.num_lanes   = num_lanes

        if _TORCHVISION_OK:
            backbone = _tv_resnet18(weights=None)
            # Sequential index phải khớp với key prefix trong PTH:
            # model.0=conv1, model.1=bn1, model.2=relu, model.3=maxpool,
            # model.4=layer1, model.5=layer2, model.6=layer3, model.7=layer4
            self.model = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            )
        else:
            self.model = self._build_fallback_backbone()

        # pool: Conv2d(512, 8, 1x1) — spatial pooling trước flatten
        # Output shape sau pool: (B, 8, 9, 25) → flatten → (B, 1800)
        self.pool = nn.Conv2d(512, 8, 1)

        # cls: 2-layer MLP khớp PTH
        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, (griding_num + 1) * cls_num * num_lanes),
        )

    @staticmethod
    def _build_fallback_backbone():
        """Fallback khi torchvision không có — giữ nguyên key naming."""
        class _Block(nn.Module):
            def __init__(self, ic, oc, stride=1):
                super().__init__()
                self.c = nn.Sequential(
                    nn.Conv2d(ic, oc, 3, stride, 1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
                    nn.Conv2d(oc, oc, 3, 1,      1, bias=False), nn.BatchNorm2d(oc),
                )
                self.d = nn.Sequential(
                    nn.Conv2d(ic, oc, 1, stride, bias=False), nn.BatchNorm2d(oc)
                ) if (stride != 1 or ic != oc) else nn.Identity()
                self.r = nn.ReLU(True)
            def forward(self, x): return self.r(self.c(x) + self.d(x))

        def _layer(ic, oc, n, s):
            return nn.Sequential(_Block(ic, oc, s), *[_Block(oc, oc) for _ in range(n-1)])

        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
            _layer( 64,  64, 2, 1),
            _layer( 64, 128, 2, 2),
            _layer(128, 256, 2, 2),
            _layer(256, 512, 2, 2),
        )

    def forward(self, x):
        x = self.model(x)    # (B, 512, 9, 25) với input 288×800
        x = self.pool(x)     # (B, 8, 9, 25)
        x = x.flatten(1)     # (B, 1800)
        x = self.cls(x)      # (B, 101*56*4)
        return x.view(-1, self.griding_num + 1, self.cls_num, self.num_lanes)


# ==========================================
# HELPER: BGR guard
# ==========================================
def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# ==========================================
# #19: LANE SMOOTHER — EMA trên poly coefficients
# ==========================================
class LaneSmootherPoly:
    """
    #16 + #19: Temporal smoothing cho polynomial lane.

    Lưu hệ số poly (a, b, c) của mỗi lane (left/right).
    Mỗi frame mới: coeff_smooth = EMA_ALPHA * coeff_new + (1-EMA_ALPHA) * coeff_old
    → Lane không nhảy đột ngột giữa các frame.

    #24 v4: Thêm _frame_updated để detect() và get_ego_lane_boundaries()
    không update smoother 2 lần trong cùng 1 frame.
    """

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        self.alpha    = alpha
        self._left:  Optional[np.ndarray] = None   # shape (3,) — [a, b, c] bậc 2
        self._right: Optional[np.ndarray] = None
        self._frame_updated: int = -1               # #24: frame_id lần cập nhật gần nhất

    def update(
        self,
        left_coeff:  Optional[np.ndarray],
        right_coeff: Optional[np.ndarray],
        frame_id:    int = -1,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Nhận hệ số poly mới, trả về hệ số đã làm mượt.
        Nếu frame này không detect được lane → trả về hệ số cũ (giữ nguyên).

        #24 v4: Nếu frame_id == _frame_updated → bỏ qua update, chỉ trả về
        hệ số đã smooth (tránh double-update khi 2 hàm cùng gọi trong 1 frame).
        """
        if frame_id >= 0 and frame_id == self._frame_updated:
            # Đã update frame này rồi — trả về cached result, không EMA lần 2
            return (
                self._left.copy()  if self._left  is not None else None,
                self._right.copy() if self._right is not None else None,
            )

        if left_coeff is not None:
            if self._left is None:
                self._left = left_coeff.copy()
            else:
                self._left = self.alpha * left_coeff + (1 - self.alpha) * self._left

        if right_coeff is not None:
            if self._right is None:
                self._right = right_coeff.copy()
            else:
                self._right = self.alpha * right_coeff + (1 - self.alpha) * self._right

        if frame_id >= 0:
            self._frame_updated = frame_id

        return (
            self._left.copy()  if self._left  is not None else None,
            self._right.copy() if self._right is not None else None,
        )

    def reset(self) -> None:
        """Gọi khi scene change lớn để tránh dùng coeff cũ sai."""
        self._left         = None
        self._right        = None
        self._frame_updated = -1


# ==========================================
# #15: POLYNOMIAL FIT BẬC 2
# ==========================================
def _fit_poly2(
    lines: list, h: int, w: int
) -> Optional[np.ndarray]:
    """
    #15: Fit polynomial bậc 2 x = f(y) từ list HoughLinesP.
    Trả về array hệ số [a, b, c] hoặc None nếu không đủ điểm.

    Dùng fit x=f(y) thay vì y=f(x) vì lane đường dọc theo y,
    tránh vô định khi lane thẳng đứng (infinite slope).

    #23 v4: warnings.catch_warnings chỉ filter RankWarning, không ẩn hết.
    """
    x_pts, y_pts = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        for t in np.linspace(0, 1, 5):
            x_pts.append(x1 + t * (x2 - x1))
            y_pts.append(y1 + t * (y2 - y1))

    if len(x_pts) < 6:
        return None

    y_arr  = np.array(y_pts, dtype=np.float64)
    x_arr  = np.array(x_pts, dtype=np.float64)
    y_norm = y_arr / h

    try:
        # #23 v4: chỉ suppress RankWarning, không dùng "ignore" toàn bộ
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning
                                  if hasattr(np, "exceptions") else np.RankWarning)
            coeff = np.polyfit(y_norm, x_arr, 2)   # x = a*yn² + b*yn + c

        # Sanity check: hệ số bậc 2 quá lớn → đường cong bất thường → fallback bậc 1
        if abs(coeff[0]) > w * 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.exceptions.RankWarning
                                      if hasattr(np, "exceptions") else np.RankWarning)
                coeff = np.array([0.0, *np.polyfit(y_norm, x_arr, 1)])
        return coeff
    except Exception:
        return None


def _poly_eval(coeff: np.ndarray, y: float, h: int, w: int) -> int:
    """Evaluate poly tại y pixel, trả về x pixel đã clamp vào [0, w-1]."""
    y_norm = y / h
    x = np.polyval(coeff, y_norm)
    return int(np.clip(x, 0, w - 1))


def _poly_lane_to_boundary(
    coeff: np.ndarray, h: int, w: int, y_top: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    #20 + v5: Convert poly coefficients → 2 điểm (bottom, top).

    v5 fix:
      - Không cho y_top kéo quá cao lên trời/horizon thật.
      - Top lane nằm trong vùng gần xe hơn để UI không thành chữ X lớn.
    """
    safe_y_top = max(int(h * LANE_TOP_RATIO), int(y_top))
    safe_y_top = min(max(safe_y_top, 0), h - 1)

    x_bot = _poly_eval(coeff, h - 1,      h, w)
    x_top = _poly_eval(coeff, safe_y_top, h, w)
    return ((x_bot, h - 1), (x_top, safe_y_top))


def _is_sane_ego_lane(
    left_lane: Tuple[Tuple[int, int], Tuple[int, int]],
    right_lane: Tuple[Tuple[int, int], Tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    """
    v5: Chặn lane bị bắt nhầm thành chữ X.

    Format lane: ((x_bottom, y_bottom), (x_top, y_top))
    Lane hợp lệ phải có:
      - left luôn nằm bên trái right ở cả bottom và top
      - độ rộng top nhỏ hơn bottom, nhưng không âm/không quá nhỏ
      - độ rộng bottom nằm trong khoảng hợp lý của ego lane
    """
    if left_lane is None or right_lane is None:
        return False

    lx_bot, ly_bot = left_lane[0]
    lx_top, ly_top = left_lane[1]
    rx_bot, ry_bot = right_lane[0]
    rx_top, ry_top = right_lane[1]

    bottom_width = rx_bot - lx_bot
    top_width = rx_top - lx_top

    if bottom_width <= 0 or top_width <= 0:
        return False

    # Ego lane width limits (rất nới lỏng)
    if bottom_width < w * 0.05 or bottom_width > w * 1.5:
        return False

    if top_width < 0 or top_width > bottom_width * 1.5:
        return False

    # Hỗ trợ góc nhìn xa, chỉ bỏ khi lane đâm thẳng lên trời (top y quá nhỏ)
    if ly_top < h * 0.15 or ry_top < h * 0.15:
        return False

    return True


def _safe_ego_lane_or_virtual(
    frame: np.ndarray,
    left_lane: Tuple[Tuple[int, int], Tuple[int, int]],
    right_lane: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Tuple[
    Tuple[Tuple[int, int], Tuple[int, int]],
    Tuple[Tuple[int, int], Tuple[int, int]],
]:
    """Trả lane detect nếu hợp lệ, ngược lại fallback virtual lane gọn."""
    h, w = frame.shape[:2]
    if _is_sane_ego_lane(left_lane, right_lane, h, w):
        return left_lane, right_lane
    return _virtual_ego_lane(frame)


# ==========================================
# #17: ADAPTIVE COLOR THRESHOLD
# ==========================================
def _adaptive_hls_masks(
    frame_enhanced: np.ndarray, roi_y: int
) -> np.ndarray:
    """
    #17: Tự động điều chỉnh ngưỡng L (lightness) theo độ sáng ROI.

    Ban ngày:  ROI sáng → threshold L cao (160-255) → lọc chặt → ít nhiễu
    Ban đêm:   ROI tối  → threshold L thấp (80-255)  → bắt vạch phản quang
    Mưa/sương: ROI mờ  → giảm threshold trung bình để bắt vạch mờ

    #22 v4: Yellow mask S_min giảm từ 100→60 để bắt vạch vàng mờ khi mưa/sương.
    """
    h, w = frame_enhanced.shape[:2]
    roi  = frame_enhanced[roi_y:, :]

    hls       = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    mean_l    = float(np.mean(l_channel))

    # Adaptive L_min: tỷ lệ theo độ sáng trung bình, clamp [50, 175]
    l_min = int(np.clip(mean_l * 1.1, 50, 175))

    white_mask = cv2.inRange(hls,
                             np.array([  0, l_min,   0], dtype=np.uint8),
                             np.array([255,   255, 255], dtype=np.uint8))

    # #22 v4: S_min 100→60 để bắt vạch vàng mờ trong điều kiện xấu
    yellow_mask = cv2.inRange(hls,
                              np.array([ 15,  30,  60], dtype=np.uint8),
                              np.array([ 40, 220, 255], dtype=np.uint8))

    mask      = cv2.bitwise_or(white_mask, yellow_mask)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[roi_y:, :] = mask
    return full_mask


# ==========================================
# HOUGH LINES FALLBACK — v4
# ==========================================
def detect_lanes_hough(
    frame: np.ndarray,
    smoother: Optional["LaneSmootherPoly"] = None,
    frame_id: int = -1,
) -> Tuple[
    Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
    Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
]:
    """
    Hough Lines + Polynomial fit bậc 2 + Temporal smoothing.
    Trả về (left_lane, right_lane): ((x_bot, h), (x_top, y_top)).

    smoother : LaneSmootherPoly instance — None thì bỏ qua temporal smoothing.
    frame_id : Frame hiện tại, truyền để smoother tránh double-update (#24 v4).

    #21 v4: Slope upper bound 8.0→3.5 (loại bỏ đường thẳng đứng không phải lane).
    #24 v4: Truyền frame_id vào smoother.update() để idempotent trong 1 frame.
    """
    frame = _ensure_bgr(frame)
    h, w  = frame.shape[:2]
    cx    = w // 2
    y_top_roi = int(h * HORIZON_Y_RATIO)

    # --- Tăng tương phản CLAHE ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
    frame_enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # #17: Adaptive color mask
    color_mask = _adaptive_hls_masks(frame_enhanced, y_top_roi)

    blur  = cv2.GaussianBlur(color_mask, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 130)

    roi_vertices = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.95), h),
        (int(w * 0.60), y_top_roi),
        (int(w * 0.40), y_top_roi),
    ]], dtype=np.int32)
    mask_roi     = np.zeros_like(edges)
    cv2.fillPoly(mask_roi, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask_roi)

    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180,
        threshold=25,       # Tăng độ nhạy bắt vạch kẻ mờ
        maxLineGap=200,     # Nối các vạch đứt quãng xa hơn
        minLineLength=30,   # Bắt các vạch đứt quãng ngắn tốt hơn
    )

    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # #21 v4: upper bound 8.0→3.5 — loại cột điện/cạnh xe dựng đứng
            if abs(slope) < 0.3 or abs(slope) > 3.5:
                continue
            centroid_x = (x1 + x2) / 2.0
            if slope < 0 and centroid_x < cx:
                left_lines.append(line)
            elif slope > 0 and centroid_x > cx:
                right_lines.append(line)

    # #15: Polynomial fit bậc 2
    left_coeff  = _fit_poly2(left_lines,  h, w) if left_lines  else None
    right_coeff = _fit_poly2(right_lines, h, w) if right_lines else None

    # #16 + #24 v4: Temporal smoothing — truyền frame_id để tránh double-update
    if smoother is not None:
        left_coeff, right_coeff = smoother.update(left_coeff, right_coeff, frame_id=frame_id)

    # Convert poly coefficients → boundary points
    left_lane  = _poly_lane_to_boundary(left_coeff,  h, w, y_top_roi) if left_coeff  is not None else None
    right_lane = _poly_lane_to_boundary(right_coeff, h, w, y_top_roi) if right_coeff is not None else None

    return left_lane, right_lane


# ==========================================
# VIRTUAL EGO LANE (last-resort fallback)
# ==========================================
def _virtual_ego_lane(
    frame: np.ndarray,
) -> Tuple[
    Tuple[Tuple[int, int], Tuple[int, int]],
    Tuple[Tuple[int, int], Tuple[int, int]],
]:
    h, w      = frame.shape[:2]
    top_y     = int(h * LANE_TOP_RATIO)
    half_top  = int(w * VIRTUAL_HALF_TOP_RATIO)
    half_bot  = int(w * VIRTUAL_HALF_BOTTOM_RATIO)

    # v5: virtual lane gọn hơn, không kéo lên horizon thật nên không thành chữ X.
    left_lane  = ((w // 2 - half_bot, h - 1), (w // 2 - half_top, top_y))
    right_lane = ((w // 2 + half_bot, h - 1), (w // 2 + half_top, top_y))
    return left_lane, right_lane


# ==========================================
# HELPER: pts list → poly coeff (cho UFLD path)
# ==========================================
def _select_ego_lane_pts(
    lanes: Dict[int, List[Tuple[int, int]]], cx: int
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """
    Chọn 2 làn bao quanh xe ego: làn gần tâm nhất ở BÊN TRÁI và BÊN PHẢI.

    Robust hơn việc hardcode UFLD index 1/2 — vì khi xe lệch làn, đổi làn,
    hoặc UFLD chỉ bắt được vài làn, index cố định sẽ chọn sai làn ego.
    Ở đây xét tọa độ x tại điểm THẤP NHẤT (gần xe nhất) của mỗi làn so với
    tâm khung hình để quyết định trái/phải và độ gần.
    """
    left_best, right_best = None, None
    left_dx, right_dx = float("inf"), float("inf")
    for pts in lanes.values():
        if pts is None or len(pts) < 2:
            continue
        bot_x = max(pts, key=lambda p: p[1])[0]   # x ở điểm gần xe nhất
        if bot_x <= cx:
            dx = cx - bot_x
            if dx < left_dx:
                left_dx, left_best = dx, pts
        else:
            dx = bot_x - cx
            if dx < right_dx:
                right_dx, right_best = dx, pts
    return left_best, right_best


def _pts_to_poly2(pts: list, h: int, w: int) -> Optional[np.ndarray]:
    """
    #25 v4: Convert list of (x, y) points → poly coeff bậc 2.
    Dùng để UFLD path cũng đi qua smoother EMA.

    v6: Thêm 1 vòng loại điểm ngoại lai (outlier) dựa trên residual — fit thử,
    bỏ điểm lệch > 2.5*std rồi fit lại → đường cong không bị 1-2 điểm nhiễu kéo lệch.
    """
    if len(pts) < 3:
        return None
    xs     = np.array([p[0] for p in pts], dtype=np.float64)
    ys     = np.array([p[1] for p in pts], dtype=np.float64)
    y_norm = ys / h
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning
                                  if hasattr(np, "exceptions") else np.RankWarning)
            coeff = np.polyfit(y_norm, xs, 2)

            # v6: Loại outlier theo residual rồi fit lại (giữ nếu còn đủ điểm)
            resid = np.abs(np.polyval(coeff, y_norm) - xs)
            std   = float(np.std(resid))
            if std > 1e-6:
                keep = resid <= 2.5 * std
                if keep.sum() >= 3 and keep.sum() < len(xs):
                    coeff = np.polyfit(y_norm[keep], xs[keep], 2)

        if abs(coeff[0]) > w * 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.exceptions.RankWarning
                                      if hasattr(np, "exceptions") else np.RankWarning)
                coeff = np.array([0.0, *np.polyfit(y_norm, xs, 1)])
        return coeff
    except Exception:
        return None


# ==========================================
# MAIN CLASS
# ==========================================
class LaneDetector:
    """
    Wrapper UFLD + Hough fallback + Polynomial curve + Temporal smoothing.

    Thứ tự ưu tiên:
      1. UFLD   (nếu model file tồn tại + load thành công)
      2. Hough + Poly fit bậc 2 + EMA smoothing
      3. Virtual Ego Lane (last resort)

    v4 Changes:
      - detect() và get_ego_lane_boundaries() dùng chung _current_frame_id
        để smoother.update() idempotent → không double-update EMA
      - _hough_cache: cache kết quả Hough trong frame hiện tại,
        get_ego_lane_boundaries() dùng lại thay vì chạy Hough lần 2
      - UFLD path trong get_ego_lane_boundaries() giờ cũng qua EMA smoother
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self._use_ufld = False
        self._model: Optional[UFLDNet] = None

        # #19: smoother dùng chung cho detect() và get_ego_lane_boundaries()
        self._smoother = LaneSmootherPoly(alpha=EMA_ALPHA)

        # v6: Lane ego hợp lệ gần nhất — dùng lại khi frame hiện tại detect bất hợp lý
        self._last_good_ego: Optional[Tuple] = None

        # #27 v4: Cache Hough result trong frame hiện tại
        self._hough_cache: Optional[Tuple] = None   # (left_lane, right_lane)
        self._hough_cache_frame: int = -1

        self._ufld_cache: Optional[Dict[int, List[Tuple[int, int]]]] = None
        self._ufld_cache_frame: int = -1

        # Frame id hiện tại để truyền vào smoother (tránh double-update)
        self._current_frame_id: int = 0

        self._try_load_ufld()
        if not self._use_ufld:
            warnings.warn(
                "[LaneDetector] UFLD khong load duoc — dung Hough+Poly fallback.",
                RuntimeWarning, stacklevel=2,
            )
            print("[LaneDetector WARNING] UFLD failed → Hough + Poly bậc 2 + EMA", flush=True)

    def _try_load_ufld(self) -> None:
        model_path = UFLD_MODEL_PATH_PTH
        if not os.path.exists(model_path):
            model_path = UFLD_MODEL_PATH_PT

        if not os.path.exists(model_path):
            print(f"[LaneDetector] Model not found: {UFLD_MODEL_PATH_PTH} or .pt", flush=True)
            return
        try:
            model = UFLDNet(UFLD_GRIDING_NUM, UFLD_CLS_NUM, UFLD_NUM_LANES)
            raw   = torch.load(model_path, map_location="cpu", weights_only=False)
            state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

            # FIX v5: tusimple_18.pth dùng key prefix "model.conv1", "model.layer1" ...
            # nhưng UFLDNet dùng Sequential index "model.0", "model.4" ...
            # Remap để load đúng weights.
            remapped = {}
            for k, v in state.items():
                nk = k
                nk = nk.replace("model.conv1.",  "model.0.")
                nk = nk.replace("model.bn1.",    "model.1.")
                nk = nk.replace("model.layer1.", "model.4.")
                nk = nk.replace("model.layer2.", "model.5.")
                nk = nk.replace("model.layer3.", "model.6.")
                nk = nk.replace("model.layer4.", "model.7.")
                remapped[nk] = v

            missing, unexpected = model.load_state_dict(remapped, strict=False)
            if missing:
                print(f"[LaneDetector] Missing keys ({len(missing)}): {missing[:3]} ...", flush=True)
            if unexpected:
                print(f"[LaneDetector] Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...", flush=True)
            if not missing and not unexpected:
                print(f"[LaneDetector] Weights loaded: PERFECT MATCH (0 missing, 0 unexpected)", flush=True)

            model.to(self.device).eval()
            if self.device.type == "cuda":
                model.half()
            self._model    = model
            self._use_ufld = True
            print(f"[LaneDetector] UFLD loaded OK on {self.device}", flush=True)
        except Exception as e:
            print(f"[LaneDetector] UFLD load error: {e}", flush=True)

    # ------------------------------------------------------------------
    def _preprocess_roi(self, frame: np.ndarray) -> Tuple[torch.Tensor, int]:
        frame  = _ensure_bgr(frame)
        h, w   = frame.shape[:2]
        roi_y  = 0
        inp = cv2.resize(frame, (UFLD_IMG_W, UFLD_IMG_H)).astype(np.float32) / 255.0
        inp = (inp - _MEAN) / _STD
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0)
        if self.device.type == "cuda":
            inp = inp.half()
        return inp.to(self.device), roi_y

    # ------------------------------------------------------------------
    def _decode_ufld(self, out, orig_h, orig_w, roi_y):
        out    = out[0].cpu().float()
        argmax = out.argmax(dim=0)
        loc    = torch.arange(1, UFLD_GRIDING_NUM + 1, dtype=torch.float32)
        prob   = torch.softmax(out[:-1], dim=0)
        expect = (prob * loc.view(-1, 1, 1)).sum(0)
        # Tin cậy mỗi điểm = xác suất đỉnh trên các bin vị trí
        conf_peak = prob.max(dim=0).values

        roi_h  = orig_h - roi_y
        lanes  = {}
        for lane_idx in range(UFLD_NUM_LANES):
            pts = []
            for row_idx in range(UFLD_CLS_NUM):
                if argmax[row_idx, lane_idx].item() == UFLD_GRIDING_NUM:
                    continue
                # Loại điểm model không chắc (phân bố vị trí quá phẳng)
                if conf_peak[row_idx, lane_idx].item() < UFLD_POINT_CONF:
                    continue
                x_ratio = expect[row_idx, lane_idx].item() / UFLD_GRIDING_NUM
                if x_ratio <= 0.005 or x_ratio >= 0.995:
                    continue
                x_roi = int(x_ratio * orig_w)
                y_roi = int(UFLD_ROW_ANCHOR[row_idx] / UFLD_IMG_H * roi_h)
                pts.append((x_roi, y_roi + roi_y))
            # Cần >=3 điểm để fit poly bậc 2 đáng tin (giảm lane ngắn nhiễu)
            if len(pts) >= 3:
                lanes[lane_idx] = pts
        return lanes

    # ------------------------------------------------------------------
    def _get_hough_cached(
        self, frame: np.ndarray, frame_id: int
    ) -> Tuple[
        Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
    ]:
        """
        #27 v4: Trả về kết quả Hough từ cache nếu đã chạy trong frame này,
        hoặc chạy mới và lưu cache.
        Smoother chỉ update 1 lần nhờ frame_id guard trong LaneSmootherPoly.
        """
        if self._hough_cache_frame == frame_id and self._hough_cache is not None:
            return self._hough_cache

        result = detect_lanes_hough(frame, smoother=self._smoother, frame_id=frame_id)
        self._hough_cache       = result
        self._hough_cache_frame = frame_id
        return result

    # ------------------------------------------------------------------
    def _get_ufld_cached(
        self, frame_input: np.ndarray, frame_id: int, h_in: int, w_in: int
    ) -> Dict[int, List[Tuple[int, int]]]:
        if self._ufld_cache_frame == frame_id and self._ufld_cache is not None:
            return self._ufld_cache

        lanes = {}
        if self._use_ufld and self._model is not None:
            try:
                inp, _ = self._preprocess_roi(frame_input)
                with torch.no_grad():
                    out = self._model(inp)
                lanes = self._decode_ufld(out, h_in, w_in, 0)
            except Exception as e:
                print(f"[LaneDetector] UFLD inference error: {e}", flush=True)

        self._ufld_cache = lanes
        self._ufld_cache_frame = frame_id
        return lanes

    # ------------------------------------------------------------------
    def set_frame_id(self, frame_id: int) -> None:
        """
        Gọi từ main_pipeline mỗi đầu frame để đồng bộ frame_id.
        Cho phép smoother và hough_cache biết frame mới bắt đầu.
        Nếu không gọi, frame_id tự tăng qua detect().
        """
        self._current_frame_id = frame_id

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, frame_id: Optional[int] = None) -> List[List[Tuple[int, int]]]:
        """
        Detect tất cả làn trong frame.
        Trả về List[List[Tuple[int,int]]], list rỗng nếu không detect được.

        frame_id: truyền frame number từ pipeline để tránh double EMA.
                  Nếu None → dùng _current_frame_id (tự tăng).
        """
        if frame is None or frame.size == 0:
            return []
        frame = _ensure_bgr(frame)
        h, w  = frame.shape[:2]

        # Tiền xử lý cho video dọc (Portrait): crop phần hình vuông dưới đáy để tránh méo tỉ lệ
        if w < h:
            roi_y = h - w
            frame_input = frame[roi_y:, :]
        else:
            roi_y = 0
            frame_input = frame

        if frame_id is None:
            frame_id = self._current_frame_id
            self._current_frame_id += 1

        detected_lanes = []
        if self._use_ufld and self._model is not None:
            h_in, w_in = frame_input.shape[:2]
            lanes = self._get_ufld_cached(frame_input, frame_id, h_in, w_in)
            if lanes:
                # [SMOOTHING FIX] Fit polynomial curves to all UFLD lanes instead of drawing raw points
                for idx, pts in lanes.items():
                    if len(pts) >= 3:
                        coeff = _pts_to_poly2(pts, h_in, w_in)
                        if coeff is not None:
                            bot_y = float(h_in - 1)
                            top_y = float(min([p[1] for p in pts]))
                            poly_pts = []
                            for y in np.linspace(bot_y, top_y, 16):
                                x = _poly_eval(coeff, float(y), h_in, w_in)
                                poly_pts.append((x, int(y)))
                            detected_lanes.append(poly_pts)
                        else:
                            detected_lanes.append(pts)
                    else:
                        detected_lanes.append(pts)

        if not detected_lanes:
            # Dùng Hough fallback
            left, right = self._get_hough_cached(frame_input, frame_id)
            for lane_line in (left, right):
                if lane_line is not None:
                    (x_bot, y_bot), (x_top, y_top) = lane_line
                    pts = [
                        (int(x_bot + (x_top - x_bot) * t), int(y_bot + (y_top - y_bot) * t))
                        for t in np.linspace(0, 1, 12)
                    ]
                    detected_lanes.append(pts)

        # Trả tọa độ về hệ tọa độ gốc của frame dọc
        if roi_y > 0 and detected_lanes:
            shifted_lanes = []
            for lane in detected_lanes:
                shifted_lanes.append([(x, y + roi_y) for (x, y) in lane])
            return shifted_lanes
        return detected_lanes

    # ------------------------------------------------------------------
    def get_ego_lane_boundaries(
        self, frame: np.ndarray, frame_id: Optional[int] = None
    ) -> Tuple[
        Tuple[Tuple[int, int], Tuple[int, int]],
        Tuple[Tuple[int, int], Tuple[int, int]],
    ]:
        """
        Trả về (left_boundary, right_boundary) của làn ego.
        Luôn có giá trị — virtual lane nếu cần.
        Path UFLD và Hough tách biệt (#12).

        v4: EMA smoother áp dụng cho cả UFLD path (#25).
            Hough path dùng cache → không chạy lại Hough lần 2 (#27).
            frame_id guard → smoother không double-update (#24).
        """
        frame = _ensure_bgr(frame)
        h, w  = frame.shape[:2]

        # Tiền xử lý cho video dọc (Portrait): crop phần hình vuông dưới đáy để tránh méo tỉ lệ
        if w < h:
            roi_y = h - w
            frame_input = frame[roi_y:, :]
        else:
            roi_y = 0
            frame_input = frame

        h_in, w_in = frame_input.shape[:2]
        cx = w_in // 2

        if frame_id is None:
            frame_id = self._current_frame_id

        left_lane, right_lane = None, None
        ufld_success = False

        # --- Path UFLD ---
        if self._use_ufld and self._model is not None:
            try:
                lanes = self._get_ufld_cached(frame_input, frame_id, h_in, w_in)

                # Robust: chọn 2 làn gần tâm xe nhất (trái/phải) thay vì hardcode index 1/2.
                left_pts, right_pts = _select_ego_lane_pts(lanes, cx)
                if left_pts is None:
                    left_pts = lanes.get(1)
                if right_pts is None:
                    right_pts = lanes.get(2)

                if left_pts is not None and right_pts is not None:
                    # Dùng điểm để nội suy ra phương trình bậc 2
                    left_coeff  = _pts_to_poly2(left_pts,  h_in, w_in)
                    right_coeff = _pts_to_poly2(right_pts, h_in, w_in)
                    left_coeff, right_coeff = self._smoother.update(
                        left_coeff, right_coeff, frame_id=frame_id
                    )

                    y_top_roi = int(h_in * HORIZON_Y_RATIO)
                    if left_coeff is not None and right_coeff is not None:
                        left_lane = _poly_lane_to_boundary(left_coeff,  h_in, w_in, y_top_roi)
                        right_lane = _poly_lane_to_boundary(right_coeff, h_in, w_in, y_top_roi)
                        ufld_success = True
                    else:
                        def _pts_to_line(pts):
                            bot = max(pts, key=lambda p: p[1])
                            top = min(pts, key=lambda p: p[1])
                            return (bot, top)
                        left_lane = _pts_to_line(left_pts)
                        right_lane = _pts_to_line(right_pts)
                        ufld_success = True

            except Exception as e:
                print(f"[LaneDetector] UFLD ego boundaries error: {e}", flush=True)

        if not ufld_success:
            # --- Path Hough + Poly + EMA ---
            left_hough, right_hough = self._get_hough_cached(frame_input, frame_id)
            y_top_roi = int(h_in * HORIZON_Y_RATIO)

            if left_hough is not None and right_hough is not None:
                left_lane, right_lane = left_hough, right_hough
            else:
                lane_width_bot = int(w_in * 0.8)
                lane_width_top = int(w_in * 0.14)

                if left_hough is not None:
                    left_lane = left_hough
                    (xl_bot, _), (xl_top, _) = left_hough
                    rx_bot = min(w_in - 1, xl_bot + lane_width_bot)
                    rx_top = min(w_in - 1, xl_top + lane_width_top)
                    right_lane = ((rx_bot, h_in - 1), (rx_top, y_top_roi))
                elif right_hough is not None:
                    right_lane = right_hough
                    (xr_bot, _), (xr_top, _) = right_hough
                    lx_bot = max(0, xr_bot - lane_width_bot)
                    lx_top = max(0, xr_top - lane_width_top)
                    left_lane = ((lx_bot, h_in - 1), (lx_top, y_top_roi))
                else:
                    left_lane, right_lane = _virtual_ego_lane(frame_input)

        # v6: Giữ lane TỐT gần nhất thay vì nhảy về virtual lane (giả, ở giữa)
        # khi frame hiện tại detect bất hợp lý → lane ổn định, không nhấp nháy.
        h_f, w_f = frame_input.shape[:2]
        if _is_sane_ego_lane(left_lane, right_lane, h_f, w_f):
            self._last_good_ego = (left_lane, right_lane)
        elif self._last_good_ego is not None:
            left_lane, right_lane = self._last_good_ego
        else:
            left_lane, right_lane = _virtual_ego_lane(frame_input)

        # Trả tọa độ về hệ tọa độ gốc nếu là video dọc
        if roi_y > 0:
            def _shift_line(line):
                if line is None: return None
                (x1, y1), (x2, y2) = line
                return ((x1, y1 + roi_y), (x2, y2 + roi_y))
            left_lane = _shift_line(left_lane)
            right_lane = _shift_line(right_lane)

        return left_lane, right_lane

    # ------------------------------------------------------------------
    def get_vanishing_point(self, lanes: List[List[Tuple[int, int]]]) -> int:
        if not lanes or len(lanes) < 2:
            return -1

        def _fit_line(pts):
            if len(pts) < 2:
                return None
            xs = np.array([p[0] for p in pts], dtype=np.float64)
            ys = np.array([p[1] for p in pts], dtype=np.float64)
            if np.std(xs) < 1e-6:
                return None
            try:
                a, b = np.polyfit(xs, ys, 1)
                return float(a), float(b)
            except Exception:
                return None

        fits = [f for f in [_fit_line(l) for l in lanes] if f is not None]
        intersections_y = []
        for i in range(len(fits)):
            for j in range(i + 1, len(fits)):
                a1, b1 = fits[i]
                a2, b2 = fits[j]
                if abs(a1 - a2) < 1e-6:
                    continue
                x_int = (b2 - b1) / (a1 - a2)
                y_int = a1 * x_int + b1
                intersections_y.append(y_int)

        if not intersections_y:
            return -1
        return int(np.median(intersections_y))

    # ------------------------------------------------------------------
    def reset_smoother(self) -> None:
        """
        Gọi khi scene change lớn (camera xoay, cắt cảnh) để
        EMA không dùng hệ số lane từ cảnh cũ.
        v4: Cũng reset hough_cache để tránh dùng kết quả cũ.
        """
        self._smoother.reset()
        self._hough_cache       = None
        self._hough_cache_frame = -1
        self._last_good_ego     = None
        print("[LaneDetector] Smoother + cache reset (scene change detected)", flush=True)


# ==========================================
# v5: DRAW HELPER CHO MAIN PIPELINE
# ==========================================
def draw_ego_lane_overlay(
    frame: np.ndarray,
    left_lane: Tuple[Tuple[int, int], Tuple[int, int]],
    right_lane: Tuple[Tuple[int, int], Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 180),
    alpha: float = 0.25,
    draw_border: bool = True,
) -> np.ndarray:
    """
    Vẽ vùng lane kiểu ADAS thay vì 2 đường chéo dài.

    Dùng trong main_pipeline thay cho cv2.line(...):
        left_b, right_b = lane_detector.get_ego_lane_boundaries(frame, frame_id)
        frame = draw_ego_lane_overlay(frame, left_b, right_b)
    """
    if frame is None or frame.size == 0:
        return frame

    vis = frame.copy()
    h, w = vis.shape[:2]

    if not _is_sane_ego_lane(left_lane, right_lane, h, w):
        left_lane, right_lane = _virtual_ego_lane(vis)

    poly_pts = np.array([
        left_lane[0],
        left_lane[1],
        right_lane[1],
        right_lane[0],
    ], dtype=np.int32)

    overlay = vis.copy()
    cv2.fillPoly(overlay, [poly_pts], color)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    if draw_border:
        cv2.line(vis, left_lane[0], left_lane[1], color, 3)
        cv2.line(vis, right_lane[0], right_lane[1], color, 3)

    return vis


# ==========================================
# TEST ĐỘC LẬP
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None)
    args = parser.parse_args()

    if args.img and os.path.exists(args.img):
        frame = cv2.imread(args.img)
        if frame is None:
            frame = None
        else:
            print(f"[TEST] Dung anh: {args.img}")

    if not (args.img and os.path.exists(str(args.img or ""))):
        print("[TEST] Dung anh synthetic")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for y in range(720):
            gray = int(30 + 80 * y / 720)
            frame[y, :] = (gray, gray, gray)
        for y in range(400, 720):
            t       = (y - 400) / 320.0
            x_left  = int(400 - 80 * t + 30 * t * t)
            x_right = int(880 + 80 * t - 30 * t * t)
            x_left  = max(0, min(1279, x_left))
            x_right = max(0, min(1279, x_right))
            cv2.line(frame, (x_left - 4, y), (x_left + 4, y), (220, 220, 220), 2)
            cv2.line(frame, (x_right - 4, y), (x_right + 4, y), (220, 220, 220), 2)
        cv2.putText(frame, "SYNTHETIC CURVED ROAD", (380, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (160, 160, 160), 3)

    print(f"[TEST] Frame: {frame.shape[1]}x{frame.shape[0]}")

    # Test _ensure_bgr
    bgr_from_gray = _ensure_bgr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert bgr_from_gray.shape[2] == 3
    print("[OK] _ensure_bgr OK")

    # Test LaneSmootherPoly + frame_id guard (#24)
    sm = LaneSmootherPoly(alpha=0.5)
    c1 = np.array([0.0, 300.0, 200.0])
    c2 = np.array([0.0, 400.0, 100.0])
    r1, _ = sm.update(c1, None, frame_id=0)
    r2, _ = sm.update(c2, None, frame_id=1)
    assert r2 is not None and abs(r2[1] - 350.0) < 1.0, f"EMA fail: {r2}"
    print("[OK] LaneSmootherPoly EMA OK")

    # Test frame_id guard: update cùng frame_id → phải trả về kết quả cũ, không EMA lại
    r3, _ = sm.update(np.array([0.0, 999.0, 999.0]), None, frame_id=1)
    assert r3 is not None and abs(r3[1] - 350.0) < 1.0, f"frame_id guard fail: {r3}"
    print("[OK] LaneSmootherPoly frame_id guard OK (no double-update)")

    # Test Poly fit bậc 2
    fake_lines = [np.array([[400, 720, 450, 500]]), np.array([[420, 600, 460, 400]])]
    coeff = _fit_poly2(fake_lines, 720, 1280)
    print(f"[OK] _fit_poly2: coeff={coeff}")

    # Test detector
    detector = LaneDetector(device="cpu")
    print(f"[TEST] Mode: {'UFLD' if detector._use_ufld else 'Hough+Poly+EMA'}")

    # Test: detect() + get_ego_lane_boundaries() cùng frame_id → smoother chỉ update 1 lần (#24, #27)
    print("\n[TEST] detect() + get_ego_lane_boundaries() cùng frame_id=10 (test no-double-update)...")
    detector.set_frame_id(10)
    lanes_a  = detector.detect(frame, frame_id=10)
    left_b_a, right_b_a = detector.get_ego_lane_boundaries(frame, frame_id=10)
    # Gọi lại lần nữa cùng frame_id — smoother không được thay đổi
    left_b_b, right_b_b = detector.get_ego_lane_boundaries(frame, frame_id=10)
    assert left_b_a[0] == left_b_b[0], "FAIL: Kết quả khác nhau khi gọi 2 lần cùng frame_id"
    print("[OK] No double-update: kết quả nhất quán giữa 2 lần gọi cùng frame")

    print("\n[TEST] detect() x3 frame (test EMA)...")
    for i in range(3):
        detector.set_frame_id(20 + i)
        lanes = detector.detect(frame, frame_id=20 + i)
        print(f"  Frame {20+i}: {len(lanes)} lanes")
        for j, lane in enumerate(lanes):
            xs = [p[0] for p in lane]
            ys = [p[1] for p in lane]
            if xs:
                print(f"    Lane {j}: {len(lane)} pts x=[{min(xs)},{max(xs)}] y=[{min(ys)},{max(ys)}]")

    print("\n[TEST] get_ego_lane_boundaries()...")
    detector.set_frame_id(23)
    left_b, right_b = detector.get_ego_lane_boundaries(frame, frame_id=23)
    print(f"  Left : bottom={left_b[0]}, top={left_b[1]}")
    print(f"  Right: bottom={right_b[0]}, top={right_b[1]}")
    for name, b in [("left", left_b), ("right", right_b)]:
        for pt in b:
            assert pt[0] >= 0, f"FAIL: {name} x={pt[0]} < 0"
    print("[OK] Boundary coords >= 0")

    # Test reset_smoother
    detector.reset_smoother()
    assert detector._hough_cache is None, "FAIL: hough_cache không được xóa sau reset"
    print("[OK] reset_smoother() OK — smoother + cache đã reset")

    # Vẽ kết quả
    vis = frame.copy()
    h, w = vis.shape[:2]
    colors = [(0,255,0),(255,255,0),(0,255,255),(255,0,255)]
    for i, lane in enumerate(lanes):
        c = colors[i % len(colors)]
        for pt in lane:
            cv2.circle(vis, pt, 4, c, -1)
        for k in range(len(lane)-1):
            cv2.line(vis, lane[k], lane[k+1], c, 2)

    cv2.line(vis, left_b[0],  left_b[1],  (0,200,255), 4)
    cv2.line(vis, right_b[0], right_b[1], (0,200,255), 4)

    poly_pts = np.array([left_b[0], left_b[1], right_b[1], right_b[0]], dtype=np.int32)
    overlay  = vis.copy()
    cv2.fillPoly(overlay, [poly_pts], (0, 255, 100))
    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    vp_y = detector.get_vanishing_point(lanes)
    if 0 < vp_y < h:
        cv2.line(vis, (0, vp_y), (w, vp_y), (255,100,100), 2)
        cv2.putText(vis, f"VP y={vp_y}", (10, max(vp_y-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,100), 2)

    cv2.putText(vis, "Hough + Poly deg2 + EMA Smoothing v4", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,220,255), 3)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/lane_test_v4.jpg", vis)
    print("\n[OK] Luu: output/lane_test_v4.jpg")
    print("[OK] Test hoan tat!")