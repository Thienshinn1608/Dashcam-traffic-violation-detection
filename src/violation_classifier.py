"""
violation_classifier.py
Xác nhận và ghi nhận vi phạm giao thông từ góc nhìn dashcam.
Căn cứ theo Điều 13 Luật GTĐB Việt Nam.

Author  : AI Engineer
Project : Traffic Violation Detection System (Đồ án tốt nghiệp)

FIXES APPLIED (v1):
  #1 Type hints dùng Dict/Tuple từ typing — tương thích Python 3.8/3.9/3.10+
  #2 _pending reset đúng: dùng defaultdict(int), xóa key thay vì set 0
  #3 CSV header dùng list thực (không phải string duy nhất)
  #4 Guard frame is None trước khi copy/imwrite
  #5 Unicode OpenCV: encode text ASCII, fallback tiếng Anh cho cv2.putText
  #6 reset() xóa cả _last_confirmed để tránh cooldown ghost
  #7 Progress bar tqdm khi test (optional, graceful fallback nếu thiếu tqdm)
  #8 get_stats() trả về total tổng hợp cuối dict

FIXES APPLIED (v2):
  #9  FIX CRITICAL: frame=None → vòng lặp confirm vô hạn.
      Nguyên nhân: khi exception xảy ra trong _save_violation_image(),
      _stats không tăng, _last_confirmed không set, _pending không xóa
      → frame sau lại hit confirm threshold, lặp vô hạn đến hết cooldown.
      Fix: tách riêng việc tăng _stats + set _last_confirmed + xóa _pending
      ra ngoài try/except của _save_violation_image, đảm bảo chạy luôn.
      img_path = "N/A" khi save thất bại, CSV vẫn được ghi.
  #10 FIX HIGH: _active_warnings ghost sau reset().
      Nguyên nhân: sau khi confirm, key bị xóa khỏi _pending nhưng
      reset() chỉ tìm key trong _pending để xóa _active_warnings
      → key đã-confirm không bao giờ xóa khỏi _active_warnings.
      Fix: reset() xóa _active_warnings theo track_id trực tiếp,
      độc lập với _pending.
  #11 FIX: Test frame=None trong __main__ sửa lại đúng logic
      (trước đó test báo OK nhưng thực ra đang pass sai vì bug #9).
"""

import os
import csv
import sys
import cv2
import numpy as np
from enum import Enum
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# #7: Progress bar — optional, không crash nếu thiếu tqdm
try:
    from tqdm import tqdm as _tqdm
    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)
except ImportError:
    def _progress(iterable, **kw):
        return iterable


# ==========================================
# ENUM VI PHẠM
# ==========================================
class ViolationType(Enum):
    UNSAFE_DISTANCE    = "UNSAFE_DISTANCE"
    DANGEROUS_CUT_IN   = "DANGEROUS_CUT_IN"
    EGO_LANE_DEPARTURE = "EGO_LANE_DEPARTURE"
    OUTSIDE_LANE       = "OUTSIDE_LANE"


# ==========================================
# CONSTANTS
# ==========================================
CONFIRM_FRAMES  = 8     # Số lần liên tục điều kiện vi phạm phải giữ để xác nhận.
                        # Hạ từ 15→8 để bắt vi phạm nhạy/nhanh hơn (~0.9s ở nhịp detect ~9fps).
                        # Hạ thêm nữa (vd 5) sẽ nhạy hơn nhưng dễ báo nhầm hơn.
COOLDOWN_FRAMES = 60
OUTPUT_DIR      = "output"
VIOLATIONS_DIR  = os.path.join(OUTPUT_DIR, "violations")
CSV_PATH        = os.path.join(OUTPUT_DIR, "violation_log.csv")

VIOLATION_COLORS: Dict[ViolationType, Tuple[int, int, int]] = {
    ViolationType.UNSAFE_DISTANCE:    (0, 0, 255),
    ViolationType.DANGEROUS_CUT_IN:   (0, 100, 255),
    ViolationType.EGO_LANE_DEPARTURE: (0, 165, 255),
    ViolationType.OUTSIDE_LANE:       (255, 0, 180),
}

_VTYPE_ASCII: Dict[ViolationType, str] = {
    ViolationType.UNSAFE_DISTANCE:    "[VIOLATION] UNSAFE DISTANCE",
    ViolationType.DANGEROUS_CUT_IN:   "[VIOLATION] DANGEROUS CUT-IN",
    ViolationType.EGO_LANE_DEPARTURE: "[VIOLATION] EGO LANE DEPARTURE",
    ViolationType.OUTSIDE_LANE:       "[VIOLATION] OUTSIDE LANE",
}

_CSV_HEADER: List[str] = [
    "timestamp", "frame_id", "track_id",
    "violation_type", "distance_m", "image_path",
]


# ==========================================
# CLASS CHÍNH
# ==========================================
class ViolationConfirmer:
    """
    Xác nhận vi phạm theo ngưỡng liên tục (temporal confirmation).

    Luồng hoạt động:
      1. update() được gọi mỗi frame khi phát hiện điều kiện nghi ngờ vi phạm.
      2. Khi điều kiện kéo dài >= CONFIRM_FRAMES → confirmed → lưu ảnh + CSV.
      3. Cùng track_id + cùng loại vi phạm không log lại trong COOLDOWN_FRAMES.
      4. reset() khi track bị mất để tránh rác trong buffer.
    """

    def __init__(self) -> None:
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._init_csv()

        # key: (track_id, ViolationType) → số frame liên tục
        self._pending: Dict[Tuple[int, ViolationType], int] = defaultdict(int)

        # Cooldown: frame_id lần cuối confirmed
        self._last_confirmed: Dict[Tuple[int, ViolationType], int] = {}

        # Thống kê tổng số vi phạm theo loại
        self._stats: Dict[ViolationType, int] = defaultdict(int)

        # Đánh dấu những vi phạm đã ghi nhận vĩnh viễn (mỗi track_id chỉ ghi 1 lần cho 1 lỗi)
        self._ever_confirmed: set = set()

        # Vi phạm đang WARNING (chưa confirmed) — dùng để vẽ HUD
        self._active_warnings: Dict[Tuple[int, ViolationType], dict] = {}

    # ------------------------------------------------------------------
    def _init_csv(self) -> None:
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(_CSV_HEADER)

    # ------------------------------------------------------------------
    def _save_violation_image(
        self,
        frame: np.ndarray,
        violation_type: ViolationType,
        track_id: int,
        frame_id: int,
        metadata: dict,
        timestamp_str: str,
    ) -> str:
        """
        Vẽ overlay thông tin lên ảnh và lưu.
        Raise ValueError nếu frame None/rỗng.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame is None hoac rong, khong the luu anh vi pham")

        img   = frame.copy()
        h, w  = img.shape[:2]
        color = VIOLATION_COLORS.get(violation_type, (0, 0, 255))

        cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 6)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        dist_m = metadata.get("distance_m", 0.0)
        font   = cv2.FONT_HERSHEY_SIMPLEX

        label_lines = [
            _VTYPE_ASCII.get(violation_type, violation_type.value),
            f"Track ID: {track_id}  |  Distance: {dist_m:.1f}m  |  Frame: {frame_id}",
            f"Time: {timestamp_str}",
        ]
        for i, line in enumerate(label_lines):
            cv2.putText(img, line, (10, 22 + i * 22), font, 0.58, color, 2, cv2.LINE_AA)

        bbox = metadata.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img,
                f"ID:{track_id} {dist_m:.1f}m",
                (x1, max(y1 - 10, 10)),
                font, 0.7, color, 2, cv2.LINE_AA,
            )

        safe_ts = timestamp_str.replace(":", "-").replace(" ", "_")
        fname   = f"{violation_type.value}_{track_id}_{safe_ts}.jpg"
        fpath   = os.path.join(VIOLATIONS_DIR, fname)
        cv2.imwrite(fpath, img)
        return fpath

    # ------------------------------------------------------------------
    def _write_csv(
        self,
        timestamp_str: str,
        frame_id: int,
        track_id: int,
        violation_type: ViolationType,
        distance_m: float,
        image_path: str,
    ) -> None:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str, frame_id, track_id,
                violation_type.value, f"{distance_m:.2f}", image_path,
            ])

    # ------------------------------------------------------------------
    def update(
        self,
        track_id: int,
        violation_type: ViolationType,
        frame_id: int,
        frame: Optional[np.ndarray],
        metadata: dict,
    ) -> bool:
        """
        Gọi mỗi frame khi phát hiện điều kiện vi phạm.
        Trả về True nếu vi phạm vừa đạt ngưỡng CONFIRM_FRAMES ở lần gọi này.
        """
        key = (track_id, violation_type)

        # Nếu lỗi này của track_id này đã được confirm trước đây rồi thì bỏ qua luôn
        if key in self._ever_confirmed:
            return False

        last = self._last_confirmed.get(key, -COOLDOWN_FRAMES - 1)
        if frame_id - last < COOLDOWN_FRAMES:
            return False

        self._pending[key] += 1
        self._active_warnings[key] = {
            "frames": self._pending[key],
            "metadata": metadata,
        }

        if self._pending[key] < CONFIRM_FRAMES:
            return False

        # ===================== CONFIRMED =====================
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dist_m        = float(metadata.get("distance_m", 0.0))

        # #9 v2 FIX: Cập nhật state TRƯỚC khi try/except save ảnh
        self._stats[violation_type] += 1
        self._last_confirmed[key]    = frame_id
        self._ever_confirmed.add(key)
        del self._pending[key]
        self._active_warnings.pop(key, None)

        # In console trước khi thực hiện I/O có thể fail
        print(
            f"[VI PHAM XAC NHAN] {violation_type.value} | "
            f"Track {track_id} | {dist_m:.1f}m | Frame {frame_id}"
        )

        # Lưu ảnh — thất bại không ảnh hưởng state đã commit ở trên
        img_path = "N/A"
        try:
            img_path = self._save_violation_image(
                frame, violation_type, track_id, frame_id, metadata, timestamp_str
            )
        except Exception as e:
            print(f"[WARN] Khong luu duoc anh vi pham (track={track_id}): {e}",
                  file=sys.stderr)

        # Ghi CSV — thất bại cũng không rollback state
        try:
            self._write_csv(timestamp_str, frame_id, track_id,
                            violation_type, dist_m, img_path)
        except Exception as e:
            print(f"[WARN] Khong ghi duoc CSV: {e}", file=sys.stderr)
            
        return True

    # ------------------------------------------------------------------
    def reset(self, track_id: int) -> None:
        """
        Xóa toàn bộ buffer của track_id khi track bị mất.

        #10 v2 FIX: Xóa _active_warnings theo track_id trực tiếp,
        độc lập với _pending. Trước đây chỉ xóa _active_warnings khi
        key còn trong _pending → key đã-confirm bị ghost mãi.
        """
        # Xóa _pending (chưa confirmed)
        keys_to_remove_pending = [k for k in list(self._pending.keys()) if k[0] == track_id]
        for k in keys_to_remove_pending:
            del self._pending[k]

        # #10 v2 FIX: Xóa _active_warnings theo track_id trực tiếp
        # (không phụ thuộc vào _pending, bắt cả key đã confirm lẫn đang pending)
        keys_to_remove_warnings = [k for k in list(self._active_warnings.keys()) if k[0] == track_id]
        for k in keys_to_remove_warnings:
            del self._active_warnings[k]

        # #6 FIX v1: Xóa cooldown ghost
        cooldown_keys = [k for k in list(self._last_confirmed.keys()) if k[0] == track_id]
        for k in cooldown_keys:
            del self._last_confirmed[k]

    # ------------------------------------------------------------------
    def get_active(self) -> Dict[Tuple[int, ViolationType], dict]:
        """Trả về dict vi phạm đang WARNING (chưa confirmed)."""
        return dict(self._active_warnings)

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, int]:
        """
        Trả về tổng số vi phạm đã confirmed theo từng loại + tổng cộng.
        """
        result = {vtype.value: self._stats.get(vtype, 0) for vtype in ViolationType}
        result["TOTAL"] = sum(result.values())
        return result


# ==========================================
# TEST ĐỘC LẬP
# ==========================================
if __name__ == "__main__":
    import shutil

    print("=" * 60)
    print("TEST: ViolationConfirmer v2")
    print("=" * 60)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    confirmer = ViolationConfirmer()

    fake_frame = np.full((600, 800, 3), 80, dtype=np.uint8)
    cv2.putText(fake_frame, "FAKE DASHCAM FRAME", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 4)

    FAKE_TRACK_ID = 42
    meta = {"distance_m": 8.5, "bbox": (200, 150, 500, 450)}

    # --- EGO VEHICLE: EGO_LANE_DEPARTURE ---
    print("\n--- EGO VEHICLE: EGO_LANE_DEPARTURE (20 frames) ---")
    for fid in _progress(range(20), desc="EGO_LANE_DEPARTURE", leave=False):
        confirmer.update(
            track_id=0,
            violation_type=ViolationType.EGO_LANE_DEPARTURE,
            frame_id=fid,
            frame=fake_frame,
            metadata={"distance_m": 0.0},
        )
    print(f"Active warnings: {list(confirmer.get_active().keys())}")

    # --- OTHER VEHICLES: UNSAFE_DISTANCE ---
    print("\n--- OTHER VEHICLES: UNSAFE_DISTANCE (30 frames) ---")
    for fid in _progress(range(30), desc="UNSAFE_DISTANCE", leave=False):
        confirmer.update(
            track_id=FAKE_TRACK_ID,
            violation_type=ViolationType.UNSAFE_DISTANCE,
            frame_id=fid,
            frame=fake_frame,
            metadata=meta,
        )

    # #9 v2 FIX TEST: frame=None không crash VÀ không vòng lặp vô hạn
    print("\n--- Test frame=None (khong crash + khong loop vo han) ---")
    try:
        NONE_TRACK_ID = 99
        stats_before_none = confirmer.get_stats()["OUTSIDE_LANE"]

        for fid in range(200, 200 + CONFIRM_FRAMES + 2):
            confirmer.update(
                track_id=NONE_TRACK_ID,
                violation_type=ViolationType.OUTSIDE_LANE,
                frame_id=fid,
                frame=None,
                metadata={"distance_m": 5.0},
            )

        stats_after_none = confirmer.get_stats()["OUTSIDE_LANE"]

        # #9 v2: Vi phạm PHẢI được đếm dù ảnh không lưu được
        assert stats_after_none == stats_before_none + 1, (
            f"FAIL: Vi pham voi frame=None khong duoc dem. "
            f"before={stats_before_none}, after={stats_after_none}"
        )
        print(f"[OK] frame=None: vi pham duoc dem ({stats_after_none}), anh ghi 'N/A' vao CSV")

        # #9 v2: Sau khi confirm, _pending phải được xóa → không loop lại
        pending_after = [k for k in confirmer._pending if k[0] == NONE_TRACK_ID]
        assert len(pending_after) == 0, f"FAIL: _pending khong duoc xoa sau confirm: {pending_after}"
        print("[OK] _pending da xoa sau confirm (khong loop vo han)")

        # Gọi thêm vài frame để chắc chắn cooldown đang hoạt động (không confirm lần 2)
        stats_before_cd = confirmer.get_stats()["OUTSIDE_LANE"]
        for fid in range(200 + CONFIRM_FRAMES + 2, 200 + CONFIRM_FRAMES + 10):
            confirmer.update(
                track_id=NONE_TRACK_ID,
                violation_type=ViolationType.OUTSIDE_LANE,
                frame_id=fid,
                frame=None,
                metadata={"distance_m": 5.0},
            )
        stats_after_cd = confirmer.get_stats()["OUTSIDE_LANE"]
        assert stats_after_cd == stats_before_cd, (
            f"FAIL: Cooldown sau frame=None confirm khong hoat dong: "
            f"before={stats_before_cd}, after={stats_after_cd}"
        )
        print("[OK] Cooldown hoat dong sau khi confirm voi frame=None")

    except AssertionError as e:
        print(f"[FAIL] {e}")
    except Exception as e:
        print(f"[FAIL] Crash khong mong muon: {e}")

    # Kiểm tra cooldown frame thường
    print("\n--- Kiem tra COOLDOWN (khong duoc log lai) ---")
    stats_before = confirmer.get_stats()["UNSAFE_DISTANCE"]
    for fid in range(30, 40):
        confirmer.update(
            track_id=FAKE_TRACK_ID,
            violation_type=ViolationType.UNSAFE_DISTANCE,
            frame_id=fid,
            frame=fake_frame,
            metadata=meta,
        )
    stats_after = confirmer.get_stats()["UNSAFE_DISTANCE"]
    assert stats_before == stats_after, "FAIL: Cooldown khong hoat dong!"
    print(f"[OK] Cooldown hoat dong: count = {stats_after} (khong tang)")

    # #10 v2 FIX TEST: reset() xóa _active_warnings kể cả key đã confirm
    print("\n--- Test reset() xoa _active_warnings (ke ca key da confirm) ---")
    # Tạo 1 track mới, confirm nó (key sẽ bị xóa khỏi _pending sau confirm)
    RESET_TRACK_ID = 77
    for fid in range(300, 300 + CONFIRM_FRAMES + 1):
        confirmer.update(
            track_id=RESET_TRACK_ID,
            violation_type=ViolationType.DANGEROUS_CUT_IN,
            frame_id=fid,
            frame=fake_frame,
            metadata={"distance_m": 5.0, "bbox": (100, 100, 300, 300)},
        )
    # Thêm 1 vi phạm đang pending (chưa confirm)
    for fid in range(400, 405):
        confirmer.update(
            track_id=RESET_TRACK_ID,
            violation_type=ViolationType.UNSAFE_DISTANCE,
            frame_id=fid,
            frame=fake_frame,
            metadata={"distance_m": 7.0},
        )

    confirmer.reset(RESET_TRACK_ID)

    remaining_warnings = [k for k in confirmer._active_warnings if k[0] == RESET_TRACK_ID]
    remaining_pending  = [k for k in confirmer._pending         if k[0] == RESET_TRACK_ID]
    remaining_cooldown = [k for k in confirmer._last_confirmed  if k[0] == RESET_TRACK_ID]

    assert len(remaining_warnings) == 0, f"FAIL: _active_warnings ghost: {remaining_warnings}"
    assert len(remaining_pending)  == 0, f"FAIL: _pending ghost: {remaining_pending}"
    assert len(remaining_cooldown) == 0, f"FAIL: _last_confirmed ghost: {remaining_cooldown}"
    print("[OK] reset() xoa sach _active_warnings, _pending, _last_confirmed")

    # #6 FIX: reset xóa _last_confirmed của FAKE_TRACK_ID
    confirmer.reset(FAKE_TRACK_ID)
    remaining_confirmed = [k for k in confirmer._last_confirmed if k[0] == FAKE_TRACK_ID]
    assert len(remaining_confirmed) == 0, "FAIL: _last_confirmed chua duoc xoa!"
    print(f"[OK] reset() xoa ca _last_confirmed: {remaining_confirmed}")

    remaining_pending = [k for k in confirmer._pending if k[0] == FAKE_TRACK_ID]
    print(f"[OK] reset() xoa _pending: {remaining_pending}")

    # #8: Thống kê với TOTAL
    print("\n--- THONG KE VI PHAM ---")
    stats = confirmer.get_stats()
    for name, count in stats.items():
        marker = "─" * 25 if name != "TOTAL" else "═" * 25
        print(f"  {name:<25}: {count}  {marker if count > 0 else ''}")

    # Verify output files
    print(f"\n--- VERIFY OUTPUT ---")
    print(f"CSV exists   : {os.path.exists(CSV_PATH)}")
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, encoding="utf-8") as f:
            rows = f.readlines()
        print(f"CSV rows     : {len(rows)} (1 header + {len(rows)-1} violations)")
        # Kiểm tra dòng có frame=None phải có img_path = N/A
        none_rows = [r for r in rows[1:] if "OUTSIDE_LANE" in r]
        if none_rows:
            assert "N/A" in none_rows[0], "FAIL: CSV khong ghi N/A cho frame=None"
            print(f"[OK] CSV ghi N/A cho vi pham frame=None: {none_rows[0].strip()}")

    violation_imgs = os.listdir(VIOLATIONS_DIR) if os.path.exists(VIOLATIONS_DIR) else []
    print(f"Anh vi pham  : {len(violation_imgs)} file(s)")
    for fn in violation_imgs[:5]:
        print(f"  -> {fn}")
    if len(violation_imgs) > 5:
        print(f"  ... va {len(violation_imgs) - 5} file khac")

    print("\n[OK] Test hoan tat!")