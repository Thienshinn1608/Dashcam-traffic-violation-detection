import argparse
import json
import os
import sys
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Thêm thư mục hiện tại (src) vào sys.path để import lane_detector.py
sys.path.append(str(Path(__file__).parent))

from lane_detector import (
    UFLDNet,
    UFLD_IMG_W,
    UFLD_IMG_H,
    UFLD_GRIDING_NUM,
    UFLD_CLS_NUM,
    UFLD_NUM_LANES,
    UFLD_ROW_ANCHOR,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CurveLanesUFLDDataset(Dataset):
    """
    Dataset class cho CurveLanes, thiết kế riêng để huấn luyện model UFLD.
    """
    def __init__(self, dataset_dir: str, split: str = "train", limit: int = 0, augment: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.split_dir = self.dataset_dir / split
        self.augment = augment
        
        txt_path = self.split_dir / f"{split}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Khong tim thay file list tai: {txt_path}")

        print(f"Dang doc danh sach anh tu {txt_path}...")
        self.img_paths = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.img_paths.append(line)

        # Hỗ trợ giới hạn số lượng ảnh để train thử nghiệm nhanh
        if limit > 0:
            self.img_paths = self.img_paths[:limit]
            print(f"Gioi han so luong anh {split}: {len(self.img_paths)}")
        else:
            print(f"Tong so luong anh {split}: {len(self.img_paths)}")
        if augment:
            print(f"[Augmentation] BAT - se ap dung ngau nhien cho tung anh train.")

    def __len__(self):
        return len(self.img_paths)

    def _encode_label(self, lines_data, orig_w: int, orig_h: int):
        # Khởi tạo label với giá trị UFLD_GRIDING_NUM (100) tức là không có lane
        label = np.full((UFLD_CLS_NUM, UFLD_NUM_LANES), UFLD_GRIDING_NUM, dtype=np.int64)

        if not lines_data:
            return label

        # 1. Parse các line từ JSON
        parsed_lines = []
        for line in lines_data:
            pts = []
            for pt in line:
                try:
                    x = float(pt["x"])
                    y = float(pt["y"])
                    pts.append((x, y))
                except (ValueError, KeyError):
                    continue
            if len(pts) >= 2:
                parsed_lines.append(pts)

        if not parsed_lines:
            return label

        # 2. Định nghĩa hàm tính x trung bình tương đối để phân loại trái/phải
        def get_line_mean_x(line):
            return float(np.mean([p[0] / orig_w for p in line]))

        # 3. Phân nhóm lane thành trái (mean_x < 0.5) và phải (mean_x >= 0.5)
        left_lines = []
        right_lines = []
        for line in parsed_lines:
            mean_x = get_line_mean_x(line)
            if mean_x < 0.5:
                left_lines.append((mean_x, line))
            else:
                right_lines.append((mean_x, line))

        # Sắp xếp từ trái sang phải
        left_lines.sort(key=lambda x: x[0])
        right_lines.sort(key=lambda x: x[0])

        # 4. Gán các lane vào 4 lane_id của UFLD:
        # 0: left-outer, 1: left-inner, 2: right-inner, 3: right-outer
        assigned_lanes = {}

        # Gán lane trái
        if len(left_lines) == 1:
            assigned_lanes[1] = left_lines[0][1]
        elif len(left_lines) >= 2:
            assigned_lanes[0] = left_lines[-2][1]
            assigned_lanes[1] = left_lines[-1][1]

        # Gán lane phải
        if len(right_lines) == 1:
            assigned_lanes[2] = right_lines[0][1]
        elif len(right_lines) >= 2:
            assigned_lanes[2] = right_lines[0][1]
            assigned_lanes[3] = right_lines[1][1]

        # 5. Nội suy tọa độ x tại các row anchor của UFLD
        for lane_id, pts in assigned_lanes.items():
            pts_sorted = sorted(pts, key=lambda p: p[1])
            ys = [p[1] for p in pts_sorted]
            xs = [p[0] for p in pts_sorted]
            
            y_min, y_max = ys[0], ys[-1]

            for r_idx, y_resized in enumerate(UFLD_ROW_ANCHOR):
                # Ánh xạ y_resized (ở 288) về ảnh gốc
                y_orig = y_resized / UFLD_IMG_H * orig_h

                # Chỉ nội suy nếu y_orig nằm trong dải y của lane
                if y_min <= y_orig <= y_max:
                    x_orig = -1.0
                    for i in range(len(ys) - 1):
                        if ys[i] <= y_orig <= ys[i+1]:
                            if ys[i+1] == ys[i]:
                                x_orig = xs[i]
                            else:
                                t = (y_orig - ys[i]) / (ys[i+1] - ys[i])
                                x_orig = xs[i] + t * (xs[i+1] - xs[i])
                            break

                    if x_orig >= 0:
                        x_resized = x_orig / orig_w * UFLD_IMG_W
                        grid = int(x_resized / UFLD_IMG_W * UFLD_GRIDING_NUM)
                        grid = max(0, min(UFLD_GRIDING_NUM - 1, grid))
                        label[r_idx, lane_id] = grid

        return label

    def _augment(self, img_bgr: np.ndarray, label_np: np.ndarray):
        """
        Áp dụng ngẫu nhiên các biến đổi ảnh và nhãn để chống Overfitting.
        Tất cả biến đổi đều nhất quán giữa ảnh và nhãn tương ứng.
        """
        # 1. Lật ngang ngẫu nhiên (p=0.5) — hoán đổi làn trái/phải
        if np.random.random() < 0.5:
            img_bgr = cv2.flip(img_bgr, 1)
            new_label = np.full_like(label_np, UFLD_GRIDING_NUM)
            # 0↔3 (left-outer↔right-outer), 1↔2 (left-inner↔right-inner)
            for src, dst in [(0, 3), (1, 2), (2, 1), (3, 0)]:
                mask = label_np[:, src] != UFLD_GRIDING_NUM
                new_label[mask, dst] = UFLD_GRIDING_NUM - 1 - label_np[mask, src]
            label_np = new_label

        # 2. Thay đổi độ sáng và tương phản ngẫu nhiên (p=0.6)
        if np.random.random() < 0.6:
            alpha = np.random.uniform(0.65, 1.40)  # tương phản
            beta  = float(np.random.randint(-45, 45))  # độ sáng
            img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 3. Làm mờ Gaussian nhẹ ngẫu nhiên (p=0.3)
        if np.random.random() < 0.3:
            ksize = int(np.random.choice([3, 5]))
            img_bgr = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

        # 4. Thay đổi màu sắc HSV ngẫu nhiên (p=0.4)
        if np.random.random() < 0.4:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + np.random.uniform(-12, 12), 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.75, 1.30), 0, 255)
            img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return img_bgr, label_np

    def __getitem__(self, idx):
        rel_path = self.img_paths[idx]
        img_path = self.split_dir / rel_path
        
        # Đường dẫn nhãn tương ứng (.lines.json)
        label_filename = Path(rel_path).name.replace(".jpg", ".lines.json")
        label_path = self.split_dir / "labels" / label_filename

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Khong tim thay anh tai: {img_path}")

        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (UFLD_IMG_W, UFLD_IMG_H), interpolation=cv2.INTER_LINEAR)

        # Đọc và mã hóa nhãn TRƯỚC augmentation
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            lines_data = label_data.get("Lines", [])
            label_np = self._encode_label(lines_data, orig_w, orig_h)
        else:
            label_np = np.full((UFLD_CLS_NUM, UFLD_NUM_LANES), UFLD_GRIDING_NUM, dtype=np.int64)

        # Áp dụng Data Augmentation (chỉ với tập train)
        if self.augment:
            img_resized, label_np = self._augment(img_resized, label_np)

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_normalized = (img_rgb - MEAN) / STD
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label_np).long()
        return img_tensor, label_tensor


def compute_loss(outputs, labels, criterion, structural_weight=4.0):
    """
    Tính toán hàm Loss kết hợp giữa Classification Loss và Structural Loss (continuity constraint).
    """
    # 1. Classification Loss (Cross Entropy cho từng làn)
    cls_loss = 0.0
    for lane_id in range(UFLD_NUM_LANES):
        pred = outputs[:, :, :, lane_id].permute(0, 2, 1).contiguous()  # [B, 56, 101]
        target = labels[:, :, lane_id].contiguous()                    # [B, 56]
        cls_loss = cls_loss + criterion(pred.view(-1, UFLD_GRIDING_NUM + 1), target.view(-1))
    cls_loss = cls_loss / UFLD_NUM_LANES

    # 2. Structural Loss (Continuity)
    prob = torch.softmax(outputs[:, :UFLD_GRIDING_NUM, :, :], dim=1)  # [B, 100, 56, 4]
    diff = prob[:, :, 1:, :] - prob[:, :, :-1, :]  # [B, 100, 55, 4]
    str_loss = torch.mean(torch.abs(diff))

    total_loss = cls_loss + structural_weight * str_loss
    return total_loss, cls_loss, str_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, structural_weight=4.0):
    """
    Đánh giá model trên tập validation, tính loss trung bình và các chỉ số Precision, Recall, F1
    bằng cách quét ngưỡng confidence để tối ưu hóa Recall & F1-score.
    """
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_str = 0.0
    
    all_probs = []
    all_labels = []
    
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(imgs)
        loss, cls_loss, str_loss = compute_loss(outputs, labels, criterion, structural_weight)
        
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_str += str_loss.item()
        
        prob = torch.softmax(outputs, dim=1).cpu()
        all_probs.append(prob)
        all_labels.append(labels.cpu())
        
    num_batches = max(len(loader), 1)
    avg_loss = total_loss / num_batches
    avg_cls = total_cls / num_batches
    avg_str = total_str / num_batches
    
    all_probs = torch.cat(all_probs, dim=0) # [N, 101, 56, 4]
    all_labels = torch.cat(all_labels, dim=0) # [N, 56, 4]
    
    # 1. Tính toán Exact Accuracy (UFLD cũ)
    pred_exact = all_probs.argmax(dim=1) # [N, 56, 4]
    mask_exact = all_labels != UFLD_GRIDING_NUM
    correct_exact = ((pred_exact == all_labels) & mask_exact).sum().item()
    total_exact = mask_exact.sum().item()
    accuracy = correct_exact / max(total_exact, 1)
    
    # 2. Giải mã bằng xác suất có làn: lane_conf = 1 - bg_conf
    bg_conf = all_probs[:, UFLD_GRIDING_NUM, :, :] # [N, 56, 4]
    lane_conf = 1.0 - bg_conf # [N, 56, 4]
    pred_grid = all_probs[:, :UFLD_GRIDING_NUM, :, :].argmax(dim=1) # [N, 56, 4]
    
    target_lane_mask = all_labels != UFLD_GRIDING_NUM
    close_mask_5 = torch.abs(pred_grid - all_labels) <= 5
    close_mask_10 = torch.abs(pred_grid - all_labels) <= 10
    
    # Quét ngưỡng tối ưu từ 0.05 đến 0.95
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1 = -1.0
    best_thresh = 0.4
    best_p = 0.0
    best_r = 0.0
    
    # Dung sai 10 grid (độ đo lỏng hơn)
    best_f1_10 = -1.0
    best_p_10 = 0.0
    best_r_10 = 0.0
    
    for thresh in thresholds:
        pred_lane_mask = lane_conf > thresh
        
        # 5-grid tolerance
        tp_5 = (pred_lane_mask & target_lane_mask & close_mask_5).sum().item()
        fp_5 = (pred_lane_mask & (~target_lane_mask | ~close_mask_5)).sum().item()
        fn_5 = (target_lane_mask & (~pred_lane_mask | ~close_mask_5)).sum().item()
        
        p_5 = tp_5 / max(tp_5 + fp_5, 1)
        r_5 = tp_5 / max(tp_5 + fn_5, 1)
        f1_5 = 2 * p_5 * r_5 / max(p_5 + r_5, 1e-6)
        
        if f1_5 > best_f1:
            best_f1 = f1_5
            best_thresh = thresh
            best_p = p_5
            best_r = r_5
            
        # 10-grid tolerance
        tp_10 = (pred_lane_mask & target_lane_mask & close_mask_10).sum().item()
        fp_10 = (pred_lane_mask & (~target_lane_mask | ~close_mask_10)).sum().item()
        fn_10 = (target_lane_mask & (~pred_lane_mask | ~close_mask_10)).sum().item()
        
        p_10 = tp_10 / max(tp_10 + fp_10, 1)
        r_10 = tp_10 / max(tp_10 + fn_10, 1)
        f1_10 = 2 * p_10 * r_10 / max(p_10 + r_10, 1e-6)
        
        if f1_10 > best_f1_10:
            best_f1_10 = f1_10
            best_p_10 = p_10
            best_r_10 = r_10

    # Tính ma trận nhầm lẫn 5x5 tại ngưỡng tối ưu best_thresh
    all_cm = np.zeros((5, 5), dtype=np.int64)
    pred_lane_mask_best = lane_conf > best_thresh
    
    for lane_id in range(UFLD_NUM_LANES):
        act_l = all_labels[:, :, lane_id]
        pred_l_g = pred_grid[:, :, lane_id]
        pred_l_ok = pred_lane_mask_best[:, :, lane_id]
        
        act_mask = act_l != UFLD_GRIDING_NUM
        
        tp_l = act_mask & pred_l_ok & (torch.abs(pred_l_g - act_l) <= 5)
        fn_l = act_mask & (~pred_l_ok | (torch.abs(pred_l_g - act_l) > 5))
        fp_l = (~act_mask) & pred_l_ok
        tn_l = (~act_mask) & (~pred_l_ok)
        
        all_cm[lane_id, lane_id] += tp_l.sum().item()
        all_cm[lane_id, 4] += fn_l.sum().item()
        all_cm[4, lane_id] += fp_l.sum().item()
        all_cm[4, 4] += tn_l.sum().item()
        
    print(f"[Metrics Dung Sai 10 Grid] Precision: {best_p_10:.4f} | Recall: {best_r_10:.4f} | F1: {best_f1_10:.4f}")
    
    return avg_loss, avg_cls, avg_str, accuracy, best_p, best_r, best_f1, best_thresh, all_cm


@torch.no_grad()
def evaluate_curves_data(model, loader, device):
    """
    Quét qua các ngưỡng confidence từ 0.0 đến 0.99 để thu thập dữ liệu vẽ các đường cong F1, P, R, PR.
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(imgs)
        prob = torch.softmax(outputs, dim=1).cpu()
        all_probs.append(prob)
        all_labels.append(labels.cpu())
        
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    bg_conf = all_probs[:, UFLD_GRIDING_NUM, :, :]
    lane_conf = 1.0 - bg_conf
    pred_grid = all_probs[:, :UFLD_GRIDING_NUM, :, :].argmax(dim=1)
    
    thresholds = np.linspace(0.0, 0.99, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    target_lane_mask = all_labels != UFLD_GRIDING_NUM
    close_mask = torch.abs(pred_grid - all_labels) <= 5
    
    for thresh in thresholds:
        pred_lane_mask = lane_conf > thresh
        
        tp = (pred_lane_mask & target_lane_mask & close_mask).sum().item()
        fp = (pred_lane_mask & (~target_lane_mask | ~close_mask)).sum().item()
        fn = (target_lane_mask & (~pred_lane_mask | ~close_mask)).sum().item()
        
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-6)
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        
    return thresholds, np.array(precisions), np.array(recalls), np.array(f1_scores)


def generate_preprocessing_report(dataset_dir: Path, split: str = "train", limit: int = 5000):
    """
    Phân tích và ghi lại thông tin tiền xử lý dữ liệu và thống kê nhãn ra file txt.
    """
    split_dir = dataset_dir / split
    txt_path = split_dir / f"{split}.txt"
    if not txt_path.exists():
        return f"Không tìm thấy file danh sách tại: {txt_path}"
        
    img_paths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                img_paths.append(line)
                
    total_images = len(img_paths)
    sample_paths = img_paths[:limit]
    
    lane_counts = []
    points_per_lane = []
    img_widths = []
    img_heights = []
    
    print(f"Dang quet {len(sample_paths)} nhan mau de tao report tien xu ly...")
    for rel_path in sample_paths:
        label_filename = Path(rel_path).name.replace(".jpg", ".lines.json")
        label_path = split_dir / "labels" / label_filename
        
        img_path = split_dir / rel_path
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            img_widths.append(w)
            img_heights.append(h)
            
        if label_path.exists():
            try:
                with open(label_path, "r", encoding="utf-8") as lf:
                    label_data = json.load(lf)
                lines = label_data.get("Lines", [])
                lane_counts.append(len(lines))
                for line in lines:
                    points_per_lane.append(len(line))
            except Exception:
                pass
                
    avg_lanes = np.mean(lane_counts) if lane_counts else 0
    max_lanes = np.max(lane_counts) if lane_counts else 0
    min_lanes = np.min(lane_counts) if lane_counts else 0
    avg_pts = np.mean(points_per_lane) if points_per_lane else 0
    mean_w = np.mean(img_widths) if img_widths else 0
    mean_h = np.mean(img_heights) if img_heights else 0
    
    dist = {}
    for c in lane_counts:
        dist[c] = dist.get(c, 0) + 1
    dist_str = ", ".join([f"{k} lanes: {v} anh" for k, v in sorted(dist.items())])
    
    report_content = f"""==================================================
CURVELANES DATASET PREPROCESSING & STATISTICS REPORT
==================================================
- Target Split Evaluated : {split}
- Total Images in Split  : {total_images}
- Sampled Images for Stats: {len(sample_paths)}
- Average Original Resolution: {mean_w:.1f} x {mean_h:.1f}
- Normalization Mean     : [0.485, 0.456, 0.406]
- Normalization Std      : [0.229, 0.224, 0.225]

LANE DISTRIBUTION STATS:
- Average Lanes per Image: {avg_lanes:.2f}
- Min Lanes in an Image  : {min_lanes}
- Max Lanes in an Image  : {max_lanes}
- Distribution Details   : {dist_str}
- Average Points per Lane: {avg_pts:.2f}

UFLD NETWORK TARGET CONFIGURATION:
- Network Input Width (W): {UFLD_IMG_W} (800)
- Network Input Height (H): {UFLD_IMG_H} (288)
- Griding Columns (Cls)  : {UFLD_GRIDING_NUM} (100)
- Max Supported Lanes (Ch): {UFLD_NUM_LANES} (4)
- Row Anchors (Rows)     : {UFLD_CLS_NUM} anchors (56)
- Row Anchor pixel list  : {UFLD_ROW_ANCHOR}
=================================================="""

    return report_content


def plot_confusion_matrix(cm, classes, output_path, normalize=False, title='Confusion Matrix'):
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) dạng heatmap giống YOLO.
    """
    plt.figure(figsize=(8, 6.5))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm.astype('float') / row_sums, 0.0)
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'
        
    plt.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    
    thresh = cm_plot.max() / 2.
    for i, j in np.ndindex(cm_plot.shape):
        val = cm_plot[i, j]
        color = "white" if val > thresh else "black"
        if normalize:
            text = f"{val:.2%}" if val > 0 else "0.0%"
        else:
            text = f"{val}"
        plt.text(j, i, text,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color=color, fontsize=12, fontweight='bold')
                 
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metrics_curves(thresholds, precisions, recalls, f1_scores, output_dir):
    """
    Vẽ các sơ đồ BoxF1_curve.png, BoxP_curve.png, BoxPR_curve.png, BoxR_curve.png giống YOLO.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_p = precisions[best_idx]
    best_r = recalls[best_idx]
    
    # 1. BoxF1_curve.png
    plt.figure(figsize=(8, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.plot(thresholds, f1_scores, color="#1F77B4", linewidth=3, label=f"all classes {best_f1:.2f} at {best_thresh:.2f}")
    plt.title("F1-Confidence Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("F1", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower center", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "BoxF1_curve.png", dpi=300)
    plt.close()
    
    # 2. BoxP_curve.png
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, color="#2CA02C", linewidth=3, label=f"all classes {best_p:.2f} at {best_thresh:.2f}")
    plt.title("Precision-Confidence Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "BoxP_curve.png", dpi=300)
    plt.close()
    
    # 3. BoxR_curve.png
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, recalls, color="#D62728", linewidth=3, label=f"all classes {best_r:.2f} at {best_thresh:.2f}")
    plt.title("Recall-Confidence Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower left", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "BoxR_curve.png", dpi=300)
    plt.close()
    
    # 4. BoxPR_curve.png
    plt.figure(figsize=(8, 6))
    sorted_indices = np.argsort(recalls)
    plt.plot(recalls[sorted_indices], precisions[sorted_indices], color="#9467BD", linewidth=3, label="all classes")
    plt.title("Precision-Recall Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower left", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "BoxPR_curve.png", dpi=300)
    plt.close()


def plot_results_png(csv_path, output_path):
    """
    Vẽ file results.png chứa các biểu đồ mất mát và độ chính xác qua các epoch giống YOLO.
    """
    epochs = []
    train_loss = []
    train_cls = []
    train_str = []
    val_loss = []
    val_cls = []
    val_str = []
    val_acc = []
    val_p = []
    val_r = []
    val_f1 = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            train_cls.append(float(row['train_cls_loss']))
            train_str.append(float(row['train_str_loss']))
            val_loss.append(float(row['val_loss']))
            val_cls.append(float(row['val_cls_loss']))
            val_str.append(float(row['val_str_loss']))
            val_acc.append(float(row['val_accuracy']))
            val_p.append(float(row['val_precision']))
            val_r.append(float(row['val_recall']))
            val_f1.append(float(row['val_f1']))
            
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Biểu đồ dòng 1
    axes[0, 0].plot(epochs, train_loss, 'o-', color='#1F77B4', linewidth=2)
    axes[0, 0].set_title("train/loss", fontsize=12, fontweight='bold')
    
    axes[0, 1].plot(epochs, train_cls, 'o-', color='#FF7F0E', linewidth=2)
    axes[0, 1].set_title("train/cls_loss", fontsize=12, fontweight='bold')
    
    axes[0, 2].plot(epochs, train_str, 'o-', color='#2CA02C', linewidth=2)
    axes[0, 2].set_title("train/str_loss", fontsize=12, fontweight='bold')
    
    axes[0, 3].plot(epochs, val_loss, 'o-', color='#D62728', linewidth=2)
    axes[0, 3].set_title("val/loss", fontsize=12, fontweight='bold')
    
    axes[0, 4].plot(epochs, val_cls, 'o-', color='#9467BD', linewidth=2)
    axes[0, 4].set_title("val/cls_loss", fontsize=12, fontweight='bold')
    
    # Biểu đồ dòng 2
    axes[1, 0].plot(epochs, val_p, 'o-', color='#8C564B', linewidth=2)
    axes[1, 0].set_title("metrics/precision", fontsize=12, fontweight='bold')
    
    axes[1, 1].plot(epochs, val_r, 'o-', color='#E377C2', linewidth=2)
    axes[1, 1].set_title("metrics/recall", fontsize=12, fontweight='bold')
    
    axes[1, 2].plot(epochs, val_f1, 'o-', color='#7F7F7F', linewidth=2)
    axes[1, 2].set_title("metrics/F1", fontsize=12, fontweight='bold')
    
    axes[1, 3].plot(epochs, val_acc, 'o-', color='#BCBD22', linewidth=2)
    axes[1, 3].set_title("metrics/accuracy", fontsize=12, fontweight='bold')
    
    axes[1, 4].plot(epochs, val_str, 'o-', color='#17BECF', linewidth=2)
    axes[1, 4].set_title("val/str_loss", fontsize=12, fontweight='bold')
    
    for ax in axes.flat:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_labels_jpg(dataset_dir, output_path, limit=2000):
    """
    Tạo biểu đồ labels.jpg chứa phân phối và thống kê về nhãn của dataset.
    """
    split_dir = Path(dataset_dir) / "valid"
    txt_path = split_dir / "valid.txt"
    if not txt_path.exists():
        return
        
    img_paths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                img_paths.append(line)
                
    sample_paths = img_paths[:limit]
    
    lane_point_counts = [0, 0, 0, 0] # Cho 4 làn
    x_coords = []
    y_coords = []
    
    for rel_path in sample_paths:
        label_filename = Path(rel_path).name.replace(".jpg", ".lines.json")
        label_path = split_dir / "labels" / label_filename
        
        if label_path.exists():
            try:
                with open(label_path, "r", encoding="utf-8") as lf:
                    label_data = json.load(lf)
                lines = label_data.get("Lines", [])
                
                for idx, line in enumerate(lines[:4]):
                    lane_point_counts[idx] += len(line)
                    for pt in line:
                        x = float(pt["x"])
                        y = float(pt["y"])
                        x_coords.append(x)
                        y_coords.append(y)
            except Exception:
                pass
                
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.style.use("seaborn-v0_8-whitegrid")
    
    lanes = ['L-Outer', 'L-Inner', 'R-Inner', 'R-Outer']
    axes[0, 0].bar(lanes, lane_point_counts, color='#1F77B4', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("Points per Lane Distribution", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Count")
    
    axes[0, 1].hist(x_coords, bins=50, color='#2CA02C', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("X Coordinate Distribution", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("X (pixels)")
    
    axes[1, 0].hist(y_coords, bins=50, color='#FF7F0E', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Y Coordinate Distribution (Row Anchors)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Y (pixels)")
    
    if len(x_coords) > 5000:
        indices = np.random.choice(len(x_coords), 5000, replace=False)
        xs = [x_coords[i] for i in indices]
        ys = [y_coords[i] for i in indices]
    else:
        xs, ys = x_coords, y_coords
        
    axes[1, 1].scatter(xs, ys, s=1, color='#9467BD', alpha=0.3)
    axes[1, 1].set_title("Lane Points Spatial Scatter (X vs Y)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("X (pixels)")
    axes[1, 1].set_ylabel("Y (pixels)")
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_pretrained_weights(model, path):
    if not path or not os.path.exists(path):
        print("Khong su dung hoac khong tim thay file pretrained. Bat dau train tu dau.")
        return
    
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
                
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
        print(f"Da load xong file pretrained: {path}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"Loi khi load pretrained weights: {e}")


def main():
    parser = argparse.ArgumentParser(description="Huon luyen UFLD lane model tren dataset CurveLanes.")
    parser.add_argument("--dataset-dir", default="archive/Curvelanes", help="Duong dan den thu muc dataset Curvelanes")
    parser.add_argument("--pretrained", default="", help="Duong dan den file pretrained .pth cua UFLD (bo trong de train tu dau/ImageNet)")
    parser.add_argument("--imagenet", type=bool, default=True, help="Load pretrained ImageNet resnet18 cho backbone de hoi tu nhanh hon")
    parser.add_argument("--save", default="models/curvelanes_finetuned_best.pth", help="Duong dan de luu model tot nhat")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=0, help="Gioi han so anh train (0 = khong gioi han)")
    parser.add_argument("--limit-val", type=int, default=0, help="Gioi han so anh validation (0 = khong gioi han)")
    parser.add_argument("--str-weight", type=float, default=4.0, help="Trong so cho Structural Loss (continuity)")
    parser.add_argument("--report-txt", default="curvelanes_preprocessing_info.txt", help="File luu thong tin tien xu ly")
    parser.add_argument("--results-csv", default="curvelanes_training_results.csv", help="File CSV luu lich su training")
    parser.add_argument("--charts-dir", default="anh_thong_so_lane_model", help="Thu muc luu tru cac bieu do ket qua cua model Lane")
    parser.add_argument("--compare-tusimple", type=bool, default=True, help="So sanh kết quả mô hình mới với mô hình TuSimple cũ sau khi train xong")
    parser.add_argument("--tusimple-path", default="models/tusimple_18.pth", help="Đường dẫn đến mô hình TuSimple cũ để so sánh")
    parser.add_argument("--aug", action="store_true", default=True, help="Bat Data Augmentation cho tap train (chong Overfitting)")
    parser.add_argument("--no-aug", dest="aug", action="store_false", help="Tat Data Augmentation")
    parser.add_argument("--early-stop", type=int, default=10, help="Dung train som neu Val F1 khong tang trong N epoch lien tiep (0 = tat)")
    parser.add_argument("--label-smooth", type=float, default=0.1, help="He so label smoothing cho CrossEntropyLoss (0 = tat)")
    args = parser.parse_args()

    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sinh báo cáo tiền xử lý và thống kê nhãn ra file txt
    print("=== [1] Dang sinh report tien xu ly du lieu ===")
    dataset_path = Path(args.dataset_dir)
    report_content = generate_preprocessing_report(dataset_path, split="train", limit=2000)
    with open(args.report_txt, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"==> Da luu thong tin tien xu ly tai: {args.report_txt}")

    # 2. Tạo biểu đồ phân bố nhãn ban đầu (labels.jpg)
    print("=== [2] Dang ve bieu do phan bo nhan labels.jpg ===")
    plot_labels_jpg(dataset_path, charts_dir / "labels.jpg", limit=2000)
    print(f"==> Da ve xong labels.jpg tai: {charts_dir / 'labels.jpg'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device dang dung: {device}")

    # Khởi tạo dataset & dataloader
    print("=== [3] Khoi tao dataloader ===")
    train_ds = CurveLanesUFLDDataset(args.dataset_dir, split="train", limit=args.limit_train, augment=args.aug)
    val_ds = CurveLanesUFLDDataset(args.dataset_dir, split="valid", limit=args.limit_val, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0)
    )

    # Khởi tạo UFLDNet
    model = UFLDNet().to(device)
    
    # Nạp weights
    if args.pretrained:
        load_pretrained_weights(model, args.pretrained)
    elif args.imagenet:
        # Load standard ImageNet weights cho backbone của UFLDNet
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Dang load pretrained weights tu ImageNet...")
        except Exception:
            try:
                from torchvision.models import resnet18
                resnet = resnet18(pretrained=True)
                print("Dang load pretrained weights tu ImageNet (legacy API)...")
            except Exception:
                resnet = None
                print("Khong the load ImageNet weights. Bat dau huan luyen tu dau (random initialization).")
                
        if resnet is not None:
            model.model[0].load_state_dict(resnet.conv1.state_dict())
            model.model[1].load_state_dict(resnet.bn1.state_dict())
            model.model[4].load_state_dict(resnet.layer1.state_dict())
            model.model[5].load_state_dict(resnet.layer2.state_dict())
            model.model[6].load_state_dict(resnet.layer3.state_dict())
            model.model[7].load_state_dict(resnet.layer4.state_dict())
            print("==> Da load thanh cong ImageNet weights cho ResNet18 backbone.")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    print(f"[Config] LR={args.lr} | WeightDecay=5e-4 | LabelSmooth={args.label_smooth} | Aug={args.aug} | EarlyStop={args.early_stop}")

    # Khởi tạo file CSV ghi kết quả từng epoch
    csv_fields = [
        "epoch", "train_loss", "train_cls_loss", "train_str_loss", 
        "val_loss", "val_cls_loss", "val_str_loss", 
        "val_accuracy", "val_precision", "val_recall", "val_f1", "best_threshold"
    ]
    with open(args.results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)

    best_f1 = -1.0
    no_improve_epochs = 0   # đếm epoch liên tiếp không cải thiện F1
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== [4] Bat dau huan luyen {args.epochs} epochs. ===")
    
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_str = 0.0
        
        start_time = time.time()
        
        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(imgs)
                loss, cls_loss, str_loss = compute_loss(outputs, labels, criterion, args.str_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_str += str_loss.item()

            if step % 100 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Str: {str_loss.item():.4f})")

        scheduler.step()
        epoch_time = time.time() - start_time
        
        train_avg_loss = total_loss / max(len(train_loader), 1)
        train_avg_cls = total_cls / max(len(train_loader), 1)
        train_avg_str = total_str / max(len(train_loader), 1)
        
        # Đánh giá trên tập val
        val_avg_loss, val_avg_cls, val_avg_str, val_acc, val_p, val_r, val_f1, best_thresh, epoch_cm = evaluate(
            model, val_loader, criterion, device, args.str_weight
        )
        
        print(f"Epoch {epoch} hoan thanh trong {epoch_time:.1f}s | "
              f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Best Thresh: {best_thresh:.2f} | Val P: {val_p:.4f} | Val R: {val_r:.4f} | Val F1: {val_f1:.4f}")

        # Ghi nhận kết quả vào file CSV
        with open(args.results_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_avg_loss, train_avg_cls, train_avg_str,
                val_avg_loss, val_avg_cls, val_avg_str,
                val_acc, val_p, val_r, val_f1, best_thresh
            ])

        # Lưu checkpoint tốt nhất theo F1-score
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve_epochs = 0
            torch.save({"model": model.state_dict()}, save_path)
            print(f"==> Da luu model tot nhat voi F1-Score: {val_f1:.4f} tai {save_path}")
        else:
            no_improve_epochs += 1
            print(f"[EarlyStop] Khong cai thien: {no_improve_epochs}/{args.early_stop} epoch. Best F1 hien tai: {best_f1:.4f}")
            if args.early_stop > 0 and no_improve_epochs >= args.early_stop:
                print(f"\n=== EARLY STOPPING: Val F1 khong tang trong {args.early_stop} epoch lien tiep. Dung tai Epoch {epoch}. ===")
                break

    print("=== [5] Huan luyen hoan thanh! Dang thuc hien cac danh gia va ve bieu do cuoi cung... ===")
    
    if save_path.exists():
        best_state = torch.load(save_path, map_location=device)
        model.load_state_dict(best_state["model"])
        print("Da load model tot nhat de danh gia cuoi ky.")

    # 3. Vẽ results.png từ file CSV
    print("Vẽ kết quả huấn luyện (results.png)...")
    plot_results_png(args.results_csv, charts_dir / "results.png")
    print(f"==> Da luu results.png tai: {charts_dir / 'results.png'}")

    # 4. Quét confidence và vẽ các đường cong BoxF1, BoxP, BoxR, BoxPR
    print("Sinh dữ liệu và vẽ các đường cong F1, P, R, PR...")
    thresholds, precisions, recalls, f1_scores = evaluate_curves_data(model, val_loader, device)
    plot_metrics_curves(thresholds, precisions, recalls, f1_scores, charts_dir)
    print(f"==> Da luu xong cac bieu do duong cong tai: {charts_dir}")

    # 5. Vẽ ma trận nhầm lẫn confusion_matrix.png & confusion_matrix_normalized.png
    print("Vẽ ma trận nhầm lẫn (Confusion Matrix)...")
    _, _, _, _, _, _, _, _, final_cm = evaluate(model, val_loader, criterion, device, args.str_weight)
    classes = ['L-Outer', 'L-Inner', 'R-Inner', 'R-Outer', 'Background']
    plot_confusion_matrix(final_cm, classes, charts_dir / "confusion_matrix.png", normalize=False, title="Confusion Matrix")
    plot_confusion_matrix(final_cm, classes, charts_dir / "confusion_matrix_normalized.png", normalize=True, title="Normalized Confusion Matrix")
    print(f"==> Da luu ma tran nham lan tai: {charts_dir}")

    # 6. So sánh với model TuSimple cũ sau khi train xong
    if args.compare_tusimple and os.path.exists(args.tusimple_path):
        print("\n=== [6] Dang tien hanh danh gia so sanh voi model TuSimple cu ===")
        # Khởi động lại model cũ
        model_old = UFLDNet().to(device)
        load_pretrained_weights(model_old, args.tusimple_path)
        
        # Đánh giá model cũ
        print("Danh gia model TuSimple cu...")
        loss_old, cls_old, str_old, acc_old, p_old, r_old, f1_old, thresh_old, _ = evaluate(
            model_old, val_loader, criterion, device, args.str_weight
        )
        
        # Đánh giá model mới tốt nhất
        print("Danh gia model Curvelanes moi...")
        loss_new, cls_new, str_new, acc_new, p_new, r_new, f1_new, thresh_new, _ = evaluate(
            model, val_loader, criterion, device, args.str_weight
        )
        
        # In bảng so sánh
        print("\n================== SO SANH MODEL CU (TUSIMPLE) VS MOI (CURVELANES) ==================")
        print(f"{'Metric':<25} | {'Model Cu (TuSimple)':<22} | {'Model Moi (CurveLanes)':<22}")
        print("-" * 75)
        print(f"{'Path':<25} | {args.tusimple_path:<22} | {args.save:<22}")
        print(f"{'Val Total Loss':<25} | {loss_old:<22.4f} | {loss_new:<22.4f}")
        print(f"{'Val Cls Loss':<25} | {cls_old:<22.4f} | {cls_new:<22.4f}")
        print(f"{'Val Str Loss':<25} | {str_old:<22.4f} | {str_new:<22.4f}")
        print(f"{'Accuracy (Strict)':<25} | {acc_old*100:<21.2f}% | {acc_new*100:<21.2f}%")
        print(f"{'Precision (Tolerance)':<25} | {p_old*100:<21.2f}% | {p_new*100:<21.2f}%")
        print(f"{'Recall (Tolerance)':<25} | {r_old*100:<21.2f}% | {r_new*100:<21.2f}%")
        print(f"{'F1-Score (Tolerance)':<25} | {f1_old*100:<21.2f}% | {f1_new*100:<21.2f}%")
        print(f"{'Optimal Threshold':<25} | {thresh_old:<22.2f} | {thresh_new:<22.2f}")
        print("=====================================================================================")
        
        # Ghi file txt kết quả so sánh
        compare_txt_path = charts_dir / "so_sanh_ket_qua.txt"
        with open(compare_txt_path, "w", encoding="utf-8") as f:
            f.write("================== SO SANH MODEL CU (TUSIMPLE) VS MOI (CURVELANES) ==================\n")
            f.write(f"{'Metric':<25} | {'Model Cu (TuSimple)':<22} | {'Model Moi (CurveLanes)':<22}\n")
            f.write("-" * 75 + "\n")
            f.write(f"{'Path':<25} | {args.tusimple_path:<22} | {args.save:<22}\n")
            f.write(f"{'Val Total Loss':<25} | {loss_old:<22.4f} | {loss_new:<22.4f}\n")
            f.write(f"{'Val Cls Loss':<25} | {cls_old:<22.4f} | {cls_new:<22.4f}\n")
            f.write(f"{'Val Str Loss':<25} | {str_old:<22.4f} | {str_new:<22.4f}\n")
            f.write(f"{'Accuracy (Strict)':<25} | {acc_old*100:<21.2f}% | {acc_new*100:<21.2f}%\n")
            f.write(f"{'Precision (Tolerance)':<25} | {p_old*100:<21.2f}% | {p_new*100:<21.2f}%\n")
            f.write(f"{'Recall (Tolerance)':<25} | {r_old*100:<21.2f}% | {r_new*100:<21.2f}%\n")
            f.write(f"{'F1-Score (Tolerance)':<25} | {f1_old*100:<21.2f}% | {f1_new*100:<21.2f}%\n")
            f.write(f"{'Optimal Threshold':<25} | {thresh_old:<22.2f} | {thresh_new:<22.2f}\n")
            f.write("=====================================================================================\n")
        print(f"==> Da ghi ket qua so sanh vao: {compare_txt_path}")

    print("=== HOAN THANH TOAN BO TIEN TRINH TRAIN VA DANH GIA ===")
    print(f"- File report: {args.report_txt}")
    print(f"- File CSV: {args.results_csv}")
    print(f"- Cac bieu do da duoc ghi nhan day du tai: {charts_dir}/")


if __name__ == "__main__":
    main()
