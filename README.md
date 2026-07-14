# 🚦 Traffic Violation Detection System (Dashcam ADAS)

Hệ thống phát hiện vi phạm giao thông thời gian thực từ góc nhìn camera hành trình (dashcam), sử dụng YOLOv11 + BYTETracker + UFLD Lane Detector.

---

## 🎯 Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 🚗 **Phát hiện xe** | YOLOv11 fine-tuned trên bộ dữ liệu dashcam tùy chỉnh |
| 📍 **Theo dõi xe** | BYTETracker đa đối tượng liên tục theo frame |
| 🛣️ **Phát hiện làn đường** | UFLD (Ultra-Fast Lane Detection) với fine-tune trên CurveLanes |
| ⚠️ **Vi phạm khoảng cách** | Ước tính khoảng cách theo camera pinhole, cảnh báo không giữ khoảng cách an toàn |
| 🚧 **Vi phạm lấn làn** | Phát hiện phương tiện ra ngoài làn đường (vượt vạch liền) |
| 📸 **Lưu bằng chứng** | Tự động chụp và lưu ảnh khi vi phạm được xác nhận |
| 📊 **Dashboard web** | Giao diện Streamlit hiển thị real-time + lịch sử vi phạm |

---

## 🏗️ Kiến trúc hệ thống

```
src/
├── app.py                  # Streamlit Web App (entry point)
├── main_pipeline.py        # Logic xử lý pipeline chính + Async Workers
├── lane_detector.py        # Wrapper UFLD lane detection
├── violation_classifier.py # Bộ phân loại & xác nhận vi phạm (FSM)
├── byte_tracker.py         # BYTETracker (multi-object tracking)
├── basetrack.py            # Lớp cơ sở cho tracker
├── kalman_filter.py        # Kalman Filter cho dự đoán quỹ đạo
├── matching.py             # IoU matching cho tracker
├── train_lane_curvelanes.py# Script huấn luyện UFLD trên CurveLanes
├── train_lane_tusimple.py  # Script huấn luyện UFLD trên TuSimple
└── eval_lane_model.py      # Đánh giá mô hình làn đường

models/
├── curvelanes_scratch_best.json  # Config model làn đường
├── accuracy_comparison_curve.png # Biểu đồ so sánh accuracy
└── loss_comparison_curve.png     # Biểu đồ so sánh loss

runs/detect/fast_adas_model/weights/
└── best.pt                 # Trọng số YOLOv11 (fine-tuned)

anh_thong_so/               # Ảnh kết quả train YOLO (metrics, curves)
```

---

## ⚙️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.9+
- CUDA 11.8+ (khuyến nghị, có thể chạy CPU)

### 2. Cài đặt môi trường

```bash
# Clone repository
git clone <repo-url>
cd <repo-name>

# Tạo virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

### 3. Tải model weights

Do giới hạn dung lượng GitHub, các file model phải tải thêm:

| File | Mô tả | Link |
|------|-------|------|
| `models/curvelanes_finetuned_best.pth` | UFLD fine-tune CurveLanes | *(cung cấp riêng)* |
| `models/tusimple_18.pth` | UFLD pre-trained TuSimple | *(cung cấp riêng)* |

> **Ghi chú:** File `runs/detect/fast_adas_model/weights/best.pt` (YOLOv11 ~5.5MB) đã được commit sẵn.

---

## 🚀 Chạy ứng dụng

```bash
# Chạy Streamlit Web App
streamlit run src/app.py
```

Truy cập `http://localhost:8501` trên trình duyệt.

---

## 🧪 Huấn luyện model

### Huấn luyện YOLO (phát hiện xe)

```bash
python train_fast.py
```

### Huấn luyện Lane Detector

```bash
# Trên dataset CurveLanes
python src/train_lane_curvelanes.py

# Trên dataset TuSimple
python src/train_lane_tusimple.py
```

---

## 📊 Kết quả

### YOLO Vehicle Detection (fast_adas_model)

| Metric | Giá trị |
|--------|---------|
| mAP@0.5 | Xem `anh_thong_so/results.png` |
| Precision | Xem `anh_thong_so/BoxP_curve.png` |
| Recall | Xem `anh_thong_so/BoxR_curve.png` |

### Lane Detection (CurveLanes fine-tune vs scratch)

Xem biểu đồ so sánh tại `models/accuracy_comparison_curve.png`.

---

## 📁 Công cụ hỗ trợ

| Script | Mục đích |
|--------|----------|
| `auto_label_test.py` | Tự động gán nhãn dữ liệu test |
| `shrink_dataset.py` | Thu nhỏ dataset để test nhanh |
| `shrink_all_splits.py` | Thu nhỏ tất cả splits train/val/test |
| `convert_to_70_20_10.py` | Chia dataset theo tỷ lệ 70/20/10 |
| `inspect_dataset.py` | Kiểm tra thống kê dataset |
| `detect_video.py` | Chạy inference trên video (CLI mode) |
| `src/plot_comparison.py` | Vẽ biểu đồ so sánh model |
| `src/eval_old_vs_new.py` | So sánh hai phiên bản model |

---

## 📄 License

MIT License
