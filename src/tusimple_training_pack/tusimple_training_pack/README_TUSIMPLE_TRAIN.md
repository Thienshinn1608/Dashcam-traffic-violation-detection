# TuSimple fine-tune pack cho đồ án dashcam

Đặt 2 file `.py` vào thư mục `src` của project:

- `split_tusimple_val.py`
- `train_lane_tusimple.py`

## 1) Chia tập validation

```bat
cd /d E:\data_deeplearning\yolov5_train_on_UAVDT-master\yolov5_train_on_UAVDT-master\src
python split_tusimple_val.py --train-set ..\data\TUSimple\train_set --val-ratio 0.1
```

Sinh ra:

```text
..\data\TUSimple\train_set\train_custom.json
..\data\TUSimple\train_set\val_custom.json
```

## 2) Fine-tune UFLD từ tusimple_18.pth

```bat
python train_lane_tusimple.py ^
  --train-set ..\data\TUSimple\train_set ^
  --train-json train_custom.json ^
  --val-json val_custom.json ^
  --pretrained ..\models\tusimple_18.pth ^
  --save ..\models\tusimple_finetuned_best.pth ^
  --epochs 5 ^
  --batch 4
```

Nếu GPU khỏe có thể tăng `--batch 8` và `--epochs 20`.

## 3) Kết hợp vào pipeline hiện tại

Trong `lane_detector.py`, đổi model path:

```python
UFLD_MODEL_PATH_PTH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "models", "tusimple_finetuned_best.pth")
```

Sau đó chạy:

```bat
python main_pipeline.py
```

## Ý nghĩa trong báo cáo

Train lane bằng TuSimple:
- Lane bám đường tốt hơn.
- Xác định làn ổn hơn.
- Hỗ trợ lỗi lấn làn/cut-in.

Train YOLO bằng dataset dashcam:
- Bắt xe xa tốt hơn.
- Bounding box ổn hơn.
- Đo khoảng cách ổn hơn.
