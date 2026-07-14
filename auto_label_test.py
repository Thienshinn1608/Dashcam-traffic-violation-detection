import os
import sys
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Thêm thư mục src vào sys.path để import
sys.path.append(str(Path(__file__).parent / "src"))

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

def load_pretrained_weights(model, weight_path):
    print(f"Loading weights from {weight_path}...")
    state_dict = torch.load(weight_path, map_location="cpu")
    # Hỗ trợ cả weights có và không có prefix "model." hoặc "module."
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)
    print("Weights loaded successfully!")

@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-labeling cho tap Test cua Curvelanes bang model da train.")
    parser.add_argument("--model-path", default="models/curvelanes_finetuned_best.pth", help="Duong dan den file model checkpoint .pth")
    parser.add_argument("--dataset-dir", default="archive/Curvelanes", help="Duong dan den thu muc dataset Curvelanes")
    parser.add_argument("--threshold", type=float, default=0.40, help="Confidence threshold de gán nhãn lane (lane_conf = 1 - bg_conf)")
    args = parser.parse_args()

    # Thư mục chứa test set
    base_dir = Path(args.dataset_dir)
    test_img_dir = base_dir / "test" / "images"
    test_lbl_dir = base_dir / "test" / "labels"

    if not test_img_dir.exists():
        print(f"Error: Thu muc anh test khong ton tai tai {test_img_dir}!")
        return

    # Tạo thư mục nhãn nếu chưa có
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Đọc danh sách ảnh
    img_files = list(test_img_dir.glob("*.jpg"))
    if len(img_files) == 0:
        print("Khong tim thay anh .jpg nao trong thu muc test/images/")
        return

    print(f"Tim thay {len(img_files)} anh trong tap Test.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Su dung thiet bi: {device} de chay inference.")

    # Load model
    model = UFLDNet().to(device)
    try:
        load_pretrained_weights(model, args.model_path)
    except Exception as e:
        print(f"Error khi load model weights: {e}")
        print("Vui long dam bao ban da train model va co checkpoint tai: " + args.model_path)
        print("Hoac truyen duong dan den model qua tham so --model-path.")
        return

    model.eval()

    print("\n=== Bat dau gán nhãn tu dong ===")
    labeled_count = 0

    for img_path in tqdm(img_files, desc="Auto-labeling"):
        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        orig_h, orig_w = img.shape[:2]

        # Tiền xử lý
        img_resized = cv2.resize(img, (UFLD_IMG_W, UFLD_IMG_H), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_normalized = (img_rgb - MEAN) / STD
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # Forward
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu() # [1, 101, 56, 4]

        # Giải mã tọa độ nhãn
        bg_conf = probs[0, UFLD_GRIDING_NUM, :, :] # [56, 4]
        lane_conf = 1.0 - bg_conf # [56, 4]
        pred_grid = probs[0, :UFLD_GRIDING_NUM, :, :].argmax(dim=0) # [56, 4]

        lines = []

        # CurveLanes co tối đa 4 lane (lane_id: 0, 1, 2, 3)
        for lane_id in range(UFLD_NUM_LANES):
            pts = []
            for r_idx, y_resized in enumerate(UFLD_ROW_ANCHOR):
                conf = lane_conf[r_idx, lane_id].item()
                grid_idx = pred_grid[r_idx, lane_id].item()

                # Kiểm tra xem lane co ton tai tai row anchor nay khong
                if conf > args.threshold and grid_idx < UFLD_GRIDING_NUM:
                    # Map ngược tọa độ x, y về ảnh gốc
                    x_orig = (grid_idx + 0.5) / UFLD_GRIDING_NUM * orig_w
                    y_orig = y_resized / UFLD_IMG_H * orig_h
                    
                    # Định dạng x, y la string lam tron 2 chu so thap phan dung chuan CurveLanes
                    pts.append({
                        "x": f"{x_orig:.2f}",
                        "y": f"{y_orig:.2f}"
                    })
            
            # Neu lane co it nhat 2 diem thi moi coi la lane hop le
            if len(pts) >= 2:
                lines.append(pts)

        # Lưu nhãn thành file .lines.json
        label_filename = img_path.name.replace(".jpg", ".lines.json")
        label_path = test_lbl_dir / label_filename

        label_data = {"Lines": lines}
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=4)
        
        labeled_count += 1

    print(f"\n==> Hoan thanh! Da gan nhan tu dong va luu {labeled_count} file JSON vao {test_lbl_dir}")

if __name__ == "__main__":
    main()
