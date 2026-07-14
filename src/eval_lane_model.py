import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Thêm thư mục hiện tại (src) vào sys.path để import
sys.path.append(str(Path(__file__).parent))

from lane_detector import UFLDNet
from train_lane_curvelanes import (
    CurveLanesUFLDDataset,
    evaluate,
    evaluate_curves_data,
    plot_metrics_curves,
    plot_confusion_matrix,
    load_pretrained_weights
)

def main():
    parser = argparse.ArgumentParser(description="Danh gia mo hinh lane detection va ve cac bieu do thong so.")
    parser.add_argument("--model", default="models/curvelanes_finetuned_best.pth", help="Duong dan den file model checkpoint .pth")
    parser.add_argument("--dataset-dir", default="archive/Curvelanes", help="Duong dan den thu muc dataset Curvelanes")
    parser.add_argument("--limit", type=int, default=1000, help="Gioi han so anh validation de danh gia nhanh (0 = khong gioi han)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="So luong workers")
    parser.add_argument("--charts-dir", default="anh_thong_so", help="Thu muc de luu cac bieu do thong so sinh ra")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device dang dung de danh gia: {device}")
    
    # 1. Khởi tạo dataset & dataloader
    print(f"Loading dataset tu: {args.dataset_dir}")
    val_ds = CurveLanesUFLDDataset(args.dataset_dir, split="valid", limit=args.limit)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    
    # 2. Khởi tạo model va load weights
    model = UFLDNet().to(device)
    load_pretrained_weights(model, args.model)
    
    # 3. Đánh giá mô hình
    print("\n=== Dang tien hanh danh gia mo hinh ===")
    criterion = nn.CrossEntropyLoss()
    val_loss, val_cls, val_str, val_acc, val_p, val_r, val_f1, best_thresh, cm = evaluate(
        model, val_loader, criterion, device, structural_weight=4.0
    )
    
    # In kết quả thống số
    print("\n=================== KET QUA EVALUATION ===================")
    print(f"Model File Path: {args.model}")
    print(f"Dataset Split:   valid (Limit: {args.limit if args.limit > 0 else 'None'})")
    print(f"Val Total Loss:  {val_loss:.4f}")
    print(f"Val Cls Loss:    {val_cls:.4f}")
    print(f"Val Str Loss:    {val_str:.4f}")
    print(f"Val Accuracy:    {val_acc*100:.2f}% (Strict exact match)")
    print(f"Optimal Thresh:  {best_thresh:.2f} (Soft background threshold)")
    print(f"Val Precision:   {val_p*100:.2f}% (Grid distance <= 5)")
    print(f"Val Recall:      {val_r*100:.2f}% (Grid distance <= 5)")
    print(f"Val F1-Score:    {val_f1*100:.2f}% (Grid distance <= 5)")
    print("==========================================================")
    
    # 4. Vẽ các biểu đồ thông số
    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Dang sinh cac bieu do thong so tai: {charts_dir}/ ===")
    
    print("1. Dang sinh bieu do Precision, Recall, F1, PR curves...")
    thresholds, precisions, recalls, f1_scores = evaluate_curves_data(model, val_loader, device)
    plot_metrics_curves(thresholds, precisions, recalls, f1_scores, charts_dir)
    
    print("2. Dang sinh bieu do Confusion Matrix...")
    classes = ['L-Outer', 'L-Inner', 'R-Inner', 'R-Outer', 'Background']
    plot_confusion_matrix(cm, classes, charts_dir / "confusion_matrix.png", normalize=False, title="Confusion Matrix")
    plot_confusion_matrix(cm, classes, charts_dir / "confusion_matrix_normalized.png", normalize=True, title="Normalized Confusion Matrix")
    
    print("\n==> HOAN THANH! Tat ca bieu do da duoc luu lai vao thu muc: " + args.charts_dir)

if __name__ == "__main__":
    main()
