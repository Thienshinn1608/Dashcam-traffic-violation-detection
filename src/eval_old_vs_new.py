import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Import modules from src
sys.path.append(str(Path(__file__).parent))
from train_lane_curvelanes import CurveLanesUFLDDataset, evaluate, load_pretrained_weights
from lane_detector import UFLDNet

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load 1000 validation samples for quick comparative evaluation
    val_ds = CurveLanesUFLDDataset("archive/Curvelanes", split="valid", limit=1000)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # 1. Evaluate Old Model (tusimple_18.pth)
    print("\n--- Dang danh gia MODEL CU (tusimple_18.pth) ---")
    model_old = UFLDNet().to(device)
    load_pretrained_weights(model_old, "models/tusimple_18.pth")
    loss_old, acc_old = evaluate(model_old, val_loader, criterion, device)
    print(f"Model cu (tusimple_18) -> Loss: {loss_old:.4f} | Accuracy: {acc_old:.4f} ({acc_old*100:.2f}%)")

    # 2. Evaluate New Model (curvelanes_scratch_best.pth)
    print("\n--- Dang danh gia MODEL MOI (curvelanes_scratch_best.pth) ---")
    model_new = UFLDNet().to(device)
    load_pretrained_weights(model_new, "models/curvelanes_scratch_best.pth")
    loss_new, acc_new = evaluate(model_new, val_loader, criterion, device)
    print(f"Model moi (curvelanes_scratch) -> Loss: {loss_new:.4f} | Accuracy: {acc_new:.4f} ({acc_new*100:.2f}%)")

    print("\n================== SO SANH CHUNG ==================")
    diff_acc = acc_new - acc_old
    print(f"Do chinh xac model cu (tusimple_18): {acc_old*100:.2f}%")
    print(f"Do chinh xac model moi (curvelanes): {acc_new*100:.2f}%")
    print(f"Ket luan: Model moi {'TOT HON' if diff_acc > 0 else 'KEM HON'} model cu {abs(diff_acc)*100:.2f}% ve do chinh xac tren CurveLanes.")

if __name__ == "__main__":
    main()
