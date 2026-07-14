import json
import os
import matplotlib.pyplot as plt

def main():
    scratch_log_path = "models/curvelanes_scratch_best.json"
    
    if not os.path.exists(scratch_log_path):
        print(f"Loi: Khong tim thay file log cua model moi tai: {scratch_log_path}")
        return

    with open(scratch_log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = [x["epoch"] for x in data]
    train_loss = [x["train_loss"] for x in data]
    val_loss = [x["val_loss"] for x in data]
    val_acc = [x["val_acc"] * 100 for x in data]  # Chuyen sang %

    # Chi so thuc te cua Model cu (tusimple_18) da test tren cung 1000 anh CurveLanes
    old_loss = 2.0697
    old_acc = 7.29

    # Set theme
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Bieu do so sanh Loss
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, train_loss, marker="o", color="#4A90E2", linewidth=2.5, label="Train Loss (Model Moi)")
    plt.plot(epochs, val_loss, marker="s", color="#50E3C2", linewidth=2.5, label="Val Loss (Model Moi)")
    plt.axhline(y=old_loss, color="#D0021B", linestyle="--", linewidth=2.0, label="Val Loss (Model Cu - tusimple_18)")
    
    plt.title("So Sanh Loss Giua Model Cu va Model Moi Tren CurveLanes", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig("models/loss_comparison_curve.png", dpi=300)
    print("==> Da luu bieu do loss tai: models/loss_comparison_curve.png")

    # 2. Bieu do so sanh Accuracy
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, val_acc, marker="o", color="#F5A623", linewidth=2.5, label="Val Accuracy (Model Moi)")
    plt.axhline(y=old_acc, color="#D0021B", linestyle="--", linewidth=2.0, label="Val Accuracy (Model Cu - 7.29%)")
    
    # Text annotation cho do lech
    for i, acc in enumerate(val_acc):
        plt.text(epochs[i], acc + 1.2, f"{acc:.2f}%", ha="center", fontweight="bold", color="#333333")

    plt.title("So Sanh Do Chinh Xac (Accuracy) Tren CurveLanes", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(epochs)
    plt.ylim(0, 45)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig("models/accuracy_comparison_curve.png", dpi=300)
    print("==> Da luu bieu do accuracy tai: models/accuracy_comparison_curve.png")

if __name__ == "__main__":
    main()
