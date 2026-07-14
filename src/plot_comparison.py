import json
import os
import matplotlib.pyplot as plt

def load_log(path):
    if not os.path.exists(path):
        print(f"Loi: Khong tim thay file log tai: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    scratch_log_path = "models/curvelanes_scratch_best.json"
    finetune_log_path = "models/curvelanes_finetuned_best.json"

    scratch_data = load_log(scratch_log_path)
    finetune_data = load_log(finetune_log_path)

    if not scratch_data or not finetune_data:
        print("\n[Huong dan] Vui long dam bao rang ca hai tien trinh train (From Scratch va Fine-tuning) deu da hoan thanh va tao ra cac file log:")
        print(f"  1. {scratch_log_path}")
        print(f"  2. {finetune_log_path}")
        return

    epochs_s = [x["epoch"] for x in scratch_data]
    train_loss_s = [x["train_loss"] for x in scratch_data]
    val_loss_s = [x["val_loss"] for x in scratch_data]
    val_acc_s = [x["val_acc"] for x in scratch_data]

    epochs_f = [x["epoch"] for x in finetune_data]
    train_loss_f = [x["train_loss"] for x in finetune_data]
    val_loss_f = [x["val_loss"] for x in finetune_data]
    val_acc_f = [x["val_acc"] for x in finetune_data]

    # Style do hoa cho dep va chuyen nghiep
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Bieu do Training Loss
    axes[0].plot(epochs_f, train_loss_f, marker="o", color="#1F4E79", linewidth=2.5, label="Pretrained (Fine-tuning)")
    axes[0].plot(epochs_s, train_loss_s, marker="s", color="#E06666", linewidth=2.5, label="From Scratch (Random Init)")
    axes[0].set_title("Training Loss Comparison", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_xticks(epochs_s)
    axes[0].legend(fontsize=10, loc="upper right")
    
    # 2. Bieu do Validation Loss
    axes[1].plot(epochs_f, val_loss_f, marker="o", color="#1F4E79", linewidth=2.5, label="Pretrained (Fine-tuning)")
    axes[1].plot(epochs_s, val_loss_s, marker="s", color="#E06666", linewidth=2.5, label="From Scratch (Random Init)")
    axes[1].set_title("Validation Loss Comparison", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_xticks(epochs_s)
    axes[1].legend(fontsize=10, loc="upper right")

    # 3. Bieu do Validation Accuracy
    axes[2].plot(epochs_f, val_acc_f, marker="o", color="#1F4E79", linewidth=2.5, label="Pretrained (Fine-tuning)")
    axes[2].plot(epochs_s, val_acc_s, marker="s", color="#E06666", linewidth=2.5, label="From Scratch (Random Init)")
    axes[2].set_title("Validation Accuracy Comparison", fontsize=13, fontweight="bold", pad=10)
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Accuracy (Ratio)", fontsize=11)
    axes[2].set_xticks(epochs_s)
    axes[2].legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    
    # Luu ket qua bieu do
    output_img = "models/curvelanes_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n==> Da ve va luu bieu do so sanh tai: {output_img}")
    
    # Tao them file anh rieng le cho slide trinh chieu neu can
    # Loss comparison
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_f, val_loss_f, marker="o", color="#1F4E79", linewidth=2.5, label="Pretrained (Fine-tuning)")
    plt.plot(epochs_s, val_loss_s, marker="s", color="#E06666", linewidth=2.5, label="From Scratch (Random Init)")
    plt.title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/loss_comparison.png", dpi=300)
    
    # Accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_f, val_acc_f, marker="o", color="#1F4E79", linewidth=2.5, label="Pretrained (Fine-tuning)")
    plt.plot(epochs_s, val_acc_s, marker="s", color="#E06666", linewidth=2.5, label="From Scratch (Random Init)")
    plt.title("Validation Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (Ratio)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/accuracy_comparison.png", dpi=300)
    print("==> Da luu cac bieu do don le loss_comparison.png va accuracy_comparison.png tai models/")

if __name__ == "__main__":
    main()
