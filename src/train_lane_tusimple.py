import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# File này đặt trong src để import được lane_detector.py hiện tại của bạn
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

class TuSimpleUFLDDataset(Dataset):
    def __init__(self, train_set_dir: str, json_file: str):
        self.train_set_dir = Path(train_set_dir)
        self.json_path = self.train_set_dir / json_file
        if not self.json_path.exists():
            raise FileNotFoundError(self.json_path)

        self.items = []
        with self.json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def _encode_label(self, item, orig_w: int, orig_h: int):
        # label shape: [56 row anchors, 4 lanes], value 0..99 = grid x, value 100 = no lane
        label = np.full((UFLD_CLS_NUM, UFLD_NUM_LANES), UFLD_GRIDING_NUM, dtype=np.int64)
        h_samples = item["h_samples"]
        lanes = item["lanes"]

        for lane_id, lane in enumerate(lanes[:UFLD_NUM_LANES]):
            for r_idx, y_resized in enumerate(UFLD_ROW_ANCHOR):
                # anchor y của UFLD nằm trên ảnh 288x800, cần map về ảnh gốc 720x1280
                y_orig = int(round(y_resized / UFLD_IMG_H * orig_h))
                nearest = min(range(len(h_samples)), key=lambda i: abs(h_samples[i] - y_orig))
                x_orig = lane[nearest]
                if x_orig >= 0:
                    x_resized = x_orig / orig_w * UFLD_IMG_W
                    grid = int(x_resized / UFLD_IMG_W * UFLD_GRIDING_NUM)
                    grid = max(0, min(UFLD_GRIDING_NUM - 1, grid))
                    label[r_idx, lane_id] = grid
        return label

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = self.train_set_dir / item["raw_file"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (UFLD_IMG_W, UFLD_IMG_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        label = self._encode_label(item, orig_w, orig_h)
        label = torch.from_numpy(label).long()
        return img, label


def lane_loss(outputs, labels, criterion):
    # outputs: [B, 101, 56, 4], labels: [B, 56, 4]
    loss = 0.0
    for lane_id in range(UFLD_NUM_LANES):
        pred = outputs[:, :, :, lane_id].permute(0, 2, 1).contiguous()  # [B, 56, 101]
        target = labels[:, :, lane_id].contiguous()                    # [B, 56]
        loss = loss + criterion(pred.view(-1, UFLD_GRIDING_NUM + 1), target.view(-1))
    return loss / UFLD_NUM_LANES

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = lane_loss(outputs, labels, criterion)
        total_loss += loss.item()

        pred = outputs.argmax(dim=1)  # [B, 56, 4]
        mask = labels != UFLD_GRIDING_NUM  # chỉ tính accuracy ở điểm có lane thật
        correct += ((pred == labels) & mask).sum().item()
        total += mask.sum().item()
    acc = correct / max(total, 1)
    return total_loss / max(len(loader), 1), acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune UFLD lane model on TuSimple.")
    parser.add_argument("--train-set", required=True, help="Path to TUSimple/train_set")
    parser.add_argument("--train-json", default="train_custom.json")
    parser.add_argument("--val-json", default="val_custom.json")
    parser.add_argument("--pretrained", default="", help="Optional pretrained .pth, e.g. ../models/tusimple_18.pth")
    parser.add_argument("--save", default="../models/tusimple_finetuned_best.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = TuSimpleUFLDDataset(args.train_set, args.train_json)
    val_ds = TuSimpleUFLDDataset(args.train_set, args.val_json)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = UFLDNet().to(device)
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Loaded pretrained: {args.pretrained}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val = float("inf")
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            loss = lane_loss(outputs, labels, criterion)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | step {step}/{len(train_loader)} | loss {loss.item():.4f}")

        scheduler.step()
        train_loss = total / max(len(train_loader), 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best: {save_path}")

    print("Training done")
    print(f"Best model: {save_path}")

if __name__ == "__main__":
    main()
