import argparse
import json
import os
import random
from pathlib import Path

DEFAULT_JSONS = [
    "label_data_0313.json",
    "label_data_0531.json",
    "label_data_0601.json",
]

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split TuSimple train_set JSON labels into train/val JSONL files.")
    parser.add_argument("--train-set", required=True, help="Path to TUSimple/train_set")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio, default 0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-train", default="train_custom.json")
    parser.add_argument("--out-val", default="val_custom.json")
    args = parser.parse_args()

    train_set = Path(args.train_set)
    all_rows = []
    for name in DEFAULT_JSONS:
        p = train_set / name
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        rows = read_jsonl(p)
        print(f"Loaded {len(rows):5d} rows from {p.name}")
        all_rows.extend(rows)

    random.seed(args.seed)
    random.shuffle(all_rows)
    val_n = max(1, int(len(all_rows) * args.val_ratio))
    val_rows = all_rows[:val_n]
    train_rows = all_rows[val_n:]

    out_train = train_set / args.out_train
    out_val = train_set / args.out_val
    write_jsonl(out_train, train_rows)
    write_jsonl(out_val, val_rows)

    print("Done")
    print(f"Train: {len(train_rows)} -> {out_train}")
    print(f"Val  : {len(val_rows)} -> {out_val}")

if __name__ == "__main__":
    main()
