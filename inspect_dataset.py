import os
from pathlib import Path

def count_files(dir_path):
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.glob("**/*") if p.is_file())

def main():
    base_dir = Path("archive/Curvelanes")
    for split in ["train", "valid", "test"]:
        split_dir = base_dir / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        txt_file = split_dir / f"{split}.txt"

        print(f"Split: {split}")
        if txt_file.exists():
            with open(txt_file, "r") as f:
                lines = len([line for line in f if line.strip()])
            print(f"  - Lines in txt: {lines}")
        else:
            print(f"  - No txt file found")

        print(f"  - Files in images/: {count_files(img_dir)}")
        print(f"  - Files in labels/: {count_files(lbl_dir)}")

if __name__ == "__main__":
    main()
