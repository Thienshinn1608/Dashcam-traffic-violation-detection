import os
import shutil
from pathlib import Path

def main():
    base_dir = Path("archive/Curvelanes")
    train_dir = base_dir / "train"
    valid_dir = base_dir / "valid"

    train_txt = train_dir / "train.txt"
    valid_txt = valid_dir / "valid.txt"

    if not train_txt.exists() or not valid_txt.exists():
        print("Error: train.txt or valid.txt does not exist!")
        return

    # Read current files
    with open(train_txt, "r", encoding="utf-8") as f:
        train_lines = [line.strip() for line in f if line.strip()]

    with open(valid_txt, "r", encoding="utf-8") as f:
        valid_lines = [line.strip() for line in f if line.strip()]

    print(f"Current train lines: {len(train_lines)}")
    print(f"Current valid lines: {len(valid_lines)}")

    target_train = 21000
    target_valid = 6000

    if len(train_lines) == target_train and len(valid_lines) == target_valid:
        print("Dataset is already in 70/20/10 split (21,000 train, 6,000 valid).")
        return

    if len(train_lines) < target_train:
        print("Error: Train split has less than 21,000 images. Cannot convert automatically.")
        return

    # Lines to move from Train to Valid
    lines_to_move = train_lines[target_train:]
    new_train_lines = train_lines[:target_train]

    print(f"Moving {len(lines_to_move)} images and labels from Train to Valid...")

    moved_images = 0
    moved_labels = 0

    # Ensure target directories exist
    (valid_dir / "images").mkdir(parents=True, exist_ok=True)
    (valid_dir / "labels").mkdir(parents=True, exist_ok=True)

    for line in lines_to_move:
        # File name (e.g. images/xxx.jpg)
        src_img = train_dir / line
        dst_img = valid_dir / line

        label_filename = Path(line).name.replace(".jpg", ".lines.json")
        src_lbl = train_dir / "labels" / label_filename
        dst_lbl = valid_dir / "labels" / label_filename

        # Move image
        if src_img.exists():
            try:
                # Ensure destination subfolders exist if nested
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_img), str(dst_img))
                moved_images += 1
            except Exception as e:
                print(f"Error moving image {src_img}: {e}")

        # Move label
        if src_lbl.exists():
            try:
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_lbl), str(dst_lbl))
                moved_labels += 1
            except Exception as e:
                print(f"Error moving label {src_lbl}: {e}")

    # Write new train.txt
    with open(train_txt, "w", encoding="utf-8") as f:
        for line in new_train_lines:
            f.write(line + "\n")

    # Append to valid.txt
    new_valid_lines = valid_lines + lines_to_move
    with open(valid_txt, "w", encoding="utf-8") as f:
        for line in new_valid_lines:
            f.write(line + "\n")

    print("==========================================")
    print("CONVERSION TO 70/20/10 COMPLETE!")
    print(f"- Moved images: {moved_images}")
    print(f"- Moved labels: {moved_labels}")
    print(f"- New Train size: {len(new_train_lines)}")
    print(f"- New Valid size: {len(new_valid_lines)}")
    print("==========================================")

if __name__ == "__main__":
    main()
