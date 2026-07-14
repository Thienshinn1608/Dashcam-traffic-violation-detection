import os
from pathlib import Path

def main():
    dataset_dir = Path("archive/Curvelanes")
    train_dir = dataset_dir / "train"
    train_txt = train_dir / "train.txt"

    if not train_txt.exists():
        print(f"Error: {train_txt} does not exist!")
        return

    # Read all lines
    print("Reading train.txt...")
    with open(train_txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_images = len(lines)
    print(f"Total images in train.txt: {total_images}")

    target_count = 30000
    if total_images <= target_count:
        print(f"Dataset already has {total_images} images, which is <= {target_count}. No need to shrink.")
        return

    keep_lines = lines[:target_count]
    delete_lines = lines[target_count:]

    print(f"Keeping {len(keep_lines)} images.")
    print(f"Deleting {len(delete_lines)} images from disk...")

    # Write back the kept lines
    with open(train_txt, "w", encoding="utf-8") as f:
        for line in keep_lines:
            f.write(line + "\n")
    print(f"Updated {train_txt} to contain only the first {target_count} images.")

    # Delete physical files
    deleted_images_count = 0
    deleted_labels_count = 0
    freed_bytes = 0

    for idx, rel_path in enumerate(delete_lines):
        img_path = train_dir / rel_path
        
        # Label path
        label_filename = Path(rel_path).name.replace(".jpg", ".lines.json")
        label_path = train_dir / "labels" / label_filename

        # Delete image
        if img_path.exists():
            try:
                freed_bytes += img_path.stat().st_size
                img_path.unlink()
                deleted_images_count += 1
            except Exception as e:
                print(f"Error deleting image {img_path}: {e}")

        # Delete label
        if label_path.exists():
            try:
                freed_bytes += label_path.stat().st_size
                label_path.unlink()
                deleted_labels_count += 1
            except Exception as e:
                print(f"Error deleting label {label_path}: {e}")

        if (idx + 1) % 10000 == 0 or (idx + 1) == len(delete_lines):
            print(f"Progress: Processed {idx + 1}/{len(delete_lines)} deletion files...")

    freed_mb = freed_bytes / (1024 * 1024)
    freed_gb = freed_mb / 1024
    print("==========================================")
    print("SHRINK DATASET COMPLETED SUCCESSFULLY!")
    print(f"- Deleted images: {deleted_images_count}")
    print(f"- Deleted labels: {deleted_labels_count}")
    print(f"- Freed VRAM/Disk space: {freed_gb:.2f} GB ({freed_mb:.2f} MB)")
    print("==========================================")

if __name__ == "__main__":
    main()
