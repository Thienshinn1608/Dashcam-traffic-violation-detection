import os
from pathlib import Path

def shrink_split_with_txt(split_dir, split_name, target_count):
    txt_file = split_dir / f"{split_name}.txt"
    if not txt_file.exists():
        print(f"[{split_name.upper()}] Error: {txt_file} does not exist!")
        return 0, 0, 0

    # Read current lines
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_images = len(lines)
    print(f"[{split_name.upper()}] Current images in txt: {total_images}")

    if total_images <= target_count:
        print(f"[{split_name.upper()}] Already has {total_images} images <= {target_count}. No need to shrink.")
        return 0, 0, 0

    keep_lines = lines[:target_count]
    delete_lines = lines[target_count:]

    # Update txt file
    with open(txt_file, "w", encoding="utf-8") as f:
        for line in keep_lines:
            f.write(line + "\n")
    print(f"[{split_name.upper()}] Updated txt file to {target_count} images.")

    deleted_images = 0
    deleted_labels = 0
    freed_bytes = 0

    # Delete files
    for rel_path in delete_lines:
        img_path = split_dir / rel_path
        
        # Label path
        label_filename = Path(rel_path).name.replace(".jpg", ".lines.json")
        label_path = split_dir / "labels" / label_filename

        if img_path.exists():
            try:
                freed_bytes += img_path.stat().st_size
                img_path.unlink()
                deleted_images += 1
            except Exception as e:
                pass

        if label_path.exists():
            try:
                freed_bytes += label_path.stat().st_size
                label_path.unlink()
                deleted_labels += 1
            except Exception as e:
                pass

    print(f"[{split_name.upper()}] Deleted: {deleted_images} images, {deleted_labels} labels. Freed: {freed_bytes / (1024*1024):.2f} MB")
    return deleted_images, deleted_labels, freed_bytes


def shrink_test_split(split_dir, target_count):
    img_dir = split_dir / "images"
    if not img_dir.exists():
        print(f"[TEST] Error: {img_dir} does not exist!")
        return 0, 0

    # Get all image files in images/
    all_images = sorted([p for p in img_dir.glob("**/*") if p.is_file()])
    total_images = len(all_images)
    print(f"[TEST] Current images on disk: {total_images}")

    if total_images <= target_count:
        print(f"[TEST] Already has {total_images} images <= {target_count}. No need to shrink.")
        return 0, 0

    delete_files = all_images[target_count:]
    deleted_images = 0
    freed_bytes = 0

    for p in delete_files:
        try:
            freed_bytes += p.stat().st_size
            p.unlink()
            deleted_images += 1
        except Exception as e:
            pass

    print(f"[TEST] Deleted: {deleted_images} images. Freed: {freed_bytes / (1024*1024):.2f} MB")
    return deleted_images, freed_bytes


def main():
    base_dir = Path("archive/Curvelanes")
    
    # Target counts for each split to sum up to exactly 30,000 total images
    # Train: 24,000 (80%)
    # Valid: 3,000 (10%)
    # Test:  3,000 (10%)
    train_target = 24000
    valid_target = 3000
    test_target = 3000

    print("==================================================")
    print("STARTING MULTI-SPLIT SHRINK TO TOTAL 30,000 IMAGES")
    print("==================================================")

    # 1. Shrink Train
    tr_img, tr_lbl, tr_bytes = shrink_split_with_txt(base_dir / "train", "train", train_target)

    # 2. Shrink Valid
    val_img, val_lbl, val_bytes = shrink_split_with_txt(base_dir / "valid", "valid", valid_target)

    # 3. Shrink Test
    te_img, te_bytes = shrink_test_split(base_dir / "test", test_target)

    total_deleted_images = tr_img + val_img + te_img
    total_deleted_labels = tr_lbl + val_lbl
    total_freed_bytes = tr_bytes + val_bytes + te_bytes
    total_freed_gb = total_freed_bytes / (1024 * 1024 * 1024)

    print("==================================================")
    print("SHRINK ALL SPLITS COMPLETED SUCCESSFULLY!")
    print(f"- Total deleted images: {total_deleted_images}")
    print(f"- Total deleted labels: {total_deleted_labels}")
    print(f"- Total disk space freed: {total_freed_gb:.2f} GB")
    print("==================================================")

if __name__ == "__main__":
    main()
