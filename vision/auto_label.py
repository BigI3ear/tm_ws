"""
Auto-generate YOLO bounding box labels for single-medicine images.

For each real image in thai_visual/<MedicineName>/, the medicine is the
main foreground object against the yellow basket. We isolate it by:
  1. Masking out the yellow basket background (HSV range)
  2. Finding the largest remaining contour
  3. Expanding the box 10% and clamping to image bounds

Outputs YOLO label files (.txt) alongside each image, and copies
images + labels into dataset/yolo/images/ and dataset/yolo/labels/
with an 80/20 train/val split.
"""
import cv2
import numpy as np
import shutil
import random
from pathlib import Path

THAI_VISUAL_DIR = Path(__file__).parent.parent.parent / "dataset" / "thai_visual"
YOLO_DIR = Path(__file__).parent.parent.parent / "dataset" / "yolo"
TRAIN_RATIO = 0.8

MEDICINES = [
    "Betadine",
    "Gentian_Violet",
    "Leopard_Cough_Syrup",
    "Siribuncha_Alcohol",
    "Ya_That_Nam_Khao_White_Rabbit",
]


def get_medicine_bbox(img: np.ndarray) -> tuple[float, float, float, float] | None:
    """
    Returns (cx, cy, w, h) normalised 0-1, or None if detection fails.
    Masks out the yellow basket, finds the largest foreground blob.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow basket mask — broad range to catch the basket frame
    yellow_lo = np.array([18, 80, 80], dtype=np.uint8)
    yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
    basket_mask = cv2.inRange(hsv, yellow_lo, yellow_hi)

    # Foreground = everything that is NOT yellow basket
    fg = cv2.bitwise_not(basket_mask)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour = medicine
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return None

    x, y, w, h = cv2.boundingRect(c)
    H, W = img.shape[:2]

    # Expand 10%
    pad_x = int(w * 0.10)
    pad_y = int(h * 0.10)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(W - x, w + 2 * pad_x)
    h = min(H - y, h + 2 * pad_y)

    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    nw = w / W
    nh = h / H
    return round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)


def build():
    # Clear existing YOLO split
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = YOLO_DIR / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    class_names = sorted(MEDICINES)
    class_map = {name: i for i, name in enumerate(class_names)}

    stats = {name: {"ok": 0, "fail": 0} for name in class_names}

    all_entries = []  # (img_path, label_text)

    for med in class_names:
        med_dir = THAI_VISUAL_DIR / med
        if not med_dir.exists():
            print(f"  SKIP {med} — folder not found")
            continue

        cls_id = class_map[med]
        images = sorted(med_dir.glob("*_real.jpg"))
        print(f"  {med} (class {cls_id}): {len(images)} real images")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                stats[med]["fail"] += 1
                continue

            bbox = get_medicine_bbox(img)
            if bbox is None:
                print(f"    WARN: bbox detection failed for {img_path.name}")
                stats[med]["fail"] += 1
                continue

            cx, cy, w, h = bbox
            label = f"{cls_id} {cx} {cy} {w} {h}\n"
            all_entries.append((img_path, label))
            stats[med]["ok"] += 1

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * TRAIN_RATIO)
    splits = {"train": all_entries[:split_idx], "val": all_entries[split_idx:]}

    for split, entries in splits.items():
        for img_path, label in entries:
            dst_img = YOLO_DIR / "images" / split / img_path.name
            dst_lbl = YOLO_DIR / "labels" / split / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text(label)

    # Write dataset.yaml
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {YOLO_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    print("\n--- Summary ---")
    total_ok = sum(s["ok"] for s in stats.values())
    total_fail = sum(s["fail"] for s in stats.values())
    for med, s in stats.items():
        print(f"  {med}: {s['ok']} ok, {s['fail']} failed")
    print(f"\nTotal: {total_ok} labelled, {total_fail} failed")
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    print("Auto-labelling single-medicine images...")
    build()
