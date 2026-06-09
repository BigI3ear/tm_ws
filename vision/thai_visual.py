"""
Thai medicine visual recognition using CLIP embeddings.

Workflow:
  1. collect  — capture N images of a medicine, save to dataset/thai_visual/<label>/
  2. build_db — compute CLIP embeddings for all collected images, save to thai_visual_db.npz
  3. match    — given a new frame, return the closest medicine and confidence

Usage:
  python thai_visual.py collect "Yoki_Cough_Syrup" --n 15
  python thai_visual.py build
  python thai_visual.py test image.jpg
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

THAI_VISUAL_DIR = Path(__file__).parent.parent.parent / "dataset" / "thai_visual"
DB_PATH = Path(__file__).parent.parent.parent / "dataset" / "thai_visual_db.npz"

# Similarity threshold — below this is "no match"
MATCH_THRESHOLD = 0.70


def _get_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    return model, preprocess


def _embed(model, preprocess, img_bgr: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image as PILImage
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    tensor = preprocess(pil).unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).numpy()


def collect(label: str, n: int = 15, cam_url: str = None):
    """Capture n images of a medicine from the robot camera and save to thai_visual/<label>/."""
    folder = THAI_VISUAL_DIR / label
    folder.mkdir(parents=True, exist_ok=True)

    existing = list(folder.glob("*.jpg"))
    start_idx = len(existing) + 1

    if cam_url:
        import urllib.request
        def grab():
            with urllib.request.urlopen(cam_url, timeout=5) as r:
                data = np.frombuffer(r.read(), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(0)
        def grab():
            for _ in range(3): cap.read()
            _, f = cap.read()
            return f

    print(f"Collecting {n} images for '{label}' → {folder}")
    print("Move/rotate the medicine between shots. Press ENTER for each capture.")
    captured = 0
    for i in range(n):
        if cam_url:
            input(f"  Shot {start_idx + i}/{start_idx + n - 1} — position medicine, then ENTER: ")
            frame = grab()
        else:
            input(f"  Shot {start_idx + i}/{start_idx + n - 1} — position medicine, then ENTER: ")
            frame = grab()
        if frame is None:
            print("  Failed to capture, skipping.")
            continue
        ts = time.strftime("%H%M%S")
        path = folder / f"img{start_idx + i:03d}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"  Saved → {path.name}")
        captured += 1

    if cam_url is None:
        cap.release()
    print(f"\nDone. {captured} images saved for '{label}'.")


def build_db():
    """Compute CLIP embeddings for all images in thai_visual/ and save to DB_PATH."""
    print("Loading CLIP model...")
    model, preprocess = _get_model()

    labels, embeddings, paths = [], [], []
    for med_dir in sorted(THAI_VISUAL_DIR.iterdir()):
        if not med_dir.is_dir():
            continue
        imgs = list(med_dir.glob("*.jpg")) + list(med_dir.glob("*.png"))
        if not imgs:
            continue
        print(f"  {med_dir.name}: {len(imgs)} images")
        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            emb = _embed(model, preprocess, frame)
            labels.append(med_dir.name)
            embeddings.append(emb)
            paths.append(str(img_path))

    if not embeddings:
        print("No images found. Run collect first.")
        return

    np.savez(
        str(DB_PATH),
        labels=np.array(labels),
        embeddings=np.array(embeddings),
        paths=np.array(paths),
    )
    print(f"\nDB saved → {DB_PATH}  ({len(embeddings)} embeddings, {len(set(labels))} medicines)")


def match(frame: np.ndarray) -> dict:
    """
    Match a frame against the thai_visual_db.
    Returns {medicine, label, confidence, score} or None if no match above threshold.
    """
    if not DB_PATH.exists():
        return None

    db = np.load(str(DB_PATH), allow_pickle=True)
    db_labels = db["labels"]
    db_embeddings = db["embeddings"]

    model, preprocess = _get_model()
    query = _embed(model, preprocess, frame)

    scores = db_embeddings @ query
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_label = db_labels[best_idx]

    if best_score < MATCH_THRESHOLD:
        return None

    # Aggregate scores per label for confidence estimation
    label_scores = {}
    for lbl, score in zip(db_labels, scores):
        label_scores.setdefault(lbl, []).append(float(score))
    avg_scores = {lbl: np.mean(v) for lbl, v in label_scores.items()}
    sorted_labels = sorted(avg_scores, key=avg_scores.get, reverse=True)
    top_score = avg_scores[sorted_labels[0]]

    if best_score >= 0.85:
        confidence = "high"
    elif best_score >= 0.75:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "label":      best_label,
        "medicine":   best_label.replace("_", " "),
        "score":      round(best_score, 3),
        "confidence": confidence,
    }


def test(img_path: str):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Cannot read {img_path}")
        return
    result = match(frame)
    if result:
        print(f"Match:      {result['medicine']}")
        print(f"Score:      {result['score']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"No match above threshold ({MATCH_THRESHOLD})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_col = sub.add_parser("collect", help="Capture images for a medicine")
    p_col.add_argument("label", help="Medicine label e.g. Yoki_Cough_Syrup")
    p_col.add_argument("--n", type=int, default=15, help="Number of images")
    p_col.add_argument("--cam-url", type=str, default=None, help="Robot camera snapshot URL")

    sub.add_parser("build", help="Build embedding DB from collected images")

    p_test = sub.add_parser("test", help="Test matching on an image")
    p_test.add_argument("image", help="Path to image file")

    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args.label, args.n, args.cam_url)
    elif args.cmd == "build":
        build_db()
    elif args.cmd == "test":
        test(args.image)
    else:
        parser.print_help()
