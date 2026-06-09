"""
Quick Thai medicine image collector.
Usage: python collect.py <label> [--n 15]
Grabs a frame from the robot cam every time you press ENTER.
"""
import argparse, time, sys, urllib.request
from pathlib import Path
import cv2, numpy as np

CAM_URL = "http://localhost:6189/api/snapshot"
SAVE_DIR = Path(__file__).parent.parent / "dataset" / "thai_visual"

def grab():
    with urllib.request.urlopen(CAM_URL, timeout=5) as r:
        data = np.frombuffer(r.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def collect(label, n):
    folder = SAVE_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    existing = len(list(folder.glob("*.jpg")))
    print(f"\nCollecting {n} shots for '{label}'")
    print(f"Already have: {existing} images")
    print("─" * 40)
    captured = 0
    for i in range(1, n + 1):
        input(f"[{i}/{n}] Position medicine → ENTER to capture: ")
        try:
            frame = grab()
            if frame is None:
                print("  ✗ No frame, try again")
                continue
            idx = existing + i
            path = folder / f"img{idx:03d}_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(path), frame)
            h, w = frame.shape[:2]
            print(f"  ✓ Saved {path.name}  ({w}×{h})")
            captured += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
    print(f"\nDone — {captured}/{n} captured for '{label}'")
    print(f"Folder: {folder}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("label", help="Medicine name e.g. Yoki_Cough_Syrup")
    p.add_argument("--n", type=int, default=15)
    args = p.parse_args()
    collect(args.label, args.n)
