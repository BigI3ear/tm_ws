"""
Quick labeling tool — run after taking a test shot to create the ground truth JSON.

Usage:
  python3 label_image.py test_images/baseline/shot_01.png

It will prompt you for what's in the image and save shot_01.json alongside it.
"""
import json
import sys
from pathlib import Path

def label(image_path: str):
    img = Path(image_path)
    if not img.exists():
        print(f"Image not found: {image_path}")
        return

    print(f"\nLabeling: {img.name}")
    print("="*50)

    medicines = []
    print("Enter medicine names one by one (blank line to finish):")
    while True:
        m = input("  Medicine: ").strip()
        if not m:
            break
        medicines.append(m)

    print("\nLighting condition:")
    print("  1. normal  2. dim  3. bright  4. shadow")
    lighting_map = {"1":"normal","2":"dim","3":"bright","4":"shadow","":"normal"}
    lighting = lighting_map.get(input("  Choice [1]: ").strip(), "normal")

    print("\nOcclusion:")
    print("  1. none  2. plastic wrap  3. covered by another med")
    occlusion_map = {"1":"none","2":"plastic","3":"covered","":"none"}
    occlusion = occlusion_map.get(input("  Choice [1]: ").strip(), "none")

    print("\nOrientation:")
    print("  1. normal  2. tilted  3. upside_down")
    orientation_map = {"1":"normal","2":"tilted","3":"upside_down","":"normal"}
    orientation = orientation_map.get(input("  Choice [1]: ").strip(), "normal")

    notes = input("\nNotes (optional): ").strip()

    gt = {
        "medicines":   medicines,
        "lighting":    lighting,
        "occlusion":   occlusion,
        "density":     len(medicines),
        "orientation": orientation,
        "notes":       notes,
    }

    out = img.with_suffix(".json")
    with open(out, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"\nSaved: {out}")
    print(json.dumps(gt, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 label_image.py <image_path>")
    else:
        label(sys.argv[1])
