"""
Claude Brain — natural-language medicine pick-and-place controller.

The brain knows which medicines are in the dataset, resolves fuzzy user commands
to a specific medicine label, scans the robot camera to locate it, then commands
the TM5-900 arm to pick it up and place it in the correct bin.

Usage:
  python brain.py --list
  python brain.py "pick up the Entresto"               # dry-run (default)
  python brain.py "grab the diabetes medication" --live # real arm motion
"""
import argparse
import os
import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from vision.claude_vision import analyze, capture_from_robot_cam, ROBOT_CAM_URL, DATASET_DIR
from arm_comms.tm5_connect import TM5

TABLE_X_MIN = 300
TABLE_X_MAX = 600
TABLE_Y_MIN = -200
TABLE_Y_MAX =  200
PICK_Z_DOWN =  50
PICK_Z_UP   = 200
TCP_RX, TCP_RY, TCP_RZ = 180, 0, 0

BINS = {
    "A": (500,  150, PICK_Z_UP),
    "B": (500,    0, PICK_Z_UP),
    "C": (500, -150, PICK_Z_UP),
}


def list_known_medicines() -> list[dict]:
    """Scan dataset folders to build the list of known medicines."""
    medicines = []
    if not DATASET_DIR.exists():
        return medicines
    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if not name.startswith("bin") or "_" not in name:
            continue
        bin_id = name[3]                    # "binA_..." → "A"
        label = name[5:]                    # "binA_Entresto_200mg" → "Entresto_200mg"
        count = len(list(folder.glob("*.jpg")))
        medicines.append({"label": label, "bin": bin_id, "images": count})
    return medicines


def resolve_target(user_command: str, known_medicines: list[dict]) -> str | None:
    """
    Ask Claude which known medicine label the user is referring to.
    Handles fuzzy / natural language: "heart medicine" → "Entresto_200mg".
    Returns the exact label string, or None if no match.
    """
    medicines_str = "\n".join(
        f"  - {m['label']} (bin {m['bin']}, {m['images']} images)"
        for m in known_medicines
    )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": (
                f"User command: '{user_command}'\n\n"
                f"Known medicines:\n{medicines_str}\n\n"
                "Which medicine label does the user want? "
                "Reply with ONLY the exact label string (e.g. 'Entresto_200mg'), "
                "or reply 'NONE' if nothing matches."
            ),
        }],
    )
    result = response.content[0].text.strip().strip("'\"")
    return None if result == "NONE" else result


def image_to_robot(pick_x: float, pick_y: float) -> tuple[float, float]:
    rx = TABLE_X_MIN + pick_x * (TABLE_X_MAX - TABLE_X_MIN)
    ry = TABLE_Y_MIN + (1 - pick_y) * (TABLE_Y_MAX - TABLE_Y_MIN)
    return round(rx, 1), round(ry, 1)


def execute_pick(arm: TM5, item: dict, dry_run: bool):
    """Suction pick at item position then place in the correct bin."""
    x, y = image_to_robot(item["pick_x"], item["pick_y"])
    bin_id = item["bin"]
    bx, by, bz = BINS.get(bin_id, BINS["C"])

    print(f"\n  Picking: {item['medicine']}")
    print(f"  Image ({item['pick_x']:.2f}, {item['pick_y']:.2f}) → Robot ({x} mm, {y} mm)")
    print(f"  Placing in bin {bin_id} → ({bx} mm, {by} mm)")

    if dry_run:
        print("  [dry-run] skipping arm motion")
        return

    # Descend → suction on → rise
    arm.move_cartesian(x, y, PICK_Z_UP,   TCP_RX, TCP_RY, TCP_RZ, speed=20)
    arm.move_cartesian(x, y, PICK_Z_DOWN, TCP_RX, TCP_RY, TCP_RZ, speed=8)
    arm.suction_on()
    time.sleep(0.5)
    arm.move_cartesian(x, y, PICK_Z_UP,   TCP_RX, TCP_RY, TCP_RZ, speed=15)

    # Move to bin → suction off → clear
    arm.move_cartesian(bx, by, bz,       TCP_RX, TCP_RY, TCP_RZ, speed=20)
    arm.suction_off()
    time.sleep(0.3)
    arm.move_cartesian(bx, by, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=15)


def run(user_command: str, dry_run: bool):
    print(f"Command: '{user_command}'")

    # 1. Load known medicines from dataset
    known = list_known_medicines()
    if not known:
        print("ERROR: No medicines in dataset. Run /snap first to populate it.")
        return
    print(f"\nKnown medicines ({len(known)}):")
    for m in known:
        print(f"  [{m['bin']}] {m['label']} — {m['images']} image(s)")

    # 2. Resolve which medicine the user wants
    print("\nResolving target medicine...")
    target_label = resolve_target(user_command, known)
    if not target_label:
        print(f"  Could not match '{user_command}' to any known medicine.")
        return
    print(f"  Target: {target_label}")

    # 3. Capture scene from robot camera
    print("\nCapturing scene from robot camera...")
    try:
        frame = capture_from_robot_cam(ROBOT_CAM_URL)
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure TMflow is running with Vision1 active.")
        return

    # 4. Identify all medicines visible in the frame
    print("Identifying medicines in frame...")
    items = analyze(frame)

    if not items:
        print("No medicines detected in frame.")
        return

    print(f"\nDetected {len(items)} medicine(s) in frame:")
    for item in items:
        conf = item.get("confidence", "?")
        flag = " ← TARGET" if item["label"].lower() == target_label.lower() else ""
        print(f"  [{item['bin']}] {item['label']} — confidence: {conf}{flag}")

    # 5. Find the target medicine
    target_item = next(
        (i for i in items if i["label"].lower() == target_label.lower()), None
    )
    if target_item is None:
        print(f"\n  Target '{target_label}' not found in frame.")
        print("  Ensure the medicine is visible and the camera is running, then try again.")
        return

    if target_item.get("confidence") == "low":
        print("\n  WARNING: low confidence identification — verify image before --live run.")

    # 6. Connect arm and pick
    arm = TM5()
    if not dry_run:
        if not arm.ping():
            print("ERROR: Cannot reach robot arm. Check TM5_IP and Listen Node in TMflow.")
            return
        arm.home()

    execute_pick(arm, target_item, dry_run)

    if not dry_run:
        arm.home()
        print(f"\nDone — {target_item['medicine']} placed in bin {target_item['bin']}.")
    else:
        print(f"\nDry-run complete. Re-run with --live to move the real arm.")


def main():
    parser = argparse.ArgumentParser(description="Claude Brain — medicine pick-and-place")
    parser.add_argument("command", nargs="?", default=None,
                        help="Natural-language pick command, e.g. 'pick up the Entresto'")
    parser.add_argument("--list", action="store_true",
                        help="List all known medicines and exit")
    parser.add_argument("--live", action="store_true",
                        help="Move the real arm (default is dry-run)")
    args = parser.parse_args()

    if args.list:
        known = list_known_medicines()
        if not known:
            print("No medicines in dataset yet. Run /snap to add some.")
        else:
            print(f"Known medicines ({len(known)}):")
            for m in known:
                print(f"  [{m['bin']}] {m['label']} — {m['images']} image(s)")
        return

    if not args.command:
        parser.print_help()
        return

    run(args.command, dry_run=not args.live)


if __name__ == "__main__":
    main()
