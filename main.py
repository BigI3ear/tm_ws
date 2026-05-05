"""
Digital Twin — Medicine QR Scanner Pipeline
  Camera → QR decode → Claude classifies medicine → TM5-900 picks and places

Usage:
  python main.py --dry-run                        # test without arm (webcam)
  python main.py --image photo.jpg --dry-run      # use a saved image
  python main.py --robot-cam --dry-run            # use robot eye-in-hand camera
  ROBOT_CAM_URL=http://<wsl2-ip>:6189/api/snapshot python main.py --robot-cam
  TM5_IP=192.168.1.102 python main.py --robot-cam  # full live run
"""
import argparse
import sys
import time

import cv2

from vision.claude_vision import analyze, capture_frame, capture_from_robot_cam, ROBOT_CAM_URL
from arm_comms.tm5_connect import TM5

# ---------------------------------------------------------------------------
# Workspace calibration — update these to match your physical table setup
# ---------------------------------------------------------------------------
TABLE_X_MIN =  300   # mm — left edge of camera FOV in robot frame
TABLE_X_MAX =  600   # mm — right edge
TABLE_Y_MIN = -200   # mm — far from robot
TABLE_Y_MAX =  200   # mm — near robot
PICK_Z_DOWN =   50   # mm — descent height for grasp
PICK_Z_UP   =  200   # mm — travel height between poses
TCP_RX, TCP_RY, TCP_RZ = 180, 0, 0  # top-down grasp orientation

# Bin drop-off positions (robot Cartesian, mm)
BINS = {
    "A": (500,  150, PICK_Z_UP),   # Common / OTC medicines
    "B": (500,    0, PICK_Z_UP),   # Prescription
    "C": (500, -150, PICK_Z_UP),   # Controlled / unknown
}


def image_to_robot(pick_x: float, pick_y: float) -> tuple[float, float]:
    rx = TABLE_X_MIN + pick_x * (TABLE_X_MAX - TABLE_X_MIN)
    ry = TABLE_Y_MIN + (1 - pick_y) * (TABLE_Y_MAX - TABLE_Y_MIN)
    return round(rx, 1), round(ry, 1)


def place_in_bin(arm: TM5, bin_id: str, dry_run: bool):
    bx, by, bz = BINS.get(bin_id, BINS["C"])
    print(f"    Placing in bin {bin_id} → ({bx} mm, {by} mm)")
    if dry_run:
        print("    [dry-run] skipping arm motion")
        return
    arm.move_cartesian(bx, by, bz, TCP_RX, TCP_RY, TCP_RZ, speed=20)
    arm.suction_off()
    time.sleep(0.3)
    arm.move_cartesian(bx, by, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=15)


def pick_medicine(arm: TM5, item: dict, dry_run: bool):
    x, y = image_to_robot(item["pick_x"], item["pick_y"])
    print(f"\n  [{item['bin']}] {item['medicine']}")
    print(f"    {item['description']}")
    print(f"    Image ({item['pick_x']:.2f}, {item['pick_y']:.2f}) → Robot ({x} mm, {y} mm)")

    if dry_run:
        print("    [dry-run] skipping arm motion")
        return

    arm.move_cartesian(x, y, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=20)
    arm.move_cartesian(x, y, PICK_Z_DOWN, TCP_RX, TCP_RY, TCP_RZ, speed=8)
    arm.suction_on()
    time.sleep(0.5)
    arm.move_cartesian(x, y, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=15)

    place_in_bin(arm, item["bin"], dry_run=False)


def run(image_source, dry_run: bool, robot_cam_url: str | None = None):
    arm = TM5()

    if not dry_run:
        print("Connecting to TM5-900...")
        if not arm.ping():
            print("ERROR: Cannot reach robot. Check TM5_IP and that Listen Node is active in TMflow.")
            sys.exit(1)
        print("Connected.")
        arm.home()
        arm.scan_pose()
        time.sleep(1)

    print("\nCapturing image...")
    if robot_cam_url is not None:
        print(f"  Triggering robot camera...")
        if not dry_run:
            arm.capture()
            time.sleep(1.5)  # wait for frame to arrive at Flask server
        frame = capture_from_robot_cam(robot_cam_url)
        cv2.imwrite("last_capture.jpg", frame)
        print("  Snapshot saved → last_capture.jpg")
        source = frame
    elif image_source is None:
        frame = capture_frame()
        cv2.imwrite("last_capture.jpg", frame)
        print("  Snapshot saved → last_capture.jpg")
        source = frame
    else:
        source = image_source

    print("Scanning for QR codes and classifying medicines...")
    items = analyze(source)

    if not items:
        print("\nNo QR codes detected. Make sure a medicine tablet with a QR code is visible.")
        return

    print(f"\nFound {len(items)} medicine(s):")
    for item in items:
        src = "QR" if item.get("source") == "qr" else "vision (no QR)"
        conf = item.get("confidence", "?")
        print(f"  Identified via {src} — confidence: {conf}")
        if conf == "low":
            print("  WARNING: low confidence — verify before trusting arm motion")
        pick_medicine(arm, item, dry_run=dry_run)

    if not dry_run:
        print("\nReturning to home...")
        arm.home()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="TM5-900 Medicine QR Scanner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test vision and logic without moving the arm")
    parser.add_argument("--image", type=str, default=None,
                        help="Use an image file instead of the live camera")
    parser.add_argument("--robot-cam", action="store_true",
                        help="Use the TM5-900 built-in camera via the image server")
    parser.add_argument("--robot-cam-url", type=str, default=ROBOT_CAM_URL,
                        help=f"Snapshot URL of the robot image server (default: {ROBOT_CAM_URL})")
    args = parser.parse_args()
    run(
        image_source=args.image,
        dry_run=args.dry_run,
        robot_cam_url=args.robot_cam_url if args.robot_cam else None,
    )


if __name__ == "__main__":
    main()
