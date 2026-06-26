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
# All positions are SUCTION TIP coordinates (calibrated 2026-05-12)
# Scan pose = camera home for vision — pick positions are suction tip coords
# ---------------------------------------------------------------------------

# Basket physical size, measured directly (2026-06-11): X=26cm, Y=32cm
# Spans recomputed around the calibrated centre below.
BASKET_WIDTH_MM  = 260.0  # mm — X span (image width direction)
BASKET_HEIGHT_MM = 320.0  # mm — Y span (image height direction)

# Basket corners measured by jogging robot to top-left of image (2026-06-18)
# image (0,0) top-left  → robot (-325.36, 900.85)
# image (1,1) bot-right → robot (-65.36,  580.85)  (computed from basket size)
BASKET_X_MIN = -325.36  # mm — left edge  (image x=0)
BASKET_X_MAX =  -65.36  # mm — right edge (image x=1)
BASKET_Y_MAX =  900.85  # mm — far edge   (image y=0)
BASKET_Y_MIN =  580.85  # mm — near edge  (image y=1)
BASKET_X     = (BASKET_X_MIN + BASKET_X_MAX) / 2  # -195.36
BASKET_Y     = (BASKET_Y_MIN + BASKET_Y_MAX) / 2  # 740.85

# Image resolution captured by the robot camera (whole basket fills the frame)
IMAGE_WIDTH_PX  = 2592
IMAGE_HEIGHT_PX = 1944
MM_PER_PX_X = BASKET_WIDTH_MM / IMAGE_WIDTH_PX    # ~0.100 mm/px
MM_PER_PX_Y = BASKET_HEIGHT_MM / IMAGE_HEIGHT_PX  # ~0.165 mm/px
PICK_Z_DOWN  =  187.37  # mm — suction Z touching medicine at basket centre (calibrated 2026-05-12)
PICK_Z_UP    =  254.63  # mm — safe travel height (scan position Z, 2026-06-18)

# Grasp orientation (from scan position read 2026-06-18)
TCP_RX, TCP_RY, TCP_RZ = -179.94, 0.37, -179.26

# Bin drop-off position (calibrated 2026-06-10 — physically jogged & read from /tool_pose)
BIN_X      =  174.32  # mm
BIN_Y      =  702.61  # mm
BIN_Z_TOP  =  288.26  # mm
BINS = {
    "A": (BIN_X, BIN_Y, PICK_Z_UP),
    "B": (BIN_X, BIN_Y, PICK_Z_UP),
    "C": (BIN_X, BIN_Y, PICK_Z_UP),
}

# Bin B verify scan — top-left corner (image 0,0) measured 2026-06-22
BINB_X_MIN =   -8.843  # mm
BINB_Y_MAX =  876.890  # mm
BINB_WIDTH_MM  = BASKET_WIDTH_MM   # 260mm — same as basket A
BINB_HEIGHT_MM = BASKET_HEIGHT_MM  # 320mm — same as basket A


def image_to_binb_offset(pick_x: float, pick_y: float) -> tuple[float, float]:
    """Map normalised image coords to offset from bin B top-left corner (X inverted)."""
    ox = round(-pick_x * BINB_WIDTH_MM, 1)
    oy = round(-pick_y * BINB_HEIGHT_MM, 1)
    return ox, oy


# Bin C scan — top-left corner measured 2026-06-23
BINC_X_MIN =  -21.37
BINC_Y_MAX = -898.75
BINC_WIDTH_MM  = BASKET_WIDTH_MM
BINC_HEIGHT_MM = BASKET_HEIGHT_MM


def image_to_binc_offset(pick_x: float, pick_y: float) -> tuple[float, float]:
    """Map normalised image coords to offset from bin C top-left corner (X inverted)."""
    ox = round(-pick_x * BINC_WIDTH_MM, 1)
    oy = round(-pick_y * BINC_HEIGHT_MM, 1)
    return ox, oy


def image_to_robot(pick_x: float, pick_y: float) -> tuple[float, float]:
    """Map normalised image coords (0-1) to robot Cartesian mm (absolute base frame)."""
    rx = BASKET_X_MIN + pick_x * (BASKET_X_MAX - BASKET_X_MIN)
    ry = BASKET_Y_MAX - pick_y * (BASKET_Y_MAX - BASKET_Y_MIN)
    return round(rx, 1), round(ry, 1)


def image_to_offset(pick_x: float, pick_y: float) -> tuple[float, float]:
    """Map normalised image coords (0-1) to mm offset from basket top-left corner.
    Robot goes to top-left (0,0) first, then this offset is added by the Move node."""
    ox = round(-pick_x * BASKET_WIDTH_MM, 1)
    oy = round(-pick_y * BASKET_HEIGHT_MM, 1)
    return ox, oy


def place_in_bin(arm: TM5, bin_id: str, dry_run: bool):
    bx, by, bz = BINS.get(bin_id, BINS["C"])
    print(f"    Placing in bin {bin_id} → ({bx} mm, {by} mm)")
    if dry_run:
        print("    [dry-run] skipping arm motion")
        return
    arm.move_cartesian(bx, by, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=0.5)
    arm.move_cartesian(bx, by, BIN_Z_TOP,  TCP_RX, TCP_RY, TCP_RZ, speed=0.1)
    arm.suction_off()
    time.sleep(0.3)
    arm.move_cartesian(bx, by, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=0.4)
    arm.home()


def pick_medicine(arm: TM5, item: dict, dry_run: bool):
    x, y = image_to_robot(item["pick_x"], item["pick_y"])
    print(f"\n  [{item['bin']}] {item['medicine']}")
    print(f"    {item['description']}")
    print(f"    Image ({item['pick_x']:.2f}, {item['pick_y']:.2f}) → Robot ({x} mm, {y} mm)")

    if dry_run:
        print("    [dry-run] skipping arm motion")
        return

    arm.move_cartesian(x, y, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=0.5)
    arm.move_cartesian(x, y, PICK_Z_DOWN, TCP_RX, TCP_RY, TCP_RZ, speed=0.1)
    arm.suction_on()
    time.sleep(0.5)
    arm.move_cartesian(x, y, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=0.4)

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
