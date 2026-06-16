"""
TCP server for TMflow's AI_Path_Send / AI_Recv Network nodes.

TMflow connects to this server (configured as ntd_Local -> 192.168.1.103:6190,
forwarded to this WSL2 machine), sends a trigger string like:

    "tmc_pickup,<image_path>"

This server then grabs the latest frame from the robot camera
(http://localhost:6189/api/snapshot, posted there by Vision1), runs medicine
detection, converts the first detected item's pixel position to robot mm via
image_to_robot(), and replies on the same socket with:

    "<x>,<y>"

Run with:
  ANTHROPIC_API_KEY=sk-... python3 tmflow_server.py
"""
import socket

import cv2
import numpy as np
import requests

from vision.claude_vision import detect_all_medicines, scan_qr, classify_medicine, detect_with_yolo
from main import image_to_robot

HOST = "0.0.0.0"
PORT = 6190
SNAPSHOT_URL = "http://localhost:6189/api/snapshot"


def _preload_models():
    """Load YOLO model at startup so the first real request isn't slow."""
    try:
        detect_with_yolo(np.zeros((64, 64, 3), dtype=np.uint8))
        print("[tmflow_server] YOLO model loaded")
    except Exception as e:
        print(f"[tmflow_server] YOLO preload skipped: {e}")


def get_frame():
    """Fetch latest frame from image server. Returns None if not ready yet."""
    try:
        resp = requests.get(SNAPSHOT_URL, timeout=10)
        if resp.status_code != 200:
            return None
        arr = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def detect_first_item(frame):
    qr_hits = scan_qr(frame)
    if qr_hits:
        hit = qr_hits[0]
        info = classify_medicine(hit["data"])
        return {
            "medicine": info.get("medicine", hit["data"]),
            "pick_x": hit["center_x"],
            "pick_y": hit["center_y"],
        }
    results = detect_all_medicines(frame)
    return results[0] if results else None


def handle_client(conn, addr):
    print(f"[tmflow_server] connection from {addr}")
    data = conn.recv(4096).decode("utf-8", errors="ignore").strip()
    print(f"[tmflow_server] received: {data!r}")

    if not data.startswith("tmc_pickup"):
        print(f"[tmflow_server] unrecognized command, ignoring")
        return

    try:
        frame = get_frame()
        if frame is None:
            print("[tmflow_server] no frame available (Vision1 not posted yet)")
            reply = "0,0"
            conn.sendall(reply.encode("utf-8"))
            print(f"[tmflow_server] sent: {reply!r}")
            return
        item = detect_first_item(frame)
        if item is None:
            print("[tmflow_server] no medicine detected")
            reply = "0,0"
        else:
            x, y = image_to_robot(item["pick_x"], item["pick_y"])
            print(f"[tmflow_server] {item.get('medicine')} -> ({x}, {y})")
            reply = f"{x},{y}"
    except Exception as e:
        print(f"[tmflow_server] error: {e}")
        reply = "0,0"

    conn.sendall(reply.encode("utf-8"))
    print(f"[tmflow_server] sent: {reply!r}")


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    _preload_models()
    print(f"[tmflow_server] listening on {HOST}:{PORT}")
    while True:
        conn, addr = s.accept()
        try:
            handle_client(conn, addr)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
