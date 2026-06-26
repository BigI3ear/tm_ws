"""
TM5-900 Operator UI Server
- Serves the web UI on port 8080
- Proxies /api/snapshot from the Vision image server (port 6189)
- Runs the medicine detection pipeline on /api/scan
- Tracks operator medicine selection for TMflow handoff
- Polls joint angles from ROS2 /joint_states
- Streams operation log via SSE
- Hosts the TMflow TCP bridge on port 6190 (replaces standalone tmflow_server.py)

Run:
  source /opt/ros/jazzy/setup.bash && source ~/tm_ws/install/setup.bash
  ANTHROPIC_API_KEY=sk-... python3 -u operator_ui/server.py
"""
import json
import math
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import image_to_robot, image_to_offset, image_to_binb_offset, image_to_binc_offset, PICK_Z_DOWN, PICK_Z_UP
from vision.claude_vision import detect_all_medicines, scan_qr, classify_medicine, detect_with_yolo

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_server_start_time = time.time()
_state = {
    "robot_connected": False,
    "vision_server_ok": False,
    "tmflow_connected": False,
    "joint_angles": [0.0] * 6,
    "scan_results": [],       # UI-facing — only updated by tmc_pickup scans
    "internal_results": [],   # internal — used by verify/repick/return, not shown in UI
    "scan_mode": "pickup",    # "pickup" updates UI, others use internal_results
    "selected": None,
    "last_pick": None,
}
_log_queue: queue.Queue = queue.Queue()


def add_log(level: str, msg: str):
    entry = {"t": time.strftime("%H:%M:%S"), "lvl": level, "msg": msg}
    with _lock:
        pass  # state log stored in SSE queue only (memory-efficient)
    _log_queue.put(json.dumps(entry))
    print(f"[{level}] {entry['t']} {msg}")


# ---------------------------------------------------------------------------
# Vision / detection helpers
# ---------------------------------------------------------------------------
TM_VISION_MOUNT = "/mnt/c/tm_vision"
SNAPSHOT_URL = "http://localhost:6189/api/snapshot"


def _find_vision_prefix() -> str:
    """Auto-detect the TMflow upload prefix (TM_Export/<serial>/TMROBOT_VisionImages/ActionerResult)."""
    import glob
    matches = glob.glob(f"{TM_VISION_MOUNT}/TM_Export/*/TMROBOT_VisionImages/ActionerResult")
    return matches[0] if matches else TM_VISION_MOUNT


def _get_frame(tm_path: str = "") -> np.ndarray | None:
    if tm_path:
        import glob, os
        local = tm_path.replace("\\", "/")
        filename = os.path.basename(local)
        prefix = _find_vision_prefix()
        # Retry for up to 10s — WSL2 drvfs cache can delay visibility of new files
        for attempt in range(20):
            # Try exact path first
            for base in [prefix, TM_VISION_MOUNT]:
                full = f"{base}/{local}"
                frame = cv2.imread(full)
                if frame is not None:
                    add_log("INFO", f"Loaded image: {full}")
                    return frame
            # Fall back to most recently modified PNG anywhere in the share
            candidates = sorted(
                glob.glob(f"{TM_VISION_MOUNT}/**/*.png", recursive=True),
                key=os.path.getmtime, reverse=True
            )
            if candidates:
                frame = cv2.imread(candidates[0])
                if frame is not None:
                    add_log("INFO", f"Loaded latest image: {candidates[0]}")
                    return frame
            if attempt < 19:
                time.sleep(0.5)
    try:
        resp = requests.get(SNAPSHOT_URL, timeout=8)
        if resp.status_code == 200:
            arr = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def _run_detection(frame: np.ndarray) -> list[dict]:
    qr_hits = scan_qr(frame)
    if qr_hits:
        results = []
        for hit in qr_hits:
            info = classify_medicine(hit["data"])
            ox, oy = image_to_offset(hit["center_x"], hit["center_y"])
            results.append({
                "medicine": info.get("medicine", hit["data"]),
                "description": info.get("description", ""),
                "bin": info.get("bin", "C"),
                "confidence": "high",
                "source": "qr",
                "pick_x": hit["center_x"],
                "pick_y": hit["center_y"],
                "robot_x": ox,
                "robot_y": oy,
            })
        return results

    raw = detect_all_medicines(frame)
    results = []
    for item in raw:
        ox, oy = image_to_offset(item["pick_x"], item["pick_y"])
        results.append({**item, "robot_x": ox, "robot_y": oy})
    return results


def _preload():
    try:
        detect_with_yolo(np.zeros((64, 64, 3), dtype=np.uint8))
        add_log("INFO", "YOLO model loaded")
    except Exception as e:
        add_log("WARN", f"YOLO preload skipped: {e}")


def _image_watcher():
    """Watch for new images (after server start) and run detection, updating the UI."""
    import glob, os
    last_seen = None
    while True:
        try:
            candidates = sorted(
                [f for f in glob.glob(f"{TM_VISION_MOUNT}/**/*.png", recursive=True) if "source" in f],
                key=os.path.getmtime, reverse=True
            )
            if candidates and candidates[0] != last_seen:
                latest = candidates[0]
                frame = cv2.imread(latest)
                if frame is not None:
                    last_seen = latest
                    add_log("INFO", f"New image: {os.path.basename(latest)} — scanning...")
                    results = _run_detection(frame)
                    with _lock:
                        scan_mode = _state.get("scan_mode", "pickup")
                        if scan_mode == "pickup":
                            _state["scan_results"] = results
                            _state["selected"] = None
                        else:
                            _state["internal_results"] = results
                    add_log("INFO", f"Scan done ({scan_mode}): {len(results)} medicine(s) detected")
        except Exception as e:
            add_log("WARN", f"Image watcher error: {e}")
        time.sleep(1)


# ---------------------------------------------------------------------------
# Joint angle poller (background thread)
# ---------------------------------------------------------------------------
def _joint_poller():
    while True:
        try:
            out = subprocess.check_output(
                "source /opt/ros/jazzy/setup.bash && "
                "source /home/asus/tm_ws/install/setup.bash 2>/dev/null && "
                "timeout 3 ros2 topic echo /joint_states --once 2>/dev/null",
                shell=True, executable="/bin/bash", timeout=10
            ).decode()
            positions = re.findall(r"^- (-?\d+\.\d+)", out, re.MULTILINE)
            if len(positions) >= 6:
                degs = [round(math.degrees(float(p)), 2) for p in positions[:6]]
                with _lock:
                    _state["joint_angles"] = degs
                    _state["robot_connected"] = True
        except Exception:
            with _lock:
                _state["robot_connected"] = False
        time.sleep(2)


# ---------------------------------------------------------------------------
# Vision server health checker
# ---------------------------------------------------------------------------
def _vision_checker():
    while True:
        try:
            r = requests.get(SNAPSHOT_URL, timeout=3)
            with _lock:
                _state["vision_server_ok"] = r.status_code in (200, 503)
        except Exception:
            with _lock:
                _state["vision_server_ok"] = False
        time.sleep(5)


# ---------------------------------------------------------------------------
# Flask web server
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static")
CORS(app)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/snapshot")
def snapshot():
    try:
        resp = requests.get(SNAPSHOT_URL, timeout=8)
        if resp.status_code == 200:
            return Response(resp.content, mimetype="image/jpeg")
    except Exception:
        pass
    return jsonify({"error": "no frame"}), 503


@app.route("/api/latest-image")
def latest_image():
    """Serve the most recent Vision image from the VISION share."""
    import glob, os
    candidates = sorted(
        [f for f in glob.glob(f"{TM_VISION_MOUNT}/**/*.png", recursive=True) if "source" in f],
        key=os.path.getmtime, reverse=True
    )
    if not candidates:
        return jsonify({"error": "no image yet"}), 503
    with open(candidates[0], "rb") as f:
        return Response(f.read(), mimetype="image/png")


@app.route("/api/scan", methods=["POST"])
def scan():
    add_log("INFO", "Scan requested by operator")
    frame = _get_frame()
    if frame is None:
        add_log("WARN", "Scan failed — no camera frame available")
        return jsonify({"error": "no frame"}), 503
    try:
        results = _run_detection(frame)
        with _lock:
            _state["scan_results"] = results
            _state["selected"] = None
        add_log("INFO", f"Scan complete — {len(results)} medicine(s) detected")
        return jsonify(results)
    except Exception as e:
        add_log("ERROR", f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/select", methods=["POST"])
def select():
    data = request.json
    with _lock:
        _state["selected"] = data
    add_log("PICK", f"Operator selected: {data.get('medicine')} → ({data.get('robot_x')}, {data.get('robot_y')}) mm — waiting for TMflow trigger")
    return jsonify({"ok": True})


@app.route("/api/status")
def status():
    with _lock:
        return jsonify({
            "robot_connected": _state["robot_connected"],
            "vision_server_ok": _state["vision_server_ok"],
            "tmflow_connected": _state["tmflow_connected"],
            "joint_angles": _state["joint_angles"],
            "scan_results": _state["scan_results"],
            "selected": _state["selected"],
            "last_pick": _state["last_pick"],
        })


@app.route("/api/log-stream")
def log_stream():
    def generate():
        while True:
            try:
                entry = _log_queue.get(timeout=25)
                yield f"data: {entry}\n\n"
            except queue.Empty:
                yield "data: {\"ping\":1}\n\n"
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Verify handler — called after medicine dropped in bin B
# Scans bin B image, checks if correct medicine was placed
# Reply: "0,0" = correct (loop back) | "X,Y" = wrong (pick from B, put in C)
# ---------------------------------------------------------------------------
def _handle_verify(conn, data):
    add_log("INFO", "Verifying medicine in bin B...")

    with _lock:
        _state["scan_mode"] = "verify"
        _state["internal_results"] = []
        last_pick = _state.get("last_pick")

    for _ in range(60):
        with _lock:
            results = _state["internal_results"]
        if results:
            break
        time.sleep(0.5)

    with _lock:
        results = _state["internal_results"]
        _state["scan_mode"] = "pickup"

    if not results:
        add_log("WARN", "Verify: nothing detected — assuming correct")
        conn.sendall(b"0,0")
        return

    with _lock:
        last_pick = _state.get("last_pick")

    detected = results[0]["medicine"]
    expected = last_pick["medicine"] if last_pick else None

    def _same_med(a, b):
        from difflib import SequenceMatcher
        a, b = a.lower(), b.lower()
        # Direct substring check
        if a in b or b in a:
            return True
        # Any word overlap (ignoring dosage words)
        skip = {"mg", "ml", "tab", "cap", "syrup", "solution"}
        wa = {w.strip("()[]") for w in a.split() if len(w.strip("()[]")) > 3 and w.strip("()[]") not in skip}
        wb = {w.strip("()[]") for w in b.split() if len(w.strip("()[]")) > 3 and w.strip("()[]") not in skip}
        if wa & wb:
            return True
        # Fuzzy similarity fallback
        return SequenceMatcher(None, a, b).ratio() > 0.6

    if expected and not _same_med(detected, expected):
        # Wrong medicine — send its position so robot can pick it from B
        ox, oy = image_to_binb_offset(results[0]["pick_x"], results[0]["pick_y"])
        reply = f"{ox},{oy}"
        add_log("WARN", f"Verify FAILED: expected {expected}, got {detected} → pick from B at {reply}")
    else:
        reply = "0,0"
        add_log("INFO", f"Verify OK: {detected} ✓")

    conn.sendall(reply.encode("utf-8"))
    add_log("SEND", f"→ TMflow verify: {reply}")


# ---------------------------------------------------------------------------
# Repick handler — auto re-pick the same medicine after wrong pick dropped in C
# Scans basket A, finds the medicine from last_pick, returns its offset
# ---------------------------------------------------------------------------
def _handle_return(conn, data):
    """Scan bin C, find wrong medicine, return its offset so robot can pick it back to A."""
    add_log("INFO", "Return: scanning bin C for wrong medicine...")

    with _lock:
        _state["scan_mode"] = "return"
        _state["internal_results"] = []

    for _ in range(60):
        with _lock:
            results = _state["internal_results"]
        if results:
            break
        time.sleep(0.5)

    with _lock:
        results = _state["internal_results"]
        _state["scan_mode"] = "pickup"

    if not results:
        add_log("WARN", "Return: nothing detected in bin C — sending 0,0")
        conn.sendall(b"0,0")
        return

    item = results[0]
    ox, oy = image_to_binc_offset(item["pick_x"], item["pick_y"])
    reply = f"{ox},{oy}"
    add_log("INFO", f"Return: {item['medicine']} at {reply} — picking back to A")
    conn.sendall(reply.encode("utf-8"))
    add_log("SEND", f"→ TMflow return: {reply}")


def _handle_repick(conn, data):
    add_log("INFO", "Repick: waiting for fresh scan of basket A...")

    with _lock:
        _state["scan_mode"] = "repick"
        _state["internal_results"] = []
        last_pick = _state.get("last_pick")

    for _ in range(60):
        with _lock:
            results = _state["internal_results"]
        if results:
            break
        time.sleep(0.5)

    with _lock:
        results = _state["internal_results"]
        _state["scan_mode"] = "pickup"

    if not results:
        add_log("WARN", "Repick: nothing detected in basket A — sending 0,0")
        conn.sendall(b"0,0")
        return

    # Find the medicine we originally wanted, fall back to first detected
    expected = last_pick["medicine"] if last_pick else None
    match = next((r for r in results if expected and r["medicine"].lower() == expected.lower()), results[0])

    ox, oy = match["robot_x"], match["robot_y"]
    reply = f"{ox},{oy}"
    add_log("PICK", f"Repick: found {match['medicine']} → {reply}")
    conn.sendall(reply.encode("utf-8"))
    add_log("SEND", f"→ TMflow repick: {reply}")


# ---------------------------------------------------------------------------
# TMflow TCP bridge (port 6190) — runs in background thread
# ---------------------------------------------------------------------------
def _handle_tmflow(conn, addr):
    try:
        with _lock:
            _state["tmflow_connected"] = True
        add_log("INFO", f"TMflow connected from {addr}")
        while True:
            data = conn.recv(4096).decode("utf-8", errors="ignore").strip()
            if not data:
                add_log("INFO", f"TMflow disconnected from {addr}")
                break  # connection closed by TMflow

            add_log("RECV", f"TMflow: {data[:100]}")

            if data.startswith("tmc_verify"):
                _handle_verify(conn, data)
                continue

            if data.startswith("tmc_repick"):
                _handle_repick(conn, data)
                continue

            if data.startswith("tmc_return"):
                _handle_return(conn, data)
                continue

            if not data.startswith("tmc_pickup"):
                continue

            # Clear old results and wait for image watcher to scan the new image
            with _lock:
                _state["scan_results"] = []
            add_log("INFO", "Waiting for fresh scan...")
            for _ in range(60):  # wait up to 30s
                with _lock:
                    results = _state["scan_results"]
                if results:
                    break
                time.sleep(0.5)

            with _lock:
                results = _state["scan_results"]

            if not results:
                add_log("WARN", "No medicines detected — skipping pick")
                continue

            # Print menu to terminal (UI selection also works)
            print("\n" + "="*50)
            print("MEDICINES DETECTED — select in UI or type number:")
            for i, item in enumerate(results):
                print(f"  [{i+1}] {item['medicine']}  →  ({item['robot_x']}, {item['robot_y']}) mm")
            print("  [0] Skip")
            print("="*50)

            # Clear any previous UI selection
            with _lock:
                _state["selected"] = None

            choice = None
            while choice is None:
                with _lock:
                    ui_pick = _state["selected"]
                if ui_pick:
                    choice = ui_pick
                    print(f"  UI selected: {choice['medicine']}")
                    break
                time.sleep(0.3)

            if choice is None:
                continue

            x, y = choice["robot_x"], choice["robot_y"]
            reply = f"{x},{y}"
            with _lock:
                _state["last_pick"] = {
                    "medicine": choice["medicine"],
                    "x": x, "y": y,
                    "time": time.strftime("%H:%M:%S"),
                }
            conn.sendall(reply.encode("utf-8"))
            add_log("PICK", f"Sent: {choice['medicine']} → {reply}")
    except Exception as e:
        add_log("ERROR", f"TMflow connection error: {e}")
    finally:
        conn.close()
        with _lock:
            _state["tmflow_connected"] = False


def _tmflow_tcp_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 6190))
    s.listen(5)
    add_log("INFO", "TMflow bridge listening on port 6190")
    while True:
        conn, addr = s.accept()
        t = threading.Thread(target=_handle_tmflow, args=(conn, addr), daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _preload()

    for fn, name in [
        (_joint_poller, "joint-poller"),
        (_vision_checker, "vision-checker"),
        (_tmflow_tcp_server, "tmflow-bridge"),
        (_image_watcher, "image-watcher"),
    ]:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()

    add_log("INFO", "Operator UI starting on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True, use_reloader=False)
