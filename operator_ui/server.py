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
from main import image_to_robot, PICK_Z_DOWN, PICK_Z_UP
from vision.claude_vision import detect_all_medicines, scan_qr, classify_medicine, detect_with_yolo

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_state = {
    "robot_connected": False,
    "vision_server_ok": False,
    "tmflow_connected": False,
    "joint_angles": [0.0] * 6,   # degrees
    "scan_results": [],
    "selected": None,             # medicine chosen by operator
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
TM_VISION_MOUNT = "/mnt/tm_vision"
SNAPSHOT_URL = "http://localhost:6189/api/snapshot"


def _get_frame(tm_path: str = "") -> np.ndarray | None:
    if tm_path:
        local = tm_path.replace("\\", "/")
        full = f"{TM_VISION_MOUNT}/{local}"
        frame = cv2.imread(full)
        if frame is not None:
            return frame
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
            rx, ry = image_to_robot(hit["center_x"], hit["center_y"])
            results.append({
                "medicine": info.get("medicine", hit["data"]),
                "description": info.get("description", ""),
                "bin": info.get("bin", "C"),
                "confidence": "high",
                "source": "qr",
                "pick_x": hit["center_x"],
                "pick_y": hit["center_y"],
                "robot_x": rx,
                "robot_y": ry,
            })
        return results

    raw = detect_all_medicines(frame)
    results = []
    for item in raw:
        rx, ry = image_to_robot(item["pick_x"], item["pick_y"])
        results.append({**item, "robot_x": rx, "robot_y": ry})
    return results


def _preload():
    try:
        detect_with_yolo(np.zeros((64, 64, 3), dtype=np.uint8))
        add_log("INFO", "YOLO model loaded")
    except Exception as e:
        add_log("WARN", f"YOLO preload skipped: {e}")


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
# TMflow TCP bridge (port 6190) — runs in background thread
# ---------------------------------------------------------------------------
def _handle_tmflow(conn, addr):
    try:
        with _lock:
            _state["tmflow_connected"] = True
        data = conn.recv(4096).decode("utf-8", errors="ignore").strip()
        add_log("RECV", f"TMflow: {data[:100]}")

        if not data.startswith("tmc_pickup"):
            return

        tm_path = data.split(",", 1)[1] if "," in data else ""

        with _lock:
            selected = _state["selected"]

        if selected:
            x, y = selected["robot_x"], selected["robot_y"]
            add_log("PICK", f"Sending operator pick: {selected['medicine']} → ({x}, {y}) mm")
            with _lock:
                _state["selected"] = None
                _state["last_pick"] = {
                    "medicine": selected["medicine"],
                    "x": x, "y": y,
                    "time": time.strftime("%H:%M:%S"),
                }
        else:
            frame = _get_frame(tm_path)
            if frame is None:
                add_log("WARN", "No frame — sending 0,0")
                conn.sendall(b"0,0")
                return
            results = _run_detection(frame)
            with _lock:
                _state["scan_results"] = results
            if not results:
                add_log("WARN", "No medicine detected — sending 0,0")
                conn.sendall(b"0,0")
                return
            item = results[0]
            x, y = item["robot_x"], item["robot_y"]
            add_log("INFO", f"Auto-pick: {item['medicine']} → ({x}, {y}) mm")
            with _lock:
                _state["last_pick"] = {
                    "medicine": item["medicine"],
                    "x": x, "y": y,
                    "time": time.strftime("%H:%M:%S"),
                }

        reply = f"{x},{y}"
        conn.sendall(reply.encode("utf-8"))
        add_log("SEND", f"→ TMflow: {reply}")
    except Exception as e:
        add_log("ERROR", str(e))
        try:
            conn.sendall(b"0,0")
        except Exception:
            pass
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
    ]:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()

    add_log("INFO", "Operator UI starting on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True, use_reloader=False)
