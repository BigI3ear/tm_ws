"""
Vision module — QR code scanner for medicine tablets.

Stage 1: pyzbar decodes QR code from camera frame (free, instant, no API call)
Stage 2: Claude identifies the medicine and decides which bin it goes to
"""
import anthropic
import base64
import json
import sys
from pathlib import Path

import cv2
from pyzbar.pyzbar import decode as qr_decode


def capture_frame(camera_index: int = 1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    return frame


def scan_qr(frame) -> list[dict]:
    """
    Decode all QR codes in a frame.
    Returns list of dicts: {data, center_x, center_y} where center_x/y are normalized 0-1.
    """
    h, w = frame.shape[:2]
    results = []
    for obj in qr_decode(frame):
        data = obj.data.decode("utf-8")
        pts = obj.polygon
        cx = sum(p.x for p in pts) / len(pts) / w
        cy = sum(p.y for p in pts) / len(pts) / h
        results.append({"data": data, "center_x": round(cx, 3), "center_y": round(cy, 3)})
    return results


def classify_medicine(medicine_code: str) -> dict:
    """
    Ask Claude what this medicine is and where to put it.
    Returns: {medicine, description, bin, action}
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"A robot arm scanned a QR code on a medicine tablet. The QR code says: '{medicine_code}'.\n"
                "Return a JSON object with:\n"
                "  medicine: full medicine name\n"
                "  description: one-line description (dosage, type)\n"
                "  bin: which bin to place it in — 'A' (common/OTC), 'B' (prescription), 'C' (controlled/unknown)\n"
                "  action: 'pick_and_place'\n"
                "Return ONLY valid JSON, no markdown."
            )
        }]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def analyze(frame_or_path) -> list[dict]:
    """
    Full pipeline: decode QR → classify medicine via Claude.
    Returns list of dicts ready for main.py to act on.
    """
    if isinstance(frame_or_path, str):
        frame = cv2.imread(frame_or_path)
    else:
        frame = frame_or_path

    qr_hits = scan_qr(frame)
    if not qr_hits:
        return []

    results = []
    for hit in qr_hits:
        info = classify_medicine(hit["data"])
        results.append({
            "qr_data":   hit["data"],
            "medicine":  info.get("medicine", hit["data"]),
            "description": info.get("description", ""),
            "bin":       info.get("bin", "A"),
            "action":    info.get("action", "pick_and_place"),
            "pick_x":    hit["center_x"],
            "pick_y":    hit["center_y"],
        })
    return results


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        src = sys.argv[1]
    else:
        print("Capturing from iPhone camera...")
        src = capture_frame()

    results = analyze(src)
    if not results:
        print("No QR codes detected.")
    else:
        for r in results:
            print(f"[{r['bin']}] {r['medicine']} — {r['description']}")
            print(f"     QR: {r['qr_data']}")
            print(f"     Position: x={r['pick_x']}, y={r['pick_y']}")
