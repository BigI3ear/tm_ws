"""
Vision module — medicine identification pipeline.

Stage 1: pyzbar decodes QR code from camera frame (free, instant, no API call)
Stage 2a: if QR found → Claude classifies by QR text
Stage 2b: if no QR → Claude visually identifies the medicine from the image
"""
import anthropic
import base64
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pyzbar.pyzbar import decode as qr_decode

ROBOT_CAM_URL = os.environ.get("ROBOT_CAM_URL", "http://localhost:6189/api/snapshot")
DATASET_DIR = Path(os.environ.get("DATASET_DIR", Path(__file__).parent.parent.parent / "dataset" / "snapshots"))


def _medicine_folder(label: str, bin_id: str) -> Path:
    """Return (and create) the folder for this medicine using the short consistent label."""
    safe = label.replace(" ", "_").replace("/", "-").replace("+", "plus")[:50]
    folder = DATASET_DIR / f"bin{bin_id}_{safe}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_to_dataset(frame: np.ndarray, medicine: str, bin_id: str, source: str) -> Path:
    """Save snapshot into medicine-specific subfolder with angle index."""
    folder = _medicine_folder(medicine, bin_id)
    existing = list(folder.glob("*.jpg"))
    idx = len(existing) + 1
    ts = datetime.now().strftime("%H%M%S")
    filename = f"angle{idx:03d}_{ts}_{source}.jpg"
    out = folder / filename
    cv2.imwrite(str(out), frame)
    return out


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


def capture_from_robot_cam(url: str = ROBOT_CAM_URL) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=5) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Robot cam snapshot returned HTTP {resp.status}")
        jpg_bytes = resp.read()
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode robot cam snapshot")
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
    Returns: {medicine, label, description, bin, action}
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
                "  label: short consistent dataset folder name, e.g. 'Entresto_200mg' or 'Paracetamol_500mg' (no spaces, no special chars except underscore)\n"
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


def classify_from_image(frame) -> dict:
    """
    No QR found — send the image to Claude vision to visually identify the medicine.
    Returns: {medicine, description, bin, action}
    """
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        raise RuntimeError("Failed to encode frame for Claude vision")
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                },
                {
                    "type": "text",
                    "text": (
                        "This is an image from a robot arm camera looking at a medicine tablet/blister pack. "
                        "No QR code was detected. Identify the medicine visually.\n"
                        "Return a JSON object with:\n"
                        "  medicine: full medicine name (or 'Unknown' if unreadable)\n"
                        "  label: short consistent dataset folder name, e.g. 'Entresto_200mg' or 'Paracetamol_500mg' (no spaces, no special chars except underscore)\n"
                        "  description: one-line description (dosage, type)\n"
                        "  bin: 'A' (common/OTC), 'B' (prescription), 'C' (controlled/unknown)\n"
                        "  action: 'pick_and_place'\n"
                        "  confidence: 'high', 'medium', or 'low'\n"
                        "Return ONLY valid JSON, no markdown."
                    ),
                },
            ],
        }]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def analyze(frame_or_path) -> list[dict]:
    """
    Full pipeline: try QR decode first, fall back to Claude vision if no QR found.
    Returns list of dicts ready for main.py to act on.
    """
    if isinstance(frame_or_path, str):
        frame = cv2.imread(frame_or_path)
    else:
        frame = frame_or_path

    qr_hits = scan_qr(frame)

    if qr_hits:
        # QR path — fast, accurate
        results = []
        for hit in qr_hits:
            info = classify_medicine(hit["data"])
            result = {
                "source":      "qr",
                "qr_data":     hit["data"],
                "medicine":    info.get("medicine", hit["data"]),
                "label":       info.get("label", info.get("medicine", hit["data"])),
                "description": info.get("description", ""),
                "bin":         info.get("bin", "C"),
                "action":      info.get("action", "pick_and_place"),
                "pick_x":      hit["center_x"],
                "pick_y":      hit["center_y"],
                "confidence":  "high",
            }
            saved = save_to_dataset(frame, result["label"], result["bin"], "qr")
            print(f"  Saved to dataset: {saved.name}")
            results.append(result)
        return results
    else:
        # No QR — use Claude vision on the full image
        print("  No QR code found — using Claude vision to identify medicine...")
        info = classify_from_image(frame)
        result = {
            "source":      "vision",
            "qr_data":     None,
            "medicine":    info.get("medicine", "Unknown"),
            "label":       info.get("label", info.get("medicine", "Unknown")),
            "description": info.get("description", ""),
            "bin":         info.get("bin", "C"),
            "action":      info.get("action", "pick_and_place"),
            "pick_x":      0.5,
            "pick_y":      0.5,
            "confidence":  info.get("confidence", "low"),
        }
        saved = save_to_dataset(frame, result["label"], result["bin"], "vision")
        print(f"  Saved to dataset: {saved.name}")
        return [result]


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
