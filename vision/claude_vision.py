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
YOLO_WEIGHTS = Path(__file__).parent.parent.parent / "dataset" / "yolo" / "runs" / "thai_medicine_v1-4" / "weights" / "best.pt"

_BIN_MAP = {
    "Betadine":                    "A",
    "Gentian_Violet":              "A",
    "Leopard_Cough_Syrup":         "A",
    "Siribuncha_Alcohol":          "A",
    "Ya_That_Nam_Khao_White_Rabbit": "A",
}


def detect_with_yolo(frame: np.ndarray) -> list[dict] | None:
    """
    Run YOLOv8 on frame. Returns list of detections (one per class, highest conf wins)
    or None if weights not found.
    """
    if not YOLO_WEIGHTS.exists():
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(YOLO_WEIGHTS))
        H, W = frame.shape[:2]
        results = model(frame, conf=0.45, iou=0.6, imgsz=1280, verbose=False)[0]

        # Keep only highest-confidence detection per class
        best = {}
        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if cls not in best or conf > best[cls]["conf"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best[cls] = {
                    "conf": conf,
                    "label": model.names[cls],
                    "pick_x": round(((x1 + x2) / 2) / W, 3),
                    "pick_y": round(((y1 + y2) / 2) / H, 3),
                }

        detections = []
        for cls, item in sorted(best.items()):
            label = item["label"]
            detections.append({
                "source":      "yolo",
                "qr_data":     None,
                "medicine":    label.replace("_", " "),
                "label":       label,
                "description": "",
                "bin":         _BIN_MAP.get(label, "A"),
                "action":      "pick_and_place",
                "pick_x":      item["pick_x"],
                "pick_y":      item["pick_y"],
                "confidence":  "high" if item["conf"] >= 0.6 else "medium" if item["conf"] >= 0.4 else "low",
            })
        return detections
    except Exception as e:
        print(f"  YOLO error: {e}")
        return None


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


_BIN_RULES = """\
Bin assignment rules:
  A = Common / OTC — available without prescription, e.g.:
      Paracetamol, Ibuprofen, antacids, vitamins, cough syrups (OTC),
      Thai herbal/traditional medicines (ยาสมุนไพร, ยาธาตุ, ยาแผนโบราณ),
      Ya That Nam Khao (ยาธาตุน้ำขาว), Yoki, Krabok, rehydration salts,
      topical creams, eye drops (OTC), allergy tablets (OTC).
  B = Prescription — requires a doctor's prescription, e.g.:
      Statins, ACE inhibitors, ARBs, SSRIs, antidiabetics, antihypertensives,
      Entresto, Januvia, Serlift, Valosine, Samsca, antibiotics, antivirals,
      Thai hospital dispensary bags (ถุงยาโรงพยาบาล) with patient HN/name printed.
  C = Controlled / Unknown — narcotics, psychotropics, or unreadable label.\
"""


def classify_medicine(medicine_code: str) -> dict:
    """
    Ask Claude what this medicine is and where to put it.
    Returns: {medicine, label, description, bin, action}
    """
    client = anthropic.Anthropic(timeout=30.0)
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"A robot arm scanned a QR code on a medicine tablet. The QR code says: '{medicine_code}'.\n"
                f"{_BIN_RULES}\n"
                "Return a JSON object with:\n"
                "  medicine: full medicine name\n"
                "  label: short consistent dataset folder name, e.g. 'Entresto_200mg' or 'Paracetamol_500mg' (no spaces, no special chars except underscore)\n"
                "  description: one-line description (dosage, type)\n"
                "  bin: which bin to place it in — 'A', 'B', or 'C' per the rules above\n"
                "  action: 'pick_and_place'\n"
                "Return ONLY valid JSON, no markdown."
            )
        }]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def _prepare_image(frame) -> bytes:
    """Validate, normalise channels, resize, and JPEG-encode a frame for the Claude API."""
    if frame is None or frame.size == 0:
        raise RuntimeError("Invalid frame: None or empty")
    h, w = frame.shape[:2]
    if h < 10 or w < 10:
        raise RuntimeError(f"Frame too small: {w}x{h}")

    # Float frames from some pipelines produce corrupt JPEG — always encode as uint8
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Normalise to 3-channel BGR
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # np.frombuffer / non-contiguous slices can cause silent OpenCV encoding failures
    frame = np.ascontiguousarray(frame)

    # Cap longest edge at 1568 px (Claude recommended vision limit)
    MAX_DIM = 1568
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Encode; reduce quality if result exceeds 4 MB
    # A valid JPEG is at least a few hundred bytes; anything smaller is corrupt
    MAX_BYTES = 4 * 1024 * 1024
    MIN_BYTES = 100
    for quality in (85, 70, 50, 35):
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ret and buf is not None and MIN_BYTES <= len(buf) <= MAX_BYTES:
            return buf.tobytes()
    raise RuntimeError("Could not encode frame within 4 MB limit")


def detect_all_medicines(frame) -> list[dict]:
    """
    Detection pipeline:
      1. YOLO (fast, local) — detects known Thai medicines, one per class, no duplicates
      2. Claude API — identifies remaining/unknown items (English meds, new items)
      3. CLIP — second opinion for Claude low-confidence Thai items
    Returns list of {medicine, label, description, bin, confidence, pick_x, pick_y, source}
    """
    # Stage 1: YOLO for known Thai medicines
    yolo_results = detect_with_yolo(frame) or []
    if yolo_results:
        print(f"  YOLO detected {len(yolo_results)} Thai medicines")

    # Stage 2: Claude for English/unknown meds not covered by YOLO
    print("  Calling Claude for English/unknown medicines...")
    img_bytes = _prepare_image(frame)
    img_b64 = base64.b64encode(img_bytes).decode()

    client = anthropic.Anthropic(timeout=30.0)
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
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
                        "You are a medicine detection system looking at a robot camera image of a basket. "
                        "Your job is to find and report EVERY physical medicine object in the frame — "
                        "do NOT skip anything just because it is partially covered, in a plastic bag, "
                        "shows only a barcode, or the label is hard to read.\n\n"
                        "IMPORTANT — canonical names for known Thai medicines. If you identify any of these, "
                        "you MUST use exactly this medicine name and label:\n"
                        "  - Betadine antiseptic (yellow/brown bottle) → medicine='Betadine', label='Betadine'\n"
                        "  - Gentian Violet (small dark purple bottle) → medicine='Gentian Violet', label='Gentian_Violet'\n"
                        "  - Leopard cough syrup (orange hexagonal bottle, Thai label) → medicine='Leopard Cough Syrup', label='Leopard_Cough_Syrup'\n"
                        "  - Siribuncha isopropyl/rubbing alcohol (blue spray bottle) → medicine='Siribuncha Alcohol', label='Siribuncha_Alcohol'\n"
                        "  - Ya That Nam Khao White Rabbit / ยาธาตุน้ำขาว (white tube/bottle, rabbit logo) → medicine='Ya That Nam Khao White Rabbit', label='Ya_That_Nam_Khao_White_Rabbit'\n\n"
                        + (lambda yr: (
                            "NOTE: These Thai medicines have ALREADY been identified by a local model — "
                            "DO NOT report them again: " + ", ".join(r["medicine"] for r in yr) + ". "
                            "Skip any object at or near their positions: " +
                            ", ".join("({},{})".format(r["pick_x"], r["pick_y"]) for r in yr) +
                            ".\n\n"
                        ) if yr else "")(yolo_results) +
                        "Scan the image systematically: top-left → top-right → middle → bottom-left → bottom-right. "
                        "For each distinct physical object (bottle, tube, blister pack, sachet, bag) NOT already listed above: "
                        "report it even if you can only see part of the label or packaging. "
                        "Use color, shape, and partial text to identify it. "
                        "If you cannot read the label at all, still report it as 'Unknown' with your best "
                        "description of what you see (color, shape, size).\n\n"
                        "Labels may be in Thai (ภาษาไทย) or English — read whichever is visible.\n"
                        f"{_BIN_RULES}\n"
                        "Return a JSON array. Each element:\n"
                        "  medicine: full medicine name (or 'Unknown' if unreadable)\n"
                        "  label: short folder-safe name e.g. 'Betadine_Solution' (underscores, no spaces)\n"
                        "  description: one-line description including color and shape if label is unclear\n"
                        "  bin: 'A', 'B', or 'C' per rules above (use 'C' if unknown)\n"
                        "  confidence: 'high', 'medium', or 'low'\n"
                        "  cx: horizontal centre of this medicine as fraction 0.0–1.0 (left=0, right=1)\n"
                        "  cy: vertical centre as fraction 0.0–1.0 (top=0, bottom=1)\n"
                        "Return ONLY a valid JSON array, no markdown, no extra text. "
                        "Return [] only if the basket is completely empty or all items are already listed above."
                    ),
                },
            ],
        }]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    items = json.loads(raw)
    if not isinstance(items, list):
        items = [items]

    # CLIP second-opinion for low-confidence items
    h, w = frame.shape[:2]
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from thai_visual import match as clip_match, DB_PATH
        clip_available = DB_PATH.exists()
    except Exception:
        clip_available = False

    results = []
    for item in items:
        cx = float(item.get("cx", 0.5))
        cy = float(item.get("cy", 0.5))
        confidence = item.get("confidence", "low")

        if clip_available and confidence in ("low", "medium"):
            # Crop a region around the detected medicine centre for CLIP
            crop_w, crop_h = int(w * 0.35), int(h * 0.35)
            x1 = max(0, int(cx * w) - crop_w // 2)
            y1 = max(0, int(cy * h) - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)
            crop = frame[y1:y2, x1:x2]
            clip_result = clip_match(crop)
            if clip_result and clip_result["score"] >= 0.75:
                item["medicine"] = clip_result["medicine"]
                item["label"] = clip_result["label"]
                item["confidence"] = clip_result["confidence"]
                item["source"] = "clip"
                item["bin"] = _BIN_MAP.get(clip_result["label"], item.get("bin", "A"))
            else:
                item["source"] = "vision"
        else:
            item["source"] = "vision" if not item.get("source") else item["source"]

        lbl = item.get("label", "Unknown")
        # Skip if YOLO already identified this Thai medicine
        if any(r["label"] == lbl for r in yolo_results):
            continue

        # If Claude identified a known Thai medicine, enforce correct bin
        bin_id = _BIN_MAP.get(lbl, item.get("bin", "C"))
        results.append({
            "source":      item.get("source", "vision"),
            "qr_data":     None,
            "medicine":    item.get("medicine", "Unknown"),
            "label":       lbl,
            "description": item.get("description", ""),
            "bin":         bin_id,
            "action":      "pick_and_place",
            "pick_x":      round(cx, 3),
            "pick_y":      round(cy, 3),
            "confidence":  item.get("confidence", "low"),
        })

    # Merge: YOLO Thai meds + Claude English/unknown meds
    return yolo_results + results


def classify_from_image(frame) -> dict:
    """
    No QR found — send the image to Claude vision to visually identify the medicine.
    Returns: {medicine, description, bin, action}
    """
    img_bytes = _prepare_image(frame)
    img_b64 = base64.b64encode(img_bytes).decode()

    client = anthropic.Anthropic(timeout=30.0)
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
                        "No QR code was detected. Identify the medicine visually. "
                        "The label may be in Thai (ภาษาไทย) or English — read whichever is present.\n"
                        f"{_BIN_RULES}\n"
                        "Return a JSON object with:\n"
                        "  medicine: full medicine name (or 'Unknown' if unreadable)\n"
                        "  label: short consistent dataset folder name, e.g. 'Entresto_200mg' or 'Paracetamol_500mg' (no spaces, no special chars except underscore)\n"
                        "  description: one-line description (dosage, type)\n"
                        "  bin: 'A', 'B', or 'C' per the rules above\n"
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
        if frame is None:
            raise RuntimeError(f"Could not load image: {frame_or_path}")
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
