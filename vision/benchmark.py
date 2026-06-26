"""
Detection Model Benchmark
Compares YOLO-only, Claude-only, and YOLO+Claude on labeled test images
across multiple real-world conditions.

Ground truth JSON format per image:
{
  "medicines": ["Betadine", "Januvia 100mg"],
  "lighting":    "normal" | "dim" | "bright" | "shadow",
  "occlusion":   "none" | "plastic" | "covered",
  "density":     1-10  (number of medicines),
  "orientation": "normal" | "tilted" | "upside_down",
  "notes":       "any extra notes"
}

Scenario folder names (one folder per test condition):
  baseline / position / occlusion / density_high / confusable / orientation

Usage:
  python3 benchmark.py --dir test_images/

Output: benchmark_results.csv + detailed summary table
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_vision import detect_with_yolo, detect_all_medicines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_match(detected: str, truth: str) -> bool:
    """Fuzzy name match — same as server verify logic."""
    from difflib import SequenceMatcher
    a, b = detected.lower(), truth.lower()
    if a in b or b in a:
        return True
    skip = {"mg", "ml", "tab", "cap", "syrup", "solution"}
    wa = {w.strip("()[]") for w in a.split() if len(w.strip("()[]")) > 3 and w not in skip}
    wb = {w.strip("()[]") for w in b.split() if len(w.strip("()[]")) > 3 and w not in skip}
    if wa & wb:
        return True
    return SequenceMatcher(None, a, b).ratio() > 0.6


def detect_yolo_only(frame) -> list[str]:
    results = detect_with_yolo(frame) or []
    return [r["medicine"] for r in results]


def detect_claude_only(frame) -> list[str]:
    """Claude-only: temporarily suppress YOLO by patching detect_with_yolo."""
    import claude_vision as cv
    orig = cv.detect_with_yolo
    cv.detect_with_yolo = lambda f: []
    try:
        results = detect_all_medicines(frame)
    finally:
        cv.detect_with_yolo = orig
    return [r["medicine"] for r in results]


def detect_combined(frame) -> list[str]:
    results = detect_all_medicines(frame)
    return [r["medicine"] for r in results]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(detected: list[str], ground_truth: list[str]) -> dict:
    """
    Returns:
      tp: correctly detected medicines
      fp: detected but not in ground truth (false positives)
      fn: in ground truth but not detected (missed)
      detection_rate: tp / len(ground_truth)
      precision: tp / (tp + fp)
      f1: harmonic mean of precision and recall
    """
    matched_truth = set()
    matched_det = set()

    for i, d in enumerate(detected):
        for j, t in enumerate(ground_truth):
            if j not in matched_truth and _name_match(d, t):
                matched_truth.add(j)
                matched_det.add(i)
                break

    tp = len(matched_truth)
    fp = len(detected) - len(matched_det)
    fn = len(ground_truth) - tp

    detection_rate = tp / len(ground_truth) if ground_truth else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = detection_rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "detection_rate": round(detection_rate * 100, 1),
        "precision": round(precision * 100, 1),
        "f1": round(f1 * 100, 1),
        "detected": detected,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(root_dir: str, output_csv: str = "benchmark_results.csv"):
    root = Path(root_dir)
    image_files = sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg"))

    rows = []
    scenario_stats = {}

    for img_path in image_files:
        gt_path = img_path.with_suffix(".json")
        if not gt_path.exists():
            print(f"  SKIP {img_path.name} — no ground truth JSON")
            continue

        with open(gt_path) as f:
            gt = json.load(f)
        ground_truth = gt.get("medicines", [])
        scenario    = img_path.parent.name
        lighting    = gt.get("lighting", "normal")
        occlusion   = gt.get("occlusion", "none")
        density     = gt.get("density", len(ground_truth))
        orientation = gt.get("orientation", "normal")
        notes       = gt.get("notes", "")

        print(f"\n{'='*60}")
        print(f"Image: {img_path.relative_to(root)}")
        print(f"Ground truth: {ground_truth}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print("  ERROR: could not read image")
            continue

        for mode, fn in [
            ("yolo_only",   detect_yolo_only),
            ("claude_only", detect_claude_only),
            ("combined",    detect_combined),
        ]:
            print(f"  Running {mode}...", end=" ", flush=True)
            t0 = time.time()
            try:
                detected = fn(frame)
            except Exception as e:
                print(f"ERROR: {e}")
                detected = []
            elapsed = round(time.time() - t0, 1)
            s = score(detected, ground_truth)
            print(f"{elapsed}s → detected: {detected}")
            print(f"    DR={s['detection_rate']}%  Precision={s['precision']}%  F1={s['f1']}%  FP={s['fp']}")

            rows.append({
                "scenario":    scenario,
                "image":       img_path.name,
                "mode":        mode,
                "lighting":    lighting,
                "occlusion":   occlusion,
                "density":     density,
                "orientation": orientation,
                "notes":       notes,
                "ground_truth": "|".join(ground_truth),
                "detected":    "|".join(s["detected"]),
                "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                "detection_rate": s["detection_rate"],
                "precision":   s["precision"],
                "f1":          s["f1"],
                "time_s":      elapsed,
            })

            if scenario not in scenario_stats:
                scenario_stats[scenario] = {"yolo_only": [], "claude_only": [], "combined": []}
            scenario_stats[scenario][mode].append(s)

    # Write CSV
    if rows:
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults saved to {output_csv}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Scenario':<15} {'Mode':<15} {'Det.Rate%':>10} {'Precision%':>11} {'F1%':>6} {'FP/img':>8}")
    print(f"{'-'*70}")

    for scenario, modes in sorted(scenario_stats.items()):
        for mode, scores in modes.items():
            if not scores:
                continue
            avg_dr = round(sum(s["detection_rate"] for s in scores) / len(scores), 1)
            avg_pr = round(sum(s["precision"] for s in scores) / len(scores), 1)
            avg_f1 = round(sum(s["f1"] for s in scores) / len(scores), 1)
            avg_fp = round(sum(s["fp"] for s in scores) / len(scores), 1)
            print(f"{scenario:<15} {mode:<15} {avg_dr:>10} {avg_pr:>11} {avg_f1:>6} {avg_fp:>8}")
        print(f"{'-'*70}")

    print(f"\nOverall across all scenarios:")
    for mode in ["yolo_only", "claude_only", "combined"]:
        all_scores = [s for modes in scenario_stats.values() for s in modes.get(mode, [])]
        if not all_scores:
            continue
        avg_dr = round(sum(s["detection_rate"] for s in all_scores) / len(all_scores), 1)
        avg_pr = round(sum(s["precision"] for s in all_scores) / len(all_scores), 1)
        avg_f1 = round(sum(s["f1"] for s in all_scores) / len(all_scores), 1)
        avg_fp = round(sum(s["fp"] for s in all_scores) / len(all_scores), 1)
        print(f"  {mode:<15} DR={avg_dr}%  Precision={avg_pr}%  F1={avg_f1}%  FP/img={avg_fp}")

    # Breakdown by condition factor
    if rows:
        print(f"\n{'='*70}")
        print("CONDITION IMPACT (combined mode only — F1 score)")
        print(f"{'='*70}")
        combined_rows = [r for r in rows if r["mode"] == "combined"]
        for factor in ["lighting", "occlusion", "orientation"]:
            vals = {}
            for r in combined_rows:
                v = r[factor]
                if v not in vals:
                    vals[v] = []
                vals[v].append(r["f1"])
            if vals:
                print(f"\n  {factor.upper()}:")
                for v, scores in sorted(vals.items()):
                    avg = round(sum(scores) / len(scores), 1)
                    print(f"    {v:<20} F1={avg}%  (n={len(scores)})")

        print(f"\n  DENSITY vs F1 (combined):")
        density_vals = {}
        for r in combined_rows:
            d = r["density"]
            if d not in density_vals:
                density_vals[d] = []
            density_vals[d].append(r["f1"])
        for d, scores in sorted(density_vals.items(), key=lambda x: int(x[0])):
            avg = round(sum(scores) / len(scores), 1)
            print(f"    {d} medicines{'':<10} F1={avg}%  (n={len(scores)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Root folder containing scenario subfolders")
    parser.add_argument("--output", default="benchmark_results.csv")
    args = parser.parse_args()
    run_benchmark(args.dir, args.output)
