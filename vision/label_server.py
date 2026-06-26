"""
Quick data collection server for benchmark labeling.
- Watches /mnt/c/tm_vision for new images
- Shows each image with a form to label it
- Saves image + JSON to test_images/<scenario>/

Run: python3 vision/label_server.py
Open: http://192.168.1.XXX:8081
"""
import glob, json, os, shutil, socket, sys, threading, time
from pathlib import Path
from flask import Flask, Response, request, jsonify, send_file
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# TMflow TCP bridge — holds connection until user clicks NEXT, then replies
# ---------------------------------------------------------------------------
_tmflow_conn = None      # current open socket to TMflow
_tmflow_lock = threading.Lock()
_next_ready   = threading.Event()  # set when user clicks NEXT

def _tmflow_server():
    global _tmflow_conn
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 6190))
    s.listen(5)
    print("TMflow bridge listening on port 6190")
    while True:
        conn, addr = s.accept()
        print(f"TMflow connected from {addr}")
        with _tmflow_lock:
            _tmflow_conn = conn
        try:
            while True:
                data = conn.recv(4096).decode("utf-8", errors="ignore").strip()
                if not data:
                    break
                print(f"TMflow: {data[:80]}")
                # Wait for user to click NEXT before replying
                _next_ready.wait()
                _next_ready.clear()
                conn.sendall(b"next\r\n")
                print("Sent: next → TMflow will loop back to Vision")
        except Exception as e:
            print(f"TMflow error: {e}")
        finally:
            with _tmflow_lock:
                _tmflow_conn = None
            conn.close()

VISION_MOUNT = "/mnt/c/tm_vision"
SAVE_DIR = Path(__file__).parent.parent.parent / "test_images"

SCENARIOS = ["baseline", "position", "occlusion", "density_high", "confusable", "orientation"]

KNOWN_MEDICINES = [
    "Betadine",
    "Gentian Violet",
    "Leopard Cough Syrup",
    "Siribuncha Alcohol",
    "Ya That Nam Khao White Rabbit",
    "Yoki Balm",
    "Januvia 100mg",
    "Entresto 200mg",
    "Samsca 15mg",
    "Serlift 50mg",
    "Valosine 75mg",
    "Calcitriol 0.25mcg",
    "Clopidogrel 75mg",
    "Kremil",
]

app = Flask(__name__)

def latest_image():
    candidates = sorted(
        [f for f in glob.glob(f"{VISION_MOUNT}/**/*.png", recursive=True) if "source" in f],
        key=os.path.getmtime, reverse=True
    )
    return candidates[0] if candidates else None

def count_saved():
    counts = {}
    for s in SCENARIOS:
        folder = SAVE_DIR / s
        counts[s] = len(list(folder.glob("*.png"))) if folder.exists() else 0
    return counts

@app.route("/api/randomize")
def randomize():
    import random
    n = int(request.args.get("n", 4))
    n = max(1, min(n, len(KNOWN_MEDICINES)))
    selected = random.sample(KNOWN_MEDICINES, n)
    return jsonify({"medicines": selected})


@app.route("/")
def index():
    counts = count_saved()
    total = sum(counts.values())
    img = latest_image()
    img_html = '<p style="color:#888;text-align:center;padding:40px">No image yet — trigger TMflow Vision</p>' if not img else \
               f'<div><img id="basket-img" src="/latest?t={int(time.time())}" style="max-width:100%;border:2px solid #00c2d4;border-radius:4px"><br><button onclick="refreshImg()" style="margin-top:6px;background:#222;color:#00c2d4;border:1px solid #00c2d4;padding:4px 12px;cursor:pointer;border-radius:3px;font-size:11px">↻ REFRESH IMAGE</button></div>'

    scenario_options = "\n".join(
        f'<option value="{s}">{s} ({counts.get(s,0)}/10)</option>' for s in SCENARIOS
    )
    med_list_json = json.dumps(KNOWN_MEDICINES)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Data Collector</title>
<style>
  body {{ background:#06080a; color:#e0e0e0; font-family:monospace; margin:0; padding:20px; }}
  h1 {{ color:#e89010; margin:0 0 8px; }}
  .container {{ display:grid; grid-template-columns:1fr 360px; gap:20px; max-width:1300px; }}
  .form-group {{ margin:10px 0; }}
  .lbl {{ color:#00c2d4; font-size:11px; display:block; margin-bottom:3px; }}
  select, input[type=text], input[type=number] {{ background:#111; color:#e0e0e0; border:1px solid #333; padding:6px; width:100%; border-radius:4px; box-sizing:border-box; }}
  .btn-main {{ background:#e89010; color:#000; border:none; padding:10px; font-weight:bold; cursor:pointer; border-radius:4px; font-size:13px; width:100%; margin-top:6px; }}
  .btn-cyan  {{ background:#00c2d4; color:#000; border:none; padding:10px; font-weight:bold; cursor:pointer; border-radius:4px; font-size:13px; width:100%; margin-top:6px; }}
  .btn-rand  {{ background:#333; color:#e89010; border:1px solid #e89010; padding:8px; font-weight:bold; cursor:pointer; border-radius:4px; font-size:13px; width:100%; margin-top:6px; }}
  .med-list  {{ background:#111; border:1px solid #333; border-radius:4px; padding:10px; min-height:80px; }}
  .med-tag   {{ display:inline-block; background:#1a2a1a; color:#00ff88; border:1px solid #00aa44; border-radius:3px; padding:2px 8px; margin:2px; font-size:12px; cursor:pointer; }}
  .med-tag:hover {{ background:#ff3333; border-color:#ff3333; color:#fff; }}
  .progress  {{ color:#00c2d4; font-size:12px; margin-bottom:12px; }}
  .msg       {{ display:none; color:#00ff88; padding:6px; background:#001a0a; border-radius:4px; margin-top:6px; font-size:12px; }}
  .hint      {{ color:#666; font-size:11px; margin-top:3px; }}
</style>
</head>
<body>
<h1>TM5-900 Data Collector</h1>
<div class="progress">Total saved: {total}/60 &nbsp;|&nbsp; {' &nbsp; '.join(f"{s}:{counts.get(s,0)}" for s in SCENARIOS)}</div>
<div class="container">
  <div>{img_html}</div>
  <div>
    <div class="form-group">
      <span class="lbl">SCENARIO</span>
      <select id="scenario" onchange="onScenarioChange(this)">{scenario_options}</select>
    </div>

    <div class="form-group">
      <span class="lbl">HOW MANY MEDICINES</span>
      <input type="number" id="num-meds" value="4" min="1" max="{len(KNOWN_MEDICINES)}" style="width:80px">
      <button class="btn-rand" onclick="randomize()">🎲 RANDOMIZE SELECTION</button>
      <div class="hint">Click a medicine tag to remove it</div>
    </div>

    <div class="form-group">
      <span class="lbl">PUT THESE IN BASKET:</span>
      <div class="med-list" id="med-list"><span style="color:#555">Press RANDOMIZE to generate</span></div>
      <div style="margin-top:6px">
        <span class="lbl">CLICK TO ADD:</span>
        <div style="background:#0a0a0a;border:1px solid #222;border-radius:4px;padding:6px">
          {''.join(f'<span onclick="addMed(this)" data-med="{m}" style="display:inline-block;background:#1a1a2e;color:#888;border:1px solid #333;border-radius:3px;padding:2px 7px;margin:2px;font-size:11px;cursor:pointer" onmouseover="this.style.borderColor=\'#e89010\';this.style.color=\'#e89010\'" onmouseout="this.style.borderColor=\'#333\';this.style.color=\'#888\'">{m}</span>' for m in KNOWN_MEDICINES)}
        </div>
      </div>
    </div>

    <div class="form-group">
      <span class="lbl">OCCLUSION</span>
      <select id="occlusion">
        <option value="none">None</option>
        <option value="plastic">Clear plastic wrap</option>
        <option value="covered">Covered by another med</option>
      </select>
    </div>
    <div class="form-group">
      <span class="lbl">ORIENTATION</span>
      <select id="orientation">
        <option value="normal">Normal (label up)</option>
        <option value="tilted">Tilted</option>
        <option value="upside_down">Upside down</option>
      </select>
    </div>
    <div class="form-group">
      <span class="lbl">NOTES (optional)</span>
      <input type="text" id="notes" placeholder="Any extra notes">
    </div>

    <button class="btn-main" onclick="save()">SAVE IMAGE</button>
    <button class="btn-cyan"  onclick="nextShot()">NEXT → TAKE NEW PHOTO</button>
    <div class="msg" id="msg"></div>
  </div>
</div>
<script>
let currentMeds = [];

// Restore saved state on load
window.addEventListener('load', () => {{
  const savedScenario = localStorage.getItem('scenario');
  if (savedScenario) {{
    const sel = document.getElementById('scenario');
    if (sel) sel.value = savedScenario;
  }}
  const savedMeds = localStorage.getItem('currentMeds');
  if (savedMeds) {{
    currentMeds = JSON.parse(savedMeds);
    renderMeds();
  }}
}});

function onScenarioChange(el) {{
  localStorage.setItem('scenario', el.value);
}}

async function randomize() {{
  const n = document.getElementById('num-meds').value;
  const r = await fetch(`/api/randomize?n=${{n}}`);
  const data = await r.json();
  currentMeds = data.medicines;
  renderMeds();
}}

function renderMeds() {{
  localStorage.setItem('currentMeds', JSON.stringify(currentMeds));
  const list = document.getElementById('med-list');
  if (!currentMeds.length) {{
    list.innerHTML = '<span style="color:#555">Press RANDOMIZE to generate</span>';
    return;
  }}
  list.innerHTML = currentMeds.map((m,i) =>
    `<span class="med-tag" onclick="removeMed(${{i}})" title="Click to remove">${{m}} ✕</span>`
  ).join('');
}}

function removeMed(i) {{
  currentMeds.splice(i, 1);
  renderMeds();
}}

function addMed(el) {{
  const val = el.dataset.med;
  if (val && !currentMeds.includes(val)) {{
    currentMeds.push(val);
    renderMeds();
  }}
}}

function refreshImg() {{
  const img = document.getElementById('basket-img');
  if (img) img.src = '/latest?t=' + Date.now();
}}

async function nextShot() {{
  await fetch('/next', {{method:'POST'}});
  const msg = document.getElementById('msg');
  msg.style.display = 'block';
  msg.textContent = '📷 Triggered — waiting for new image...';
  setTimeout(() => location.reload(), 4000);
}}

async function save() {{
  if (!currentMeds.length) {{
    alert('Press RANDOMIZE first to select medicines');
    return;
  }}
  const payload = {{
    scenario:    document.getElementById('scenario').value,
    medicines:   currentMeds,
    lighting:    'normal',
    occlusion:   document.getElementById('occlusion').value,
    orientation: document.getElementById('orientation').value,
    notes:       document.getElementById('notes').value || '',
  }};
  const r = await fetch('/save', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload)}});
  const res = await r.json();
  const msg = document.getElementById('msg');
  msg.style.display = 'block';
  msg.textContent = res.ok ? '✓ Saved: ' + res.path : '✗ Error: ' + res.error;
  if (res.ok) {{ localStorage.removeItem('currentMeds'); setTimeout(() => location.reload(), 1500); }}
}}
</script>
</body>
</html>"""

@app.route("/api/detect")
def detect():
    img = latest_image()
    if not img:
        return jsonify({"medicines": []})
    try:
        from claude_vision import detect_all_medicines
        frame = cv2.imread(img)
        results = detect_all_medicines(frame)
        names = [r["medicine"] for r in results]
        return jsonify({"medicines": names})
    except Exception as e:
        return jsonify({"medicines": [], "error": str(e)})


@app.route("/next", methods=["POST"])
def next_shot():
    _next_ready.set()
    return jsonify({"ok": True})

@app.route("/latest")
def latest():
    img = latest_image()
    if not img:
        return "no image", 404
    return send_file(img, mimetype="image/png")

@app.route("/save", methods=["POST"])
def save():
    data = request.json
    scenario = data.get("scenario", "baseline")
    medicines = data.get("medicines", [])
    if not medicines:
        return jsonify({"ok": False, "error": "No medicines selected"})

    img = latest_image()
    if not img:
        return jsonify({"ok": False, "error": "No image available"})

    folder = SAVE_DIR / scenario
    folder.mkdir(parents=True, exist_ok=True)

    existing = sorted(folder.glob("shot_*.png"))
    n = len(existing) + 1
    name = f"shot_{n:02d}"

    dst_img = folder / f"{name}.png"
    dst_json = folder / f"{name}.json"

    shutil.copy2(img, dst_img)

    gt = {
        "medicines":   medicines,
        "lighting":    data.get("lighting", "normal"),
        "occlusion":   data.get("occlusion", "none"),
        "density":     len(medicines),
        "orientation": data.get("orientation", "normal"),
        "notes":       data.get("notes", ""),
    }
    with open(dst_json, "w") as f:
        json.dump(gt, f, indent=2)

    return jsonify({"ok": True, "path": str(dst_img)})

if __name__ == "__main__":
    t = threading.Thread(target=_tmflow_server, daemon=True)
    t.start()
    print(f"Label server on http://0.0.0.0:8081")
    print(f"Images saved to: {SAVE_DIR}")
    app.run(host="0.0.0.0", port=8081, debug=False, use_reloader=False)
