from controller import Robot, Camera
import base64, os, sys, json, time
import anthropic

robot = Robot()
timestep = int(robot.getBasicTimeStep())

joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
motors = [robot.getDevice(j) for j in joints]
sensors = [robot.getDevice(f"{j}_sensor") for j in joints]

for s in sensors:
    s.enable(timestep)

camera = robot.getDevice("wrist_camera")
camera.enable(timestep * 4)

# Home position (all joints at 0)
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Scan position — tilted to look at the table
SCAN = [0.0, -0.5, 0.8, 0.0, -0.3, 0.0]

def move_to(positions, wait_steps=100):
    for motor, pos in zip(motors, positions):
        motor.setPosition(pos)
    for _ in range(wait_steps):
        robot.step(timestep)

def capture_and_analyze():
    img = camera.getImage()
    if img is None:
        return None

    width, height = camera.getWidth(), camera.getHeight()
    # Convert raw BGRA bytes to JPEG via PIL if available, else save raw
    try:
        from PIL import Image as PILImage
        import io
        raw = bytes([camera.imageGetRed(img, width, x, y)
                     for y in range(height) for x in range(width)] +
                    [0])  # placeholder — see note below
        # Proper pixel extraction
        pixels = []
        for y in range(height):
            for x in range(width):
                r = camera.imageGetRed(img, width, x, y)
                g = camera.imageGetGreen(img, width, x, y)
                b = camera.imageGetBlue(img, width, x, y)
                pixels.extend([r, g, b])
        pil_img = PILImage.frombytes("RGB", (width, height), bytes(pixels))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Fallback: use raw BGRA from Webots and encode directly
        img_b64 = base64.b64encode(img).decode()

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "You are the vision system for a TM5-900 robot arm. "
                        "List every object you see on the table. "
                        "For each object output JSON: "
                        "{\"name\": str, \"shape\": str, \"color\": str, \"position\": \"left|center|right\"}. "
                        "Return a JSON array only, no other text."
                    )
                }
            ]
        }]
    )

    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        print("[vision] raw response:", response.content[0].text)
        return None

# --- Main loop ---
print("[tm5] Moving to home position...")
move_to(HOME, wait_steps=50)

print("[tm5] Moving to scan position...")
move_to(SCAN, wait_steps=150)

print("[tm5] Capturing image and sending to Claude...")
objects = capture_and_analyze()
if objects:
    print("[vision] Detected objects:")
    for obj in objects:
        print(f"  - {obj['color']} {obj['shape']} ({obj['name']}) → {obj['position']}")
else:
    print("[vision] No objects detected or Claude API not configured.")

print("[tm5] Returning to home...")
move_to(HOME, wait_steps=150)

while robot.step(timestep) != -1:
    pass
