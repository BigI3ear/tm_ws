"""
TM5-900 command interface via ROS2 /send_script service.
The tm_driver node owns port 5890 — all commands go through the driver.
Port 5891 = TMSVR (status monitoring only, not used here)
"""
import os
import subprocess
import time

_msg_id = 0


_ROS2_SETUP = (
    "source /opt/ros/jazzy/setup.bash && "
    "source /home/asus/tm_ws/install/setup.bash 2>/dev/null ; "
)


def _ros2_send(script: str) -> bool:
    """Send a TMSCT script via the ROS2 /send_script service. Returns True if ok=True."""
    global _msg_id
    _msg_id += 1
    # Escape single quotes inside the script
    safe_script = script.replace("'", "\\'")
    bash_cmd = (
        _ROS2_SETUP +
        f"ros2 service call /send_script tm_msgs/srv/SendScript "
        f"\"{{id: '{_msg_id}', script: '{safe_script}'}}\""
    )
    try:
        result = subprocess.run(
            ["bash", "-c", bash_cmd],
            capture_output=True, text=True, timeout=15
        )
        return "ok=True" in result.stdout
    except subprocess.TimeoutExpired:
        return False


class TM5:
    def __init__(self):
        pass

    def _send(self, script: str) -> bool:
        return _ros2_send(script)

    # --- Motion ---

    def move_joints(self, j1, j2, j3, j4, j5, j6,
                    speed: int = 10, blend: int = 0) -> bool:
        script = f'PTP("JPP",{j1},{j2},{j3},{j4},{j5},{j6},{speed},{blend},0,false)'
        return self._send(script)

    def move_cartesian(self, x, y, z, rx, ry, rz,
                       speed: int = 200, blend: int = 0) -> bool:
        script = f'PTP("CPP",{x},{y},{z},{rx},{ry},{rz},{speed},{blend},0,false)'
        return self._send(script)

    def home(self) -> bool:
        """Scan pose — camera centred over basket. This is the default home."""
        return self.move_joints(-83.23, 4.14, -102.90, 8.56, -90.34, 4.97, speed=12)

    def scan_pose(self) -> bool:
        return self.home()

    def capture(self, vision_job: str = "Vision1", timeout_ms: int = 5000) -> bool:
        """Trigger Vision1 to capture one frame and POST it to the Flask image server."""
        script = f'Vision.GetVisionResult("{vision_job}","0","",{timeout_ms})'
        return self._send(script)

    def suction_on(self) -> bool:
        """Activate vacuum — picks up medicine."""
        return self._send('IO.SetDO("DO_0",true)')

    def suction_off(self) -> bool:
        """Deactivate vacuum — releases medicine."""
        return self._send('IO.SetDO("DO_0",false)')

    def ping(self) -> bool:
        return _ros2_send("QueueTag(1)")


if __name__ == "__main__":
    arm = TM5()
    print("Pinging robot via /send_script ...")
    if arm.ping():
        print("Connected. Moving to home...")
        ok = arm.home()
        print("ok:", ok)
    else:
        print("Could not reach robot. Check that tm_driver and Listen Node are running.")
