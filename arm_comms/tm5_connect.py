"""
TM5-900 command interface via ROS2 /set_positions and /set_io services.
Motion uses TMSVR direct control — bypasses the Listen Node entirely.
/send_script (TMSCT) returns ok=True for motion but does not execute motion.
"""
import math
import time
import rclpy
from rclpy.node import Node
from tm_msgs.srv import SetPositions, SetIO

_node: Node | None = None
_pos_cli = None
_io_cli = None


def _ensure_node():
    global _node, _pos_cli, _io_cli
    if _node is not None:
        return
    if not rclpy.ok():
        rclpy.init()
    _node = Node('tm5_client')
    _pos_cli = _node.create_client(SetPositions, '/set_positions')
    _io_cli  = _node.create_client(SetIO, '/set_io')
    _pos_cli.wait_for_service(timeout_sec=10.0)
    _io_cli.wait_for_service(timeout_sec=10.0)


def _move(motion_type, positions, velocity, fine=False, retries=30) -> bool:
    _ensure_node()
    req = SetPositions.Request()
    req.motion_type      = motion_type
    req.positions        = list(positions)
    req.velocity         = float(velocity)
    req.acc_time         = 200.0
    req.blend_percentage = 0
    req.fine_goal        = fine
    for _ in range(retries):
        fut = _pos_cli.call_async(req)
        rclpy.spin_until_future_complete(_node, fut, timeout_sec=5.0)
        if fut.result() and fut.result().ok:
            return True
        time.sleep(0.5)
    return False


def _set_io(pin: int, state: int) -> bool:
    _ensure_node()
    req = SetIO.Request()
    req.module = SetIO.Request.MODULE_CONTROLBOX
    req.type   = SetIO.Request.TYPE_DIGITAL_OUT
    req.pin    = pin
    req.state  = float(state)
    fut = _io_cli.call_async(req)
    rclpy.spin_until_future_complete(_node, fut, timeout_sec=5.0)
    return fut.result().ok if fut.result() else False


class TM5:
    def __init__(self):
        _ensure_node()

    # --- Motion ---

    def move_joints(self, j1, j2, j3, j4, j5, j6, speed: float = 0.5) -> bool:
        """Joint PTP. Angles in degrees, speed in rad/s."""
        positions = [math.radians(a) for a in (j1, j2, j3, j4, j5, j6)]
        return _move(SetPositions.Request.PTP_J, positions, velocity=speed)

    def move_cartesian(self, x, y, z, rx, ry, rz, speed: float = 0.2) -> bool:
        """Cartesian PTP. x/y/z in mm, rx/ry/rz in degrees, speed in m/s."""
        positions = [x / 1000, y / 1000, z / 1000,
                     math.radians(rx), math.radians(ry), math.radians(rz)]
        return _move(SetPositions.Request.PTP_T, positions, velocity=speed)

    def home(self) -> bool:
        """Scan pose — camera centred over basket."""
        return self.move_joints(-83.06, 4.14, -102.90, 8.56, -90.34, 4.97, speed=0.5)

    def scan_pose(self) -> bool:
        return self.home()

    def capture(self, vision_job: str = "Vision1", timeout_ms: int = 5000) -> bool:
        """Trigger Vision1 via /send_script (non-motion command)."""
        import subprocess
        script = f'Vision.GetVisionResult("{vision_job}","0","",{timeout_ms})'
        safe = script.replace("'", "\\'")
        bash_cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            "source /home/asus/tm_ws/install/setup.bash 2>/dev/null ; "
            f"ros2 service call /send_script tm_msgs/srv/SendScript "
            f"\"{{id: 'capture', script: '{safe}'}}\""
        )
        try:
            result = subprocess.run(["bash", "-c", bash_cmd],
                                    capture_output=True, text=True, timeout=15)
            return "ok=True" in result.stdout
        except subprocess.TimeoutExpired:
            return False

    # --- IO ---

    def suction_on(self) -> bool:
        return _set_io(0, 1)

    def suction_off(self) -> bool:
        return _set_io(0, 0)

    # --- Status ---

    def ping(self) -> bool:
        try:
            _ensure_node()
            return _pos_cli.service_is_ready()
        except Exception:
            return False


if __name__ == "__main__":
    arm = TM5()
    print("Pinging robot...")
    if arm.ping():
        print("Connected. Moving to home...")
        print("ok:", arm.home())
    else:
        print("Could not reach robot. Check that tm_driver is running.")
