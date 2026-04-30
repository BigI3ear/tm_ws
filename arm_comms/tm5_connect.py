"""
TCP communication with TM5-900 via TMflow Listen Node (port 5890).
Uses the TMSCT protocol: $TMSCT,<len>,<id>,<script>,*<checksum>\r\n
Port 5890 = TMSCT (Listen Node commands)
Port 5891 = TMSVR (status monitoring only)
"""
import socket
import os
import time

TM5_IP   = os.environ.get("TM5_IP", "192.168.1.102")
TM5_PORT = int(os.environ.get("TM5_PORT", 5890))
TIMEOUT  = 5.0

_msg_id = 0


def _checksum(data: str) -> str:
    chk = 0
    for c in data:
        chk ^= ord(c)
    return format(chk, "02X")


def _build_packet(script: str) -> bytes:
    global _msg_id
    _msg_id += 1
    id_str = str(_msg_id)
    payload = f"{id_str},{script}"
    length = len(payload)
    body = f"TMSCT,{length},{payload},"
    chk = _checksum(body)
    packet = f"${body}*{chk}\r\n"
    return packet.encode("ascii")


class TM5:
    def __init__(self, ip: str = TM5_IP, port: int = TM5_PORT):
        self.ip   = ip
        self.port = port

    def _send(self, script: str) -> str:
        packet = _build_packet(script)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(TIMEOUT)
            s.connect((self.ip, self.port))
            s.sendall(packet)
            time.sleep(0.1)
            try:
                return s.recv(4096).decode("utf-8", errors="ignore").strip()
            except socket.timeout:
                return ""

    # --- Motion ---

    def move_joints(self, j1, j2, j3, j4, j5, j6,
                    speed: int = 10, blend: int = 0) -> str:
        script = f'PTP("JPP",{j1},{j2},{j3},{j4},{j5},{j6},{speed},{blend},0,false)'
        return self._send(script)

    def move_cartesian(self, x, y, z, rx, ry, rz,
                       speed: int = 200, blend: int = 0) -> str:
        script = f'PTP("CPP",{x},{y},{z},{rx},{ry},{rz},{speed},{blend},0,false)'
        return self._send(script)

    def home(self) -> str:
        return self.move_joints(0, 0, 90, 0, 90, 0, speed=15)

    def scan_pose(self) -> str:
        return self.move_joints(0, -30, 100, 0, 20, 0, speed=12)

    def gripper_open(self) -> str:
        return self._send('IO.SetDO("DO_0",false)')

    def gripper_close(self) -> str:
        return self._send('IO.SetDO("DO_0",true)')

    def ping(self) -> bool:
        try:
            resp = self._send("QueueTag(1)")
            return len(resp) > 0
        except (ConnectionRefusedError, OSError):
            return False


if __name__ == "__main__":
    arm = TM5()
    print(f"Connecting to TM5-900 at {arm.ip}:{arm.port} ...")
    if arm.ping():
        print("Connected. Moving to home...")
        resp = arm.home()
        print("Response:", resp if resp else "(no response — normal if arm is moving)")
    else:
        print("Could not reach robot. Check TM5_IP and that Listen Node is active in TMflow.")
