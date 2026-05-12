#!/usr/bin/env python3
"""Send pick sequence to TM5-900 via ROS2 send_script service."""
import rclpy
from rclpy.node import Node
from tm_msgs.srv import SendScript
import time
import sys

STEPS = [
    ("safe_up",  'PTP("CPP",-167.64,549.72,250.0,180.0,0.0,-178.0,100,0,0,false)'),
    ("pick_down",'PTP("CPP",-167.64,549.72,187.37,180.0,0.0,-178.0,50,0,0,false)'),
    ("home",     'PTP("JPP",-83.06,4.14,-102.90,8.56,-90.34,4.97,8,0,0,false)'),
]

class PickRunner(Node):
    def __init__(self):
        super().__init__('pick_runner')
        self.cli = self.create_client(SendScript, '/send_script')
        self.get_logger().info('Waiting for /send_script service...')
        self.cli.wait_for_service(timeout_sec=10.0)
        self.get_logger().info('Service ready.')

    def send(self, script_id, script, retries=30):
        req = SendScript.Request()
        req.id = script_id
        req.script = script
        for attempt in range(retries):
            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.result() and future.result().ok:
                self.get_logger().info(f'[{script_id}] ok=True')
                return True
            self.get_logger().warn(f'[{script_id}] attempt {attempt+1} failed, retrying...')
            time.sleep(0.5)
        self.get_logger().error(f'[{script_id}] FAILED after {retries} attempts')
        return False


def main():
    rclpy.init()
    node = PickRunner()
    steps = STEPS
    if len(sys.argv) > 1 and sys.argv[1] == 'home':
        steps = [("home", 'PTP("JPP",-83.06,4.14,-102.90,8.56,-90.34,4.97,8,0,0,false)')]

    for step_id, script in steps:
        print(f'\n--- {step_id} ---')
        ok = node.send(step_id, script)
        if not ok:
            print(f'Step {step_id} failed — aborting.')
            break
        wait = 8 if 'CPP' in script else 6
        print(f'Waiting {wait}s for motion...')
        time.sleep(wait)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
