#!/usr/bin/env python3
"""Run a pick sequence on the TM5-900 using the TM5 class."""
import sys
import time

sys.path.insert(0, '/home/asus/tm_ws/src')
from arm_comms.tm5_connect import TM5

TCP_RX, TCP_RY, TCP_RZ = 180.0, 0.0, -178.0
BASKET_X, BASKET_Y     = -167.64, 549.72
PICK_Z_UP, PICK_Z_DOWN = 250.0, 187.37


def main():
    arm = TM5()

    if len(sys.argv) > 1 and sys.argv[1] == 'home':
        print('Moving to home...')
        print('ok:', arm.home())
        return

    print('--- safe_up ---')
    ok = arm.move_cartesian(BASKET_X, BASKET_Y, PICK_Z_UP, TCP_RX, TCP_RY, TCP_RZ, speed=0.3)
    print('ok:', ok)
    if not ok:
        print('FAILED'); return
    time.sleep(4)

    print('--- pick_down ---')
    ok = arm.move_cartesian(BASKET_X, BASKET_Y, PICK_Z_DOWN, TCP_RX, TCP_RY, TCP_RZ, speed=0.05)
    print('ok:', ok)
    if not ok:
        print('FAILED'); return
    time.sleep(5)

    print('--- home ---')
    ok = arm.home()
    print('ok:', ok)
    time.sleep(4)


if __name__ == '__main__':
    main()
