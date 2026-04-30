# TM5-900 Robot Project — Team Guide

## Robot
- Model: Techman TM5-900
- ROS2 Distro: Jazzy
- Robot IP: 192.168.1.102
- Robot port (status): 5891
- Image pipeline port: 6189

## Starting the Robot Stack
```bash
# From the WSL2 machine (Linux, connected to robot via Ethernet)
~/tm_ws/tm_start.sh 192.168.1.102
```
This opens a tmux session (`tm5900`) with 3 windows:
- `moveit` — MoveIt2 + RViz + tm_driver
- `joints` — live joint state monitor
- `shell` — free terminal

---

## Team Roles

### Person 1 — Robot Driver & Integration (WSL2, GTX 1660 Ti)

**You own the real robot connection and system integration.**

#### Files you work on:
- `~/tm_ws/tm_start.sh` — startup script
- `tmr_ros2/tm_driver/` — hardware driver (ethernet slave + listen node)
- `tmr_ros2/tm_move_group/launch/tm5-900_run_move_group.launch.py` — main launch file

#### Your tasks:
1. Keep the robot connection stable (ethernet slave + listen node)
2. Run integration tests on the real robot when Person 2 or Person 3 finishes a feature
3. Monitor joint states: `ros2 topic echo /joint_states`
4. Own the RViz + MoveIt2 session (GPU-accelerated rendering)

#### Useful commands:
```bash
# Start everything
~/tm_ws/tm_start.sh 192.168.1.102

# Check if robot is reachable
ping 192.168.1.102

# Watch joint states
ros2 topic echo /joint_states

# List all active nodes
ros2 node list
```

---

### Person 2 — Motion Planning & Simulation (Good PC, strong GPU)

**You own how the robot moves. Work offline in simulation — no real robot needed.**

#### Files you work on:
- `tmr_ros2/tm5-900_moveit_config/config/` — MoveIt2 config files
  - `joint_limits.yaml` — min/max joint speeds and positions
  - `ompl_planning.yaml` — motion planner settings
  - `kinematics.yaml` — IK solver config
- `tmr_ros2/tm_moveit_cpp_demo/` — C++ motion planning demos

#### Your tasks:
1. Tune MoveIt2 config for smooth, safe motion
2. Write C++ motion planning programs (pick → place sequences)
3. Build the simulation environment (collision objects, workspace boundaries)
4. Run the demo launch (no real robot needed):

```bash
# Fake hardware simulation — works without robot
ros2 launch tm5-900_moveit_config demo.launch.py

# Build after changes
cd ~/tm_ws && colcon build --packages-select tm5-900_moveit_config tm_moveit_cpp_demo
```

#### Motion planning example structure:
```cpp
// In tm_moveit_cpp_demo/
// Set a target pose → plan → execute
move_group.setPoseTarget(target_pose);
move_group.plan(plan);
move_group.execute(plan);
```

---

### Person 3 — Data Layer & Commands (Mac, no GPU needed)

**You own the software logic. Everything you build runs headlessly — no RViz, no GPU required.**

#### Files you work on:
- `tmr_ros2/techman_robot_get_status/tm_get_status/get_status.py` — robot status reader
- `tmr_ros2/techman_robot_get_status/tm_get_status/translate_jason_to_list.py` — `$TMSVR` protocol parser
- `tmr_ros2/techman_robot_get_status/tm_get_status/image_pub.py` — camera image publisher (Flask server)
- `tmr_ros2/custom_package/src/send_command.cpp` — robot command client

#### Your tasks:

**1. Fix hardcoded IP in `get_status.py` (line 43)**
```python
# Current (bad):
ip = "192.168.132.242"

# Change to accept robot IP as argument:
ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
```

**2. Publish robot status as ROS2 topic**
- `get_status.py` currently just `print()`s the data
- Make it publish `Joint_Angle`, `Coord_Base_Tool`, `Robot_Link` as ROS2 topics

**3. Add unit tests to `translate_jason_to_list.py`**
- 10 test cases are already defined at the bottom of the file
- Formalize them with `pytest`

**4. Make `send_command.cpp` accept CLI arguments**
```cpp
// Current (hardcoded):
request->command = "MOVE_JOG";
request->command_parameter_string = "0,0,90,0,90,0";

// Change to: accept argv[1] and argv[2]
```

**5. Replace `fake_result()` in `image_pub.py` with real inference**
- Flask server receives images at `POST /api/CLS` or `POST /api/DET`
- Hook up a real model (classification or detection)

#### How to test on Mac (no GPU, no RViz):
```bash
# Source ROS2
source /opt/ros/jazzy/setup.bash
source ~/tm_ws/install/setup.bash

# Run status reader (connect to robot)
ros2 run techman_robot_get_status get_status 192.168.1.102

# Call send_command service
ros2 service call /tm_send_command \
  techman_robot_msgs/srv/TechmanRobotCommand \
  "{command: 'MOVE_JOG', command_parameter_string: '0,0,90,0,90,0'}"

# Check topics
ros2 topic list
ros2 topic echo /joint_states

# Run unit tests
cd tmr_ros2/techman_robot_get_status
pytest
```

---

---

## Digital Twin — Mac Vision Pipeline (added 2026-04-30)

Runs entirely on Mac, no ROS required. Purpose: iPhone camera scans QR codes on medicine tablets → Claude classifies the medicine → TM5-900 picks and places it into the correct bin.

### Stack
- Python 3.13 + venv at `Digital Twin/venv/`
- `anthropic` + `opencv-python` + `pyzbar` + `Pillow`
- Camera: iPhone via Continuity Camera (index 1), or pass `--image photo.jpg`
- Robot control: direct TCP → TMflow Listen Node (port **5890**)

### Key files
| File | Purpose |
|------|---------|
| `main.py` | Orchestrator: capture → QR scan → classify → arm motion |
| `vision/claude_vision.py` | Stage 1: pyzbar QR decode · Stage 2: Claude medicine classification |
| `arm_comms/tm5_connect.py` | TMSCT protocol over TCP to TM5-900 |
| `requirements.txt` | Python dependencies |

### Bin assignments
| Bin | Category | Robot position (mm) |
|-----|----------|-------------------|
| A | Common / OTC | (500, 150) |
| B | Prescription | (500, 0) |
| C | Controlled / Unknown | (500, −150) |

### Running
```bash
cd "Digital Twin"
source venv/bin/activate

# Test vision only (no arm)
python main.py --dry-run

# Use a saved image instead of camera
python main.py --image photo.jpg --dry-run

# Live run (arm must be reachable)
TM5_IP=192.168.1.102 python main.py
```

### TMflow setup required on robot
- Port **5890** = TMSCT (Listen Node commands)
- Port **5891** = TMSVR (status only — do NOT send commands here)
- TMflow project must be **running** with a Listen Node as the active node
- Recommended flowchart: `Start → Listen1 → (Pass loops back to Listen1)`
- WaitFor nodes between Listen loops can block command reception — avoid them or keep timeout = 0

### Environment variables (set in `~/.zshrc`)
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
export TM5_IP=192.168.1.102
```

### Workspace calibration (update in `main.py` once physically set up)
```python
TABLE_X_MIN = 300   # mm — left edge of camera FOV in robot frame
TABLE_X_MAX = 600   # mm — right edge
TABLE_Y_MIN = -200  # mm — far from robot
TABLE_Y_MAX =  200  # mm — near robot
PICK_Z_DOWN =  50   # mm — descent height for grasp
PICK_Z_UP   = 200   # mm — travel height
```

### Status (as of 2026-04-30)
- Vision pipeline (QR → Claude classify) working end-to-end on Mac
- TCP connection to TM5-900 at 192.168.1.102 established
- **Pending**: Listen Node (port 5890) not yet accepting commands — TMflow project must loop on Listen1

---

## Project Architecture

```
Person 3 (Mac)                    Person 2 (Good PC)
  get_status.py                     tm5-900_moveit_config/
  translate_jason_to_list.py   -->  tm_moveit_cpp_demo/
  send_command.cpp                  demo.launch.py (simulation)
  image_pub.py                      RViz + collision scene
        |                                   |
        +------------- Person 1 -----------+
                    (WSL2 + GTX 1660 Ti)
                      Real TM5-900 robot
                      tm_driver (ethernet slave + listen node)
                      Integration testing
```

## Build Commands
```bash
cd ~/tm_ws

# Build everything
colcon build

# Build specific package
colcon build --packages-select custom_package

# Source after build
source install/setup.bash
```
