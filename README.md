# BoxBunny ROS 2 Humble Boxing Trainer

Target platform: Jetson Orin NX 16GB (Ubuntu 22.04, ROS 2 Humble).

This workspace provides:
- RealSense D435i glove tracking (HSV + depth, optional pose verification)
- **Action recognition** using RGBD model (jab/cross/hook/uppercut/block detection)
- **3 Drill Modes:** Reaction Time, Shadow Sparring, Defence
- Punch prediction via IMU + vision fusion
- Unified GUI with 6 tabs for all training modes
- Local LLM coach with performance analysis
- Optional IMU input for menu navigation

## Workspace Layout

- `src/boxbunny_msgs` – ROS 2 message/service definitions
- `src/boxbunny_vision` – RealSense glove tracker + **action predictor node**
- `src/boxbunny_fusion` – punch fusion (vision + IMU)
- `src/boxbunny_drills` – reaction drill + **shadow sparring** + **defence drill**
- `src/boxbunny_imu` – MPU6050 node + punch classifier + **IMU input selector**
- `src/boxbunny_analytics` – rolling punch stats
- `src/boxbunny_llm` – local LLM coach node + `GenerateLLM` service
- `src/boxbunny_gui` – PySide6 GUI (6 tabs)
- `src/boxbunny_bringup` – launch files + system configuration
- `action_prediction/` – RGBD action recognition model
- `notebooks/` – test notebooks
  - `component_testing.ipynb` – detailed testing/calibration
  - `unit_tests.ipynb` – quick tests and GUI launchers
  - `ros_commands.ipynb` – ROS launch reference
- `models/` – LLM + pose models
- `docs/` – architecture + customization

## Quick Start (Humble)

```bash
cd /home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Optional model downloads (Qwen2.5-3B-Instruct GGUF + YOLO26n-pose)
chmod +x download_models.sh
./download_models.sh

# Full system
ros2 launch boxbunny_bringup boxbunny_system.launch.py enable_imu:=true enable_llm:=true enable_gui:=true
```

## Launch Modes

- **Vision-only:**
  ```bash
  ros2 launch boxbunny_bringup vision_only.launch.py
  ```
- **Full system (GUI + drill + LLM + IMU):**
  ```bash
  ros2 launch boxbunny_bringup boxbunny_system.launch.py enable_imu:=true enable_llm:=true enable_gui:=true
  ```
- **LLM off:**
  ```bash
  ros2 launch boxbunny_bringup boxbunny_system.launch.py enable_llm:=false
  ```
- **GUI-only:**
  ```bash
  ros2 launch boxbunny_bringup gui_only.launch.py
  ```
- **LLM-only:**
  ```bash
  ros2 launch boxbunny_bringup llm_only.launch.py
  ```
- **LLM Chat GUI:**
  ```bash
  ros2 run boxbunny_llm llm_chat_gui
  ```
- **IMU-only:**
  ```bash
  ros2 launch boxbunny_bringup imu_only.launch.py enable_classifier:=true
  ```

## Key Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | Image | RealSense RGB |
| `/glove_detections` | GloveDetections | Glove boxes + distance + velocity |
| `/action_prediction` | ActionPrediction | Real-time action classification |
| `/drill_progress` | DrillProgress | Current drill status/progress |
| `/punch_events` | PunchEvent | Fused punches (vision + IMU) |
| `/motor_command` | MotorCommand | Defence drill motor control |
| `/imu_selection` | Int32 | IMU-based menu selections |
| `/trash_talk` | TrashTalk | LLM coach output |

## Reaction Drill Flow

1. **Countdown** – GUI turns yellow and counts down.
2. **Baseline capture** – captures idle movement to avoid false triggers.
3. **Cue** – GUI turns green to signal punch.
4. **Detection** – uses punch events filtered by baseline velocity.

Logs are written to `~/boxbunny_logs`.

Key parameters: `src/boxbunny_drills/config/drill.yaml`.

## IMU Calibration

```bash
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch "{punch_type: 'jab_or_cross', duration_s: 2.5}"
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch "{punch_type: 'hook', duration_s: 2.5}"
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch "{punch_type: 'uppercut', duration_s: 2.5}"
```

Calibration data is stored at `~/.boxbunny/imu_calibration.json`.

## LLM Coach

Prompt via ROS service:

```bash
ros2 service call /llm/generate boxbunny_msgs/srv/GenerateLLM "{prompt: 'Give me a short pep talk', mode: 'encourage', context: 'gui'}"
```

Editable prompt datasets:
- `src/boxbunny_llm/config/persona_examples.yaml`
- `src/boxbunny_llm/config/coach_dataset.yaml`

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `component_testing.ipynb` | Detailed testing for camera, action model, IMU calibration, LLM, drills |
| `unit_tests.ipynb` | Quick tests and GUI launchers |
| `ros_commands.ipynb` | ROS 2 command reference |

## Documentation

- `docs/architecture.md` – node graph + data flow
- `docs/customization.md` – model paths + toggles + tuning

## Notes

- The glove tracker uses HSV segmentation by default; optional pose verification can be enabled in `glove_tracker.yaml`.
- For lighter CPU usage, lower `camera_fps`, reduce `resize_scale`, or increase `process_every_n`.
- If IMU is off, fusion will pass through vision punches.
