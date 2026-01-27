# BoxBunny ROS 2 Humble Boxing Trainer

Target platform: Jetson Orin NX 16GB (Ubuntu 22.04, ROS 2 Humble).

This workspace provides:
- RealSense D435i glove tracking (HSV + depth, optional pose verification)
- Reaction-time drill with countdown, baseline capture, and logging
- Punch prediction via IMU + vision fusion (jab/cross/hook/uppercut)
- GUI for drills, punch stats, and slow‑mo replay
- Local LLM coach/trash/encourage/analysis modes
- Rolling punch analytics topic for feedback

## Workspace Layout

- `src/boxbunny_msgs` – ROS 2 message/service definitions
- `src/boxbunny_vision` – RealSense glove tracker
- `src/boxbunny_fusion` – punch fusion (vision + IMU)
- `src/boxbunny_drills` – reaction drill manager + logging
- `src/boxbunny_imu` – MPU6050 node + IMU punch classifier + calibration service
- `src/boxbunny_analytics` – rolling punch stats
- `src/boxbunny_llm` – local LLM coach node + `GenerateLLM` service
- `src/boxbunny_gui` – PySide6 GUI
- `src/boxbunny_bringup` – launch files + system configuration
- `notebooks/` – test notebooks
- `notebooks/boxbunny_all_in_one.ipynb` – detailed guided tests
- `notebooks/ros_commands.ipynb` – runnable ROS commands
- `models/` – downloaded LLM + pose models
- `docs/` – architecture + customization

## Quick Start (Humble)

```bash
cd /home/boxbunny/Desktop/doomsday_integration/boxbunny_ws
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

- `/camera/color/image_raw` – RealSense RGB
- `/glove_detections` – glove boxes + distance + velocity
- `/punch_events_raw` – vision-only punches (from tracker)
- `/imu/punch` – IMU punch classification
- `/punch_events` – fused punches (vision + IMU)
- `/punch_stats` – rolling JSON summary (counts, avg velocity)
- `/drill_state` + `/drill_countdown` + `/drill_events` – reaction drill status
- `/trash_talk` – LLM coach output

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

- `notebooks/boxbunny_all_in_one.ipynb` – camera + IMU + LLM + ROS smoke tests
- `notebooks/ros_commands.ipynb` – ROS 2 command reference

## Documentation

- `docs/architecture.md` – node graph + data flow
- `docs/customization.md` – model paths + toggles + tuning

## Notes

- The glove tracker uses HSV segmentation by default; optional pose verification can be enabled in `glove_tracker.yaml`.
- For lighter CPU usage, lower `camera_fps`, reduce `resize_scale`, or increase `process_every_n`.
- If IMU is off, fusion will pass through vision punches.
