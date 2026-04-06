# BoxBunny CDE Fair — Boxing Training Robot

> ROS 2 Humble integration for live demonstrations at CDE Fair. Jetson Orin NX 16GB, Ubuntu 22.04.

---

## Overview

BoxBunny is an autonomous boxing training robot combining real-time computer vision, IMU sensing, motor-driven robotic arms, and a local LLM coach. This version is a streamlined integration designed for fair/expo demonstrations.

**Key capabilities:**
- Real-time glove tracking via RealSense D435i (HSV color + depth)
- Optional YOLO Pose + RGBD action recognition (jab, cross, hook, uppercut, block)
- IMU-based punch classification (MPU6050)
- Vision + IMU sensor fusion for confirmed punch events
- 3 training drill modes: Reaction Time, Shadow Sparring, Defence
- Local LLM coach (Qwen 2.5 3B) with real-time feedback
- 4-motor robotic arm control via Teensy 4.1 + micro-ROS
- Unified PySide6 GUI with 6 tabs

---

## Workspace Layout

```
boxing_robot_ws/
  src/
    boxbunny_msgs/          Message & service definitions (PunchEvent, DrillProgress, etc.)
    boxbunny_vision/        RealSense glove tracker + action predictor node
    boxbunny_fusion/        Punch fusion (vision + IMU)
    boxbunny_drills/        Reaction drill, shadow sparring, defence drill
    boxbunny_imu/           MPU6050 driver, punch classifier, IMU menu input
    boxbunny_analytics/     Rolling punch statistics
    boxbunny_llm/           Local LLM coach + GenerateLLM service
    boxbunny_gui/           PySide6 GUI (6 tabs)
    boxbunny_bringup/       Launch files + system config
  motor_ws/
    ros2_ws/
      arm_GUI_fair_3.py     Fair motor control GUI (Teensy + micro-ROS)
      *.json                Punch position presets (Jab, Cross, Hook, etc.)
    run_arm_gui_fair_3_all.sh   Fair startup script
    teensy_firmware/        Teensy 4.1 firmware
    odrive/                 ODrive motor controller GUI
  action_prediction/        RGBD action recognition model + inference
  models/                   LLM + pose model weights
  data/                     Drill logs + session CSVs
  notebooks/                Deployment, testing, and command reference
  docs/                     Architecture + customization guides
```

---

## Quick Start

```bash
cd ~/Desktop/boxing_robot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Download models (Qwen2.5-3B + YOLO26n-pose)
chmod +x download_models.sh && ./download_models.sh

# Full system
ros2 launch boxbunny_bringup boxbunny_system.launch.py \
  enable_imu:=true enable_llm:=true enable_gui:=true

# Fair motor control (separate terminal)
cd motor_ws && ./run_arm_gui_fair_3_all.sh
```

---

## Launch Modes

| Command | Description |
|---|---|
| `ros2 launch boxbunny_bringup boxbunny_system.launch.py` | Full system (camera, tracker, fusion, drills, analytics, LLM, GUI) |
| `ros2 launch boxbunny_bringup boxbunny_deploy.launch.py` | Production deployment with detection mode switching |
| `ros2 launch boxbunny_bringup boxbunny_deploy.launch.py detection_mode:=action` | Deploy with AI action model |
| `ros2 launch boxbunny_bringup vision_only.launch.py` | Camera + glove tracker only |
| `ros2 launch boxbunny_bringup imu_only.launch.py enable_classifier:=true` | IMU nodes only |
| `ros2 launch boxbunny_bringup llm_only.launch.py` | LLM coach only |
| `ros2 launch boxbunny_bringup gui_only.launch.py` | GUI without system nodes |
| `cd motor_ws && ./run_arm_gui_fair_3_all.sh` | Fair motor control (micro-ROS + Teensy) |

---

## System Architecture

### Data Pipeline

```
RealSense D435i ──> glove_tracker ──> /punch_events_raw ──┐
                                                           ├──> punch_fusion_node ──> /punch_events
MPU6050 IMU ──> imu_punch_classifier ──> /imu/punch ──────┘          |
                                                                      ├──> reaction_drill_manager
                                                                      ├──> shadow_sparring_drill
                                                                      ├──> defence_drill ──> /motor_command
                                                                      ├──> punch_stats_node ──> /punch_stats
                                                                      └──> llm_talk_node ──> /coaching_feedback
                                                                      
boxing_gui (subscribes to all) ──> /robot/robot_action_trigger ──> arm_GUI_fair_3 ──> Teensy ──> Robot Arm
```

### Key Topics

| Topic | Type | Description |
|---|---|---|
| `/camera/color/image_raw` | Image | RealSense RGB stream |
| `/glove_detections` | GloveDetections | Tracked glove positions + velocity |
| `/action_prediction` | ActionPrediction | Real-time action classification |
| `/punch_events_raw` | PunchEvent | Vision-only punch triggers |
| `/imu/punch` | ImuPunch | IMU-classified punches |
| `/punch_events` | PunchEvent | Fused punches (vision + IMU) |
| `/drill_state` | String | Current drill FSM state |
| `/drill_progress` | DrillProgress | Drill step-by-step progress |
| `/drill_summary` | String (JSON) | Session summary |
| `/punch_stats` | String (JSON) | Rolling punch statistics |
| `/coaching_feedback` | TrashTalk | LLM coach messages |
| `/motor_command` | MotorCommand | Defence drill motor control |
| `/robot/robot_action_trigger` | String | GUI → motor control trigger |

### Services

| Service | Type | Description |
|---|---|---|
| `/start_stop_drill` | StartStopDrill | Start/stop reaction drill |
| `/start_drill` | StartDrill | Start shadow sparring or defence drill |
| `/llm/generate` | GenerateLLM | Generate coaching response (coach/encourage/trash/analysis) |
| `/calibrate_imu_punch` | CalibrateImuPunch | IMU punch threshold calibration |
| `/reaction_drill/new_user` | Trigger | Reset reaction drill for new user |

---

## Training Modes

### Reaction Drill
1. **Countdown** (3s) — GUI turns yellow, counts down
2. **Baseline** (1.5s) — Captures idle movement to filter false triggers
3. **Cue** (random 1.5-4s delay) — GUI turns green, signals punch
4. **Detection** (2.5s max) — Measures reaction time from cue to punch

Logs written to `~/boxbunny_logs/` (CSV).

### Shadow Sparring
- Combo sequence displayed on GUI (e.g., jab-cross-hook)
- User performs punches in order
- Accuracy and timing tracked per combo step
- Configured via `config/drill_definitions.yaml`

### Defence Drill
- Robot attacks user via motor commands
- User must block/dodge
- Defence response measured and scored

---

## Hardware

| Component | Interface | Purpose |
|---|---|---|
| Intel RealSense D435i | USB 3.0 | RGB + depth camera for glove tracking |
| MPU6050 IMU | I2C | Punch type classification (jab/cross, hook, uppercut) |
| Teensy 4.1 | micro-ROS serial | Motor controller for 4 Dynamixel servos |
| 4x Dynamixel Servos | Teensy PWM | Robot arm joints (2 per arm) |
| 7" Touchscreen (1024x600) | HDMI + USB | PySide6 GUI display |

---

## IMU Calibration

```bash
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch \
  "{punch_type: 'jab_or_cross', duration_s: 2.5}"
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch \
  "{punch_type: 'hook', duration_s: 2.5}"
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch \
  "{punch_type: 'uppercut', duration_s: 2.5}"
```

Calibration stored at `~/.boxbunny/imu_calibration.json`.

---

## LLM Coach

```bash
# Prompt via ROS service
ros2 service call /llm/generate boxbunny_msgs/srv/GenerateLLM \
  "{prompt: 'Give me a pep talk', mode: 'encourage', context: 'gui'}"
```

**Modes:** coach (technique tips), encourage (motivation), trash (competitive banter), analysis (session review)

**Prompt config:** `src/boxbunny_llm/config/persona_examples.yaml`, `coach_dataset.yaml`, `llm_prompts.yaml`

---

## GUI (PySide6, 6 Tabs)

| Tab | Features |
|---|---|
| **Training** | Live punch visualization, punch count, type breakdown |
| **Drills** | Reaction drill, shadow sparring, defence drill controls |
| **Coaching** | LLM chat interface, real-time coaching feedback |
| **IMU Calibration** | Per-punch-type calibration, debug visualization |
| **Height Calibration** | Player height measurement + robot height adjustment |
| **Camera** | Live camera feeds (RGB, glove debug, action debug) |

---

## Configuration

| Config File | Purpose |
|---|---|
| `src/boxbunny_vision/config/glove_tracker.yaml` | HSV thresholds, resize scale, camera FPS |
| `src/boxbunny_fusion/config/fusion.yaml` | Fusion window, confirmation thresholds |
| `src/boxbunny_drills/config/drill.yaml` | Reaction drill timing (countdown, baseline, cue delays) |
| `src/boxbunny_drills/config/drill_definitions.yaml` | Shadow sparring combo sequences |
| `src/boxbunny_imu/config/imu.yaml` | IMU sensitivity, punch thresholds |
| `src/boxbunny_llm/config/llm.yaml` | Model path, inference settings |
| `src/boxbunny_gui/config/gui.yaml` | GUI layout, feature toggles |
| `src/boxbunny_bringup/config/realsense.yaml` | Camera resolution, FPS |

---

## Dependencies

- **Core:** Python 3.10, ROS 2 Humble, NumPy, SciPy, scikit-learn
- **Vision:** OpenCV, PyRealSense2, YOLOv8, MMEngine, Pillow
- **ML:** PyTorch 2.3, TorchVision 0.18, ONNX Runtime
- **UI:** PySide6 6.8
- **LLM:** llama-cpp-python (GGUF inference)

Full environment: `environment.yml`

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/deployment.ipynb` | Full system deployment with LLM setup |
| `notebooks/unit_tests.ipynb` | Quick component tests + GUI launchers |

---

## Notes

- The glove tracker uses HSV segmentation by default; optional pose verification can be enabled in `glove_tracker.yaml`
- For lighter CPU usage: lower `camera_fps`, reduce `resize_scale`, or increase `process_every_n`
- If IMU is disabled, fusion passes through vision punches unchanged
- Fair motor control (`arm_GUI_fair_3.py`) runs as a separate process alongside the main ROS system
- Punch position presets (JSON files) can be recorded via the motor GUI's teach mode
