# CDE Fair — ROS 2 Architecture Reference

> Complete reference for the CDE Fair version of BoxBunny. This is a separate, smaller integration from the main project, designed for live demonstrations.

---

## 1. System Overview

- **9 ROS 2 packages** on Jetson Orin NX (ROS 2 Humble)
- **Platform:** Ubuntu 22.04, Python 3.10, PyTorch 2.3
- **Hardware:** RealSense D435i, MPU6050 IMU, Teensy 4.1 (4 motors), 7" touchscreen
- **Detection:** HSV color-based glove tracking (primary) or YOLO Pose + RGBD action model (optional)
- **3 Drill Modes:** Reaction Time, Shadow Sparring, Defence
- **Local LLM Coach:** Qwen 2.5 3B (GGUF)

---

## 2. All ROS 2 Nodes

### 2.1 realsense_glove_tracker — Glove Detection
- **Package:** boxbunny_vision
- **Subscribes:** RealSense camera (direct USB, PyRealSense2)
- **Publishes:**
  - `/camera/color/image_raw` (Image) — RGB stream
  - `/camera/depth/image_rect_raw` (Image) — Depth stream
  - `/glove_detections` (GloveDetections) — Tracked glove positions, distance, velocity
  - `/punch_events_raw` (PunchEvent) — Vision-only punch triggers
  - `/glove_debug_image` (Image) — Debug overlay
- **Config:** `src/boxbunny_vision/config/glove_tracker.yaml`
- **Method:** HSV color segmentation + depth + velocity thresholding

### 2.2 action_predictor — Action Recognition (Optional)
- **Package:** boxbunny_vision
- **Subscribes:** `/camera/color/image_raw`, `/camera/depth/image_rect_raw`
- **Publishes:**
  - `/action_prediction` (ActionPrediction) — label, probabilities, confidence
  - `/action_debug_image` (Image) — Debug overlay
- **Services:**
  - Server: `/action_predictor/set_simple_mode` (SetBool) — Toggle action model on/off
  - Server: `/action_predictor/calibrate_height` (Trigger) — Calibrate player height
- **Model:** YOLO Pose + RGBD action model (jab, cross, hook, uppercut, block detection)

### 2.3 mpu6050_node — IMU Driver
- **Package:** boxbunny_imu
- **Publishes:**
  - `/imu/data` (Imu) — Raw accelerometer/gyro data
- **Hardware:** MPU6050 sensor via I2C

### 2.4 imu_punch_classifier — IMU Punch Detection
- **Package:** boxbunny_imu
- **Subscribes:** `/imu/data` (Imu)
- **Publishes:**
  - `/imu/punch` (ImuPunch) — Classified punch type + confidence
  - `/imu/debug` (ImuDebug) — Raw debug data
- **Services:**
  - Server: `/calibrate_imu_punch` (CalibrateImuPunch) — Per-punch-type calibration
- **Config:** `src/boxbunny_imu/config/imu.yaml`
- **Calibration data:** `~/.boxbunny/imu_calibration.json`

### 2.5 imu_input_selector — Menu Navigation
- **Package:** boxbunny_imu
- **Subscribes:** `/imu/data` (Imu)
- **Publishes:**
  - `/imu_selection` (Int32) — Menu selection index
  - `/imu_input_enabled` (Bool) — Input state
- **Services:**
  - Server: `/imu_input_selector/enable` (SetBool)

### 2.6 punch_fusion_node — Sensor Fusion
- **Package:** boxbunny_fusion
- **Subscribes:**
  - `/punch_events_raw` (PunchEvent) — Vision punches
  - `/imu/punch` (ImuPunch) — IMU punches
- **Publishes:**
  - `/punch_events` (PunchEvent) — Fused final punches (vision + IMU confirmed)
- **Config:** `src/boxbunny_fusion/config/fusion.yaml`
- **Logic:** Temporal matching of vision and IMU events; if IMU disabled, passes through vision punches

### 2.7 reaction_drill_manager — Reaction Drill
- **Package:** boxbunny_drills
- **Subscribes:** `/punch_events` (PunchEvent)
- **Publishes:**
  - `/drill_state` (String) — FSM state (countdown, baseline, cue, detection, result)
  - `/drill_countdown` (Int32) — Countdown timer
  - `/drill_progress` (DrillProgress) — Step-by-step progress
  - `/drill_events` (DrillEvent) — Milestone events
  - `/drill_summary` (String) — JSON session summary
- **Services:**
  - Server: `/start_stop_drill` (StartStopDrill) — Start/stop reaction drill
  - Server: `/reaction_drill/new_user` (Trigger) — Reset for new user
- **Config:** `src/boxbunny_drills/config/drill.yaml`
- **Flow:** Countdown (3s) → Baseline capture (1.5s) → Random cue delay (1.5-4s) → Detection (2.5s max)
- **Logging:** CSV to `~/boxbunny_logs/` and `data/reaction_drill/`

### 2.8 shadow_sparring_drill — Shadow Boxing
- **Package:** boxbunny_drills
- **Subscribes:** `/punch_events` (PunchEvent)
- **Publishes:**
  - `/drill_progress` (DrillProgress) — Combo validation
  - `/drill_state` (String)
- **Services:**
  - Server: `/start_drill` (StartDrill) — Start with parameters
  - Server: `/shadow_drill/new_user` (Trigger)
  - Server: `/stop_shadow_drill` (Trigger)
- **Config:** `src/boxbunny_drills/config/drill_definitions.yaml` — Combo sequences

### 2.9 defence_drill — Defence Training
- **Package:** boxbunny_drills
- **Subscribes:** `/punch_events` (PunchEvent)
- **Publishes:**
  - `/drill_progress` (DrillProgress)
  - `/motor_command` (MotorCommand) — Motor commands for robot attacks
  - `/drill_state` (String)
- **Logic:** Robot attacks user, measures defensive response

### 2.10 punch_stats_node — Analytics
- **Package:** boxbunny_analytics
- **Subscribes:** `/punch_events` (PunchEvent)
- **Publishes:**
  - `/punch_stats` (String) — JSON with rolling counts, velocities, averages
- **Config:** `src/boxbunny_analytics/config/analytics.yaml`

### 2.11 llm_talk_node — AI Coach
- **Package:** boxbunny_llm
- **Subscribes:**
  - `/punch_events` (PunchEvent)
  - `/drill_events` (DrillEvent)
  - `/drill_summary` (String)
  - `/punch_stats` (String)
- **Publishes:**
  - `/coaching_feedback` (TrashTalk) — Coach messages
  - `/llm/stream` (String) — Streaming LLM output
- **Services:**
  - Server: `/llm/generate` (GenerateLLM) — modes: coach, encourage, trash, analysis
- **Model:** Qwen 2.5 3B Instruct (GGUF), local inference
- **Config:** `src/boxbunny_llm/config/llm.yaml`, `llm_prompts.yaml`, `persona_examples.yaml`, `coach_dataset.yaml`

### 2.12 boxing_gui — Main GUI
- **Package:** boxbunny_gui
- **Subscribes (all topics):**
  - Camera: `/camera/color/image_raw`, `/glove_debug_image`, `/action_debug_image`
  - Detection: `/glove_detections`, `/punch_events`, `/punch_events_raw`, `/action_prediction`
  - Drills: `/drill_state`, `/drill_countdown`, `/drill_summary`, `/drill_progress`
  - Coach: `/coaching_feedback`, `/llm/stream`
  - IMU: `/imu/debug`, `/imu_input_enabled`
  - Robot: `/robot/robot_action_status`, `/player_height`
- **Publishes:**
  - `/robot/robot_action_trigger` (String) — Motor control trigger
  - `/boxbunny/detection_mode` (String) — Switch detection mode
- **Service Clients:** start_stop_drill, start_drill, llm/generate, reaction_drill/new_user, shadow_drill/*, defence_drill/*, action_predictor/*, calibrate_imu_punch, dynamic parameter reconfig
- **Interface:** PySide6, 6 tabs (training, drills, coaching, IMU calibration, height calibration, camera)

### 2.13 arm_GUI_fair_3 — Motor Control (Fair Version)
- **Location:** `motor_ws/ros2_ws/arm_GUI_fair_3.py`
- **ROS node name:** `combined_gui_node`
- **Subscribes:**
  - `/robot/robot_action_trigger` (String) — Action triggers from boxing_gui
  - `/motor_feedback` (Float64MultiArray) — Teensy motor state [P1-P4, count]
- **Publishes:**
  - `/robot/robot_action_status` (String) — "GUI Ready"
  - Motor commands to Teensy via micro-ROS (Float64MultiArray) — [P1-P4, S1-S4, mode]
- **Features:** Manual teaching, sequence recording/playback, punch position presets
- **Punch Presets:** `1_Jab.json`, `2_Cross.json`, `3_Hook.json`, `4_R_Hook.json`, `5_L_UC.json`, `6_R_UC.json`
- **Heartbeat:** 10 Hz motor command publish

---

## 3. Complete Topic List

### Input (Sensors)
| Topic | Type | Publisher | Subscribers |
|---|---|---|---|
| `/camera/color/image_raw` | Image | realsense_glove_tracker | action_predictor, GUI |
| `/camera/depth/image_rect_raw` | Image | realsense_glove_tracker | action_predictor |
| `/imu/data` | Imu | mpu6050_node | imu_punch_classifier, imu_input_selector |
| `/motor_feedback` | Float64MultiArray | Teensy (micro-ROS) | arm_GUI_fair_3 |

### Processing
| Topic | Type | Publisher | Subscribers |
|---|---|---|---|
| `/glove_detections` | GloveDetections | realsense_glove_tracker | GUI |
| `/punch_events_raw` | PunchEvent | realsense_glove_tracker | punch_fusion_node |
| `/imu/punch` | ImuPunch | imu_punch_classifier | punch_fusion_node |
| `/action_prediction` | ActionPrediction | action_predictor | GUI |
| `/punch_events` | PunchEvent | punch_fusion_node | drills, analytics, LLM, GUI |
| `/punch_stats` | String (JSON) | punch_stats_node | llm_talk_node, GUI |

### Drill State
| Topic | Type | Publisher | Subscribers |
|---|---|---|---|
| `/drill_state` | String | drill managers | GUI |
| `/drill_countdown` | Int32 | reaction_drill_manager | GUI |
| `/drill_progress` | DrillProgress | drill managers | GUI |
| `/drill_events` | DrillEvent | drill managers | llm_talk_node |
| `/drill_summary` | String (JSON) | drill managers | llm_talk_node, GUI |

### Output
| Topic | Type | Publisher | Subscribers |
|---|---|---|---|
| `/coaching_feedback` | TrashTalk | llm_talk_node | GUI |
| `/llm/stream` | String | llm_talk_node | GUI |
| `/motor_command` | MotorCommand | defence_drill | Teensy/motor |
| `/robot/robot_action_trigger` | String | GUI | arm_GUI_fair_3 |
| `/robot/robot_action_status` | String | arm_GUI_fair_3 | GUI |
| `/imu_selection` | Int32 | imu_input_selector | GUI |
| `/imu_input_enabled` | Bool | imu_input_selector | GUI |

### Debug
| Topic | Type | Publisher |
|---|---|---|
| `/glove_debug_image` | Image | realsense_glove_tracker |
| `/action_debug_image` | Image | action_predictor |
| `/imu/debug` | ImuDebug | imu_punch_classifier |

---

## 4. Services

| Service | Type | Server | Clients |
|---|---|---|---|
| `/start_stop_drill` | StartStopDrill | reaction_drill_manager | GUI |
| `/start_drill` | StartDrill | shadow_sparring_drill, defence_drill | GUI |
| `/reaction_drill/new_user` | Trigger | reaction_drill_manager | GUI |
| `/shadow_drill/new_user` | Trigger | shadow_sparring_drill | GUI |
| `/stop_shadow_drill` | Trigger | shadow_sparring_drill | GUI |
| `/llm/generate` | GenerateLLM | llm_talk_node | GUI |
| `/calibrate_imu_punch` | CalibrateImuPunch | imu_punch_classifier | GUI |
| `/action_predictor/set_simple_mode` | SetBool | action_predictor | GUI |
| `/action_predictor/calibrate_height` | Trigger | action_predictor | GUI |
| `/imu_input_selector/enable` | SetBool | imu_input_selector | GUI |

---

## 5. Message Types

| Message | Key Fields |
|---|---|
| **PunchEvent** | stamp, glove, type, distance, velocity, confidence, method, imu_confirmed, source |
| **ActionPrediction** | label, probabilities, confidence |
| **GloveDetections** | array of GloveDetection (bbox, distance, velocity) |
| **DrillProgress** | drill_name, step, expected/detected actions, completion |
| **DrillEvent** | timestamp, event_type |
| **DrillDefinition** | name, steps, expected_actions |
| **ImuPunch** | timestamp, type, confidence |
| **ImuDebug** | raw IMU debug data |
| **MotorCommand** | motor control commands |
| **TrashTalk** | LLM coaching feedback text |

---

## 6. Node Graph (ASCII)

```
RealSense D435i ──USB──> realsense_glove_tracker ──> /glove_detections
                           |                      ──> /punch_events_raw ──> punch_fusion_node
                           +──> /camera/color/image_raw ──> action_predictor (optional)
                                                               |
                                                               +──> /action_prediction

MPU6050 IMU ──I2C──> mpu6050_node ──> /imu/data ──> imu_punch_classifier ──> /imu/punch
                                                 +──> imu_input_selector ──> /imu_selection

punch_fusion_node (/punch_events_raw + /imu/punch) ──> /punch_events
    |
    +──> reaction_drill_manager ──> /drill_state, /drill_countdown, /drill_progress, /drill_summary
    +──> shadow_sparring_drill ──> /drill_progress
    +──> defence_drill ──> /drill_progress, /motor_command
    +──> punch_stats_node ──> /punch_stats
    |
    +──> llm_talk_node (/punch_stats + /drill_events) ──> /coaching_feedback, /llm/stream

boxing_gui (subscribes to everything) ──> /robot/robot_action_trigger
                                                    |
arm_GUI_fair_3 <── /robot/robot_action_trigger      |
    |                                               |
    +──> Teensy 4.1 (micro-ROS) ──> 4 Dynamixel motors ──> Robot Arm
    +──> /robot/robot_action_status ──> boxing_gui
```

---

## 7. Launch Files

| Launch File | Description | Key Args |
|---|---|---|
| `boxbunny_system.launch.py` | Full system | enable_imu, enable_llm, enable_gui |
| `boxbunny_deploy.launch.py` | Production | detection_mode (color/action), enable_imu, enable_llm, headless |
| `vision_only.launch.py` | Camera + tracker only | - |
| `imu_only.launch.py` | IMU nodes only | enable_classifier |
| `llm_only.launch.py` | LLM coach only | - |
| `gui_only.launch.py` | GUI without system | - |
| `realsense_only.launch.py` | Camera driver only | - |
| `run_arm_gui_fair_3_all.sh` | Fair motor control | Launches micro-ROS agent + arm_GUI_fair_3.py |

---

## 8. Hardware Connections

| Hardware | Interface | Node | Details |
|---|---|---|---|
| RealSense D435i | USB 3.0 (PyRealSense2) | realsense_glove_tracker | RGB + Depth |
| MPU6050 IMU | I2C | mpu6050_node | 6-axis accelerometer/gyro |
| Teensy 4.1 | micro-ROS serial | arm_GUI_fair_3 | 4 Dynamixel motors |
| 7" Touchscreen | HDMI + USB | boxing_gui (PySide6) | 1024x600 |

---

## 9. Key Differences from Main Project

| Aspect | CDE Fair Version | Main Project |
|---|---|---|
| Detection | HSV color glove tracking | Transformer-based action recognition |
| Fusion | Simple temporal matching | 0.8s CV buffer + pad constraint voting |
| GUI | 6-tab PySide6 | 24-page QStackedWidget |
| Drills | Reaction, Shadow, Defence | Training, Sparring, Free, Power, Stamina, Reaction |
| Dashboard | None | Vue 3 SPA + FastAPI + WebSocket |
| Motor Control | arm_GUI_fair_3 (direct) | robot_node (via V4 GUI) |
| Sessions | Per-drill logging (CSV) | Full session lifecycle state machine |
| Auth | None | JWT + pattern lock + phone login |
| LLM | Same (Qwen 2.5 3B) | Same (Qwen 2.5 3B) |
| IMU | MPU6050 (single sensor) | Teensy 4.1 (4 pad IMUs + 2 arm IMUs) |
