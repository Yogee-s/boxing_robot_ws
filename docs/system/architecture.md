# BoxBunny System Architecture

## 1. System Overview

BoxBunny is a boxing robot training system running on an **NVIDIA Jetson Orin NX** with **ROS 2 Humble**. The system consists of 10+ ROS nodes that process real-time sensor data from an Intel RealSense D435i depth camera and Teensy-based IMU sensors, fuse predictions using a hybrid CV+IMU pipeline, and control a dual-arm boxing robot for interactive training.

### Architecture Diagram

```
 +------------------------------------------------------------------+
 |                    NVIDIA Jetson Orin NX                          |
 |                                                                  |
 |  +-------------------+     +-------------------+                 |
 |  |    cv_node         |     |    imu_node        |                |
 |  | RealSense D435i   |     | Teensy IMU bridge  |                |
 |  | TensorRT FP16     |     | Dual mode:         |                |
 |  | Voxel+Pose model  |     |  NAV / TRAINING    |                |
 |  +--------+----------+     +--------+-----------+                |
 |           |                          |                           |
 |           v                          v                           |
 |  +--------------------------------------------+                  |
 |  |         punch_processor                     |                  |
 |  | CV+IMU Fusion | Pad constraints | Defense   |                  |
 |  +--------+-----------------------------------+                  |
 |           |                                                      |
 |           v                                                      |
 |  +-------------------+    +-------------------+                  |
 |  | session_manager   |--->| drill_manager     |                  |
 |  | Lifecycle/rounds  |    | 50 combos         |                  |
 |  | Summary building  |    | Accuracy scoring  |                  |
 |  +--------+----------+    +-------------------+                  |
 |           |                                                      |
 |           v                                                      |
 |  +-------------------+    +-------------------+                  |
 |  | sparring_engine   |    | free_training_eng |                  |
 |  | 5 Markov styles   |    | Reactive counters |                  |
 |  | Weakness bias     |    | Pad-to-punch map  |                  |
 |  +--------+----------+    +--------+----------+                  |
 |           |                         |                            |
 |           v                         v                            |
 |  +--------------------------------------------+                  |
 |  |           robot_node                        |                  |
 |  | Motor bridges | Punch execution | Height    |                  |
 |  +--------------------------------------------+                  |
 |           |                                                      |
 |  +-------------------+    +-------------------+                  |
 |  | llm_node          |    | analytics_node    |                  |
 |  | Qwen2.5-3B GGUF   |    | Per-session stats |                  |
 |  | Coaching tips     |    | JSON publishing   |                  |
 |  +-------------------+    +-------------------+                  |
 |                                                                  |
 |  +-------------------+    +-------------------+                  |
 |  | gesture_node      |    | boxbunny_gui      |                  |
 |  | MediaPipe hands   |    | PySide6 + Bridge  |                  |
 |  | (optional)        |    | QThread ROS conn  |                  |
 |  +-------------------+    +-------------------+                  |
 +------------------------------------------------------------------+
         |                           |
         v                           v
 +------------------+    +-------------------------+
 | Teensy V4 MCU    |    | Boxing Arm Control V4   |
 | micro-ROS agent  |    | (External GUI)          |
 | 4x motors, IMUs  |    | Motor tuning & diag     |
 +------------------+    +-------------------------+
```

---

## 2. ROS Nodes

### 2.1 cv_node

**Purpose**: Computer vision inference -- detects punch types, tracks user pose and position using Intel RealSense D435i depth camera.

**Key behaviour**:
- Opens the RealSense camera directly via `pyrealsense2` (the ROS driver crashes on Jetson due to a D435i HID bug).
- Runs a fused Voxel+Pose action recognition model with TensorRT FP16 acceleration.
- Adaptive frame rate: **6 Hz idle** / **30 Hz active** (triggered by session state).
- Applies temporal smoothing (EMA), hysteresis, state machine filtering, and block consecutive filtering to raw model output.
- Computes user bounding box, depth, lateral displacement, and depth displacement.
- Determines person direction (left/right/centre) using a 30% centre zone with 20px hysteresis.

| Direction | Topic | Message Type |
|---|---|---|
| Publishes | `/boxbunny/cv/detection` | PunchDetection |
| Publishes | `/boxbunny/cv/pose` | PoseEstimate |
| Publishes | `/boxbunny/cv/user_tracking` | UserTracking |
| Publishes | `/boxbunny/cv/person_direction` | std_msgs/String |
| Publishes | `/boxbunny/cv/status` | std_msgs/String |
| Publishes | `/boxbunny/cv/debug_info` | std_msgs/String (JSON) |
| Publishes | `/camera/color/image_raw` | sensor_msgs/Image |
| Publishes | `/camera/aligned_depth_to_color/image_raw` | sensor_msgs/Image |
| Subscribes | `/boxbunny/session/state` | SessionState |

---

### 2.2 imu_node

**Purpose**: Processes raw IMU data from Teensy microcontroller. Operates in two modes depending on session state.

**Dual mode operation**:
- **Navigation mode** (idle): Pad impacts are translated to navigation commands (prev/next/enter/back). Debounced at 300ms per-pad, 200ms global.
- **Training mode** (active session): Pad impacts become PunchEvent messages with force classification. 200ms grace period on mode transition.

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/imu/pad/impact` | PadImpact |
| Subscribes | `/boxbunny/imu/arm/strike` | ArmStrike |
| Subscribes | `/boxbunny/session/state` | SessionState |
| Subscribes | `/robot/strike_detected` | std_msgs/String |
| Publishes | `/boxbunny/imu/punch_event` | PunchEvent |
| Publishes | `/boxbunny/imu/nav_event` | NavCommand |
| Publishes | `/boxbunny/imu/arm_event` | ArmStrikeEvent |
| Publishes | `/boxbunny/imu/status` | IMUStatus |
| Service | `/boxbunny/imu/set_mode` | SetImuMode |
| Service | `/boxbunny/imu/calibrate` | CalibrateImuPunch |

---

### 2.3 punch_processor

**Purpose**: Fuses CV detections with IMU pad impacts to produce confirmed punch events. Handles defense detection when the robot attacks.

**Key behaviour**:
- **Fusion window**: +/-200ms matching between CV detection and IMU pad impact.
- **Pad-constraint filtering**: Rejects impossible CV classifications based on pad location (e.g., hooks cannot land on centre pad).
  - Centre pad: jab, cross only
  - Left pad: left hook, left uppercut only
  - Right pad: right hook, right uppercut only
  - Head pad: any offensive punch
- **Reclassification**: If CV class is invalid for the struck pad but a secondary class is valid with >=25% confidence, reclassifies.
- **Defense detection**: Opens a 500ms window when a robot punch command is received. Evaluates arm IMU contact + CV block detection + bbox displacement to classify defense as block/slip/dodge/hit.
- **Ring buffers**: Maintains recent CV and IMU events for temporal matching.

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/cv/detection` | PunchDetection |
| Subscribes | `/boxbunny/imu/punch_event` | PunchEvent |
| Subscribes | `/boxbunny/imu/arm_event` | ArmStrikeEvent |
| Subscribes | `/boxbunny/cv/user_tracking` | UserTracking |
| Subscribes | `/boxbunny/robot/command` | RobotCommand |
| Publishes | `/boxbunny/punch/confirmed` | ConfirmedPunch |
| Publishes | `/boxbunny/punch/defense` | DefenseEvent |
| Publishes | `/boxbunny/punch/session_summary` | SessionPunchSummary |

---

### 2.4 session_manager

**Purpose**: Manages the complete training session lifecycle, accumulates all punch/defense/tracking data, and builds comprehensive session summaries.

**Session lifecycle**: `idle -> countdown -> active -> rest -> active -> ... -> complete -> idle`

**Key behaviour**:
- Manages countdown (configurable, default 3s), round timers, and rest periods.
- Auto-adjusts robot height during countdown using user bbox_top_y.
- Accumulates punch distributions, force data, pad hits, defense events, depth/lateral movement, CV prediction events, IMU strikes, direction changes, and defense reaction times.
- Publishes SessionState at 2Hz heartbeat (engines use this as a watchdog).
- Builds enriched session summaries with fields: punches_per_minute, max_power, imu_confirmation_rate, max_lateral_displacement, max_depth_displacement.
- Auto-saves every 10s for crash recovery.
- Forces reset from stuck states when a new session is requested.

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/punch/confirmed` | ConfirmedPunch |
| Subscribes | `/boxbunny/punch/defense` | DefenseEvent |
| Subscribes | `/boxbunny/cv/user_tracking` | UserTracking |
| Subscribes | `/boxbunny/session/config` | SessionConfig |
| Subscribes | `/boxbunny/cv/detection` | PunchDetection |
| Subscribes | `/boxbunny/imu/punch_event` | PunchEvent |
| Subscribes | `/boxbunny/cv/person_direction` | std_msgs/String |
| Subscribes | `/boxbunny/robot/command` | RobotCommand |
| Publishes | `/boxbunny/session/state` | SessionState |
| Publishes | `/boxbunny/session/config_json` | std_msgs/String |
| Publishes | `/boxbunny/punch/session_summary` | SessionPunchSummary |
| Publishes | `/boxbunny/robot/height` | HeightCommand |
| Service | `/boxbunny/session/start` | StartSession |
| Service | `/boxbunny/session/end` | EndSession |

---

### 2.5 drill_manager

**Purpose**: Loads 50 combo drill definitions from YAML, validates detected punch sequences against expected combos, and tracks accuracy/timing/streak.

**Key behaviour**:
- Loads drills from `config/drills.yaml` -- 15 beginner, 20 intermediate, 15 advanced combos.
- Validates incoming confirmed punches against the expected combo sequence in real-time.
- Scores each attempt with accuracy (correct punches / total expected) and timing score (deviation from target tempo).
- Tracks streaks (consecutive correct combos) and publishes live progress.
- Auto-detects combo timeout based on combo length and timing tolerance.
- Difficulty parameters control timing tolerance (400ms beginner, 250ms intermediate, 150ms advanced).

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/punch/confirmed` | ConfirmedPunch |
| Subscribes | `/boxbunny/session/state` | SessionState |
| Publishes | `/boxbunny/drill/definition` | DrillDefinition |
| Publishes | `/boxbunny/drill/event` | DrillEvent |
| Publishes | `/boxbunny/drill/progress` | DrillProgress |
| Service | `/boxbunny/drill/start` | StartDrill |

---

### 2.6 sparring_engine

**Purpose**: Generates unpredictable robot attack sequences using Markov-chain transition matrices. Supports 5 boxing styles and reactive counter-punches.

**Key behaviour**:
- **5 styles**: Boxer (technical/adaptive), Brawler (aggressive/hook-heavy), Counter-Puncher (reactive/jab-cross-heavy), Pressure (relentless/jab-forward), Switch (rotates between other styles every 20s).
- **Markov chain**: 6x6 transition matrices per style. Current punch determines probability distribution of next punch.
- **Difficulty scaling**: Attack interval -- easy 2.0s, medium 1.2s, hard 0.7s.
- **Weakness bias**: Adds +0.08 bias per weakness point to target punches the user defends poorly against.
- **Idle surprise**: If user is idle >3s, triggers attack at 60% of normal interval.
- **Block reaction**: If user blocked last punch, forces a different punch type next.
- **Counter-punches**: In sparring mode, reacts to user pad strikes with probabilistic counter-punches (30%/50%/80% by difficulty).
- **Robot busy flag**: Only one arm strike at a time -- waits for strike feedback before next attack.
- **Watchdog**: Deactivates if no SessionState received for 5s (session_manager crash protection).

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/session/state` | SessionState |
| Subscribes | `/boxbunny/punch/confirmed` | ConfirmedPunch |
| Subscribes | `/boxbunny/imu/punch_event` | PunchEvent |
| Subscribes | `/boxbunny/session/config_json` | std_msgs/String |
| Subscribes | `/robot/strike_feedback` | std_msgs/String |
| Publishes | `/boxbunny/robot/command` | RobotCommand |

---

### 2.7 free_training_engine

**Purpose**: Purely reactive counter-punch engine for free training mode. When the user strikes a pad, the robot throws back a random punch from the configured set for that pad.

**Key behaviour**:
- Activates only when session state is `active` and mode is `free`.
- Pad-to-counter mapping (configurable in `boxbunny.yaml`):
  - Centre pad -> jab (code 1) or cross (code 2)
  - Left pad -> left hook (code 3) or left uppercut (code 5)
  - Right pad -> right hook (code 4) or right uppercut (code 6)
  - Head pad -> jab (code 1) or cross (code 2)
- Cooldown between counters: 1500ms (configurable).
- Single-strike-at-a-time: ignores pad hits while robot is executing.
- Speed override from session config.
- Independent from sparring_engine (different node entirely).

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/session/state` | SessionState |
| Subscribes | `/boxbunny/imu/punch_event` | PunchEvent |
| Subscribes | `/robot/strike_feedback` | std_msgs/String |
| Subscribes | `/boxbunny/session/config_json` | std_msgs/String |
| Publishes | `/boxbunny/robot/command` | RobotCommand |

---

### 2.8 analytics_node

**Purpose**: Aggregates per-session statistics and publishes them as JSON for the dashboard and other consumers.

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/punch/confirmed` | ConfirmedPunch |
| Subscribes | `/boxbunny/punch/defense` | DefenseEvent |
| Subscribes | `/boxbunny/punch/session_summary` | SessionPunchSummary |
| Publishes | `/boxbunny/analytics/session` | (JSON analytics) |

---

### 2.9 llm_node

**Purpose**: Runs Gemma 4 E2B (GGUF Q4_K_M quantised) for real-time coaching tips and post-session analysis.

**Key behaviour**:
- Model: `gemma-4-E2B-it-Q4_K_M.gguf` loaded via llama.cpp with full GPU offload (`n_gpu_layers=-1`).
- Context window: 2048 tokens, max generation: 128 tokens, temperature: 0.7.
- Publishes coaching tips every ~18s during active sessions via the CoachTip topic.
- System prompt keys: `drill_feedback`, `session_analysis`, `technique_tips`, `drill_suggestions`, `general`.
- Falls back to `config/fallback_tips.json` if the LLM is unavailable.
- Vision support: multimodal projector (`mmproj-F16.gguf`) lazy-loaded on first image chat request from phone dashboard. Does not affect real-time coaching tips.
- Respawn enabled with 5s delay.

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/drill/event` | DrillEvent |
| Subscribes | `/boxbunny/punch/session_summary` | SessionPunchSummary |
| Publishes | `/boxbunny/coach/tip` | CoachTip |
| Service | `/boxbunny/llm/generate` | GenerateLlm |

---

### 2.10 robot_node

**Purpose**: Bridge between ROS commands and the Teensy V4 motor controller. Manages punch execution sequences, height adjustment motor, and yaw tracking motor.

**Key behaviour**:
- Loads pre-recorded punch sequences from `data/punch_sequences/`.
- Publishes motor_commands at ~100Hz when armed (position targets in radians, speed in rad/s).
- Forwards person_direction to `/robot/yaw_cmd` for the turning motor.
- Manages height motor via `/robot/height_cmd` (UP:pwm, DOWN:pwm, STOP, REVERSE).
- Publishes strike completion feedback.
- Heartbeat at 10Hz, status at 0.5Hz.

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/robot/command` | RobotCommand |
| Subscribes | `/boxbunny/robot/height` | HeightCommand |
| Subscribes | `/boxbunny/robot/round_control` | RoundControl |
| Subscribes | `/boxbunny/cv/person_direction` | std_msgs/String |
| Subscribes | `motor_feedback` | std_msgs/Float64MultiArray |
| Subscribes | `/robot/strike_feedback` | std_msgs/String |
| Publishes | `motor_commands` | std_msgs/Float64MultiArray |
| Publishes | `/robot/height_cmd` | std_msgs/String |
| Publishes | `/robot/yaw_cmd` | std_msgs/String |
| Publishes | `/robot/strike_command` | std_msgs/String |
| Publishes | `/robot/system_enable` | std_msgs/String |
| Publishes | `/boxbunny/robot/strike_complete` | std_msgs/String |
| Publishes | `/boxbunny/robot/status` | std_msgs/String |

---

### 2.12 ble_bridge_node

**Purpose**: owns the single BLE link to the base-rotation Arduino (`BoxBunny Base`). Multiplexes two command streams over the one connection: height-motor commands from the GUI/phone, and live person-tracking rotation commands from CV.

**Key behaviour**:
- Auto-connects on startup via a background asyncio thread (`bleak`). Reconnects with 2 s backoff on drop.
- Subscribes to `/boxbunny/robot/height` (HeightCommand) and translates to `HUP:<pwm>` / `HDOWN:<pwm>` / `HSTOP`.
- When tracking is enabled via the `tracking_start` service, forwards `/boxbunny/cv/person_direction` as `L:<rpm>` / `R:<rpm>` / `S`, gated by `/boxbunny/cv/user_tracking.depth` (default threshold 0.8 m with 0.15 m hysteresis).
- Publishes `/boxbunny/ble/status` at 1 Hz (JSON `{connected, tracking, last_cmd, near_range_m}`) for the GUI to read.
- Height safety (belt-and-braces):
  - 500 ms Jetson-side deadman — re-issues `HSTOP` if height publishers go silent.
  - 250 ms refresh-window watchdog inside `ble_control.ino` — zeroes PWM if *any* command stream stops. This is the authoritative safety layer.
- On every BLE disconnect, the bridge drains the queued commands so a stale backlog of `HUP:200` doesn't flood the motor on reconnect.

See [hardware/ble_bridge.md](../hardware/ble_bridge.md) for the full protocol and design rationale.

| Direction | Topic / Service | Message Type |
|---|---|---|
| Subscribes | `/boxbunny/robot/height` | HeightCommand |
| Subscribes | `/boxbunny/cv/person_direction` | std_msgs/String |
| Subscribes | `/boxbunny/cv/user_tracking` | UserTracking |
| Publishes | `/boxbunny/ble/status` | std_msgs/String (JSON) |
| Service | `/boxbunny/ble/tracking_start` | std_srvs/Trigger |
| Service | `/boxbunny/ble/tracking_stop` | std_srvs/Trigger |

---

### 2.11 gesture_node (Optional)

**Purpose**: MediaPipe hand gesture recognition for touchless GUI navigation. Disabled by default.

**Key behaviour**:
- Subscribes to camera colour feed and runs MediaPipe hand detection.
- Detects gestures: open palm (enter), closed fist (back), swipe left/right (prev/next).
- Publishes NavCommand on the same topic as imu_node (`/boxbunny/imu/nav_event`) so both input methods work identically.
- Hold duration 0.7s, cooldown 1.5s, min confidence 0.7, swipe threshold 100px.
- Suspends during non-idle session states.

| Direction | Topic | Message Type |
|---|---|---|
| Subscribes | `/camera/color/image_raw` | sensor_msgs/Image |
| Subscribes | `/boxbunny/session/state` | SessionState |
| Publishes | `/boxbunny/imu/nav_event` | NavCommand |
| Publishes | `/boxbunny/gesture/status` | std_msgs/String |

---

## 3. Message Types

### 3.1 Custom Messages (boxbunny_msgs/msg)

| # | Message | Key Fields |
|---|---------|-----------|
| 1 | **PadImpact** | `timestamp`, `pad` (left/centre/right/head), `level` (light/medium/hard), `accel_magnitude` |
| 2 | **ArmStrike** | `timestamp`, `arm` (left/right), `contact` (bool) |
| 3 | **ArmStrikeEvent** | `timestamp`, `arm` (left/right), `contact` (bool) |
| 4 | **IMUStatus** | `left_pad_connected`, `centre_pad_connected`, `right_pad_connected`, `head_pad_connected`, `left_arm_connected`, `right_arm_connected`, `is_simulator` |
| 5 | **PunchEvent** | `timestamp`, `pad`, `level`, `force_normalized` (0.33/0.66/1.0), `accel_magnitude` |
| 6 | **NavCommand** | `timestamp`, `command` (prev/next/enter/back) |
| 7 | **PunchDetection** | `timestamp`, `punch_type`, `confidence`, `raw_class`, `consecutive_frames` |
| 8 | **PoseEstimate** | `timestamp`, `keypoints` (float32[] COCO-17 flattened), `movement_delta` |
| 9 | **UserTracking** | `timestamp`, `bbox_centre_x/y`, `bbox_top_y`, `bbox_width/height`, `depth`, `lateral_displacement`, `depth_displacement`, `user_detected` |
| 10 | **ConfirmedPunch** | `timestamp`, `punch_type`, `pad`, `level`, `force_normalized`, `cv_confidence`, `imu_confirmed`, `cv_confirmed`, `accel_magnitude` |
| 11 | **DefenseEvent** | `timestamp`, `arm`, `robot_punch_code`, `struck`, `defense_type` (block/slip/dodge/unknown) |
| 12 | **SessionPunchSummary** | `total_punches`, `punch_distribution_json`, `force_distribution_json`, `pad_distribution_json`, `average_confidence`, `peak_force_level`, `imu_confirmation_rate`, `robot_punches_thrown/landed`, `defense_rate`, `defense_type_breakdown_json`, `avg_depth`, `depth_range`, `lateral_movement`, `max_lateral_displacement`, `max_depth_displacement`, `movement_timeline_json`, `session_duration_sec`, `rounds_completed` |
| 13 | **SessionState** | `state` (idle/countdown/active/rest/complete), `mode` (training/sparring/free/power/stamina/reaction), `username` |
| 14 | **SessionConfig** | `mode`, `difficulty`, `combo_sequence` (JSON), `rounds`, `work_time_sec`, `rest_time_sec`, `speed`, `style` |
| 15 | **RobotCommand** | `command_type` (punch/set_speed), `punch_code` (1-6), `speed` (slow/medium/fast), `source` (scheduled/counter/drill) |
| 16 | **HeightCommand** | `target_height_px`, `current_height_px`, `action` (adjust/calibrate/manual_up/manual_down/stop) |
| 17 | **RoundControl** | `action` (start/stop) |
| 18 | **DrillDefinition** | `drill_name`, `difficulty`, `combo_sequence` (string[]), `total_combos`, `target_speed` |
| 19 | **DrillEvent** | `timestamp`, `event_type` (combo_started/completed/missed/partial), `combo_index`, `accuracy`, `timing_score`, `detected_punches`, `expected_punches` |
| 20 | **DrillProgress** | `timestamp`, `combos_completed`, `combos_remaining`, `overall_accuracy`, `current_streak`, `best_streak` |
| 21 | **CoachTip** | `timestamp`, `tip_text`, `tip_type` (technique/encouragement/correction/suggestion), `trigger`, `priority` (0-2) |

### 3.2 Services (boxbunny_msgs/srv + std_srvs)

| # | Service | Request | Response |
|---|---------|---------|----------|
| 1 | **StartSession** | `mode`, `difficulty`, `config_json`, `username` | `success`, `session_id`, `message` |
| 2 | **EndSession** | `session_id` | `success`, `summary_json`, `message` |
| 3 | **StartDrill** | `drill_name`, `difficulty`, `rounds`, `work_time_sec`, `rest_time_sec`, `speed` | `success`, `drill_id`, `message` |
| 4 | **SetImuMode** | `mode` (navigation/training) | `success`, `current_mode` |
| 5 | **CalibrateImuPunch** | `pad` (left/centre/right/head/all) | `success`, `message` |
| 6 | **GenerateLlm** | `prompt`, `context_json`, `system_prompt_key` | `success`, `response`, `generation_time_sec` |
| 7 | **tracking_start** (std_srvs/Trigger) | — | `success`, `message` — enables `ble_bridge_node` CV→BLE rotation forwarding |
| 8 | **tracking_stop** (std_srvs/Trigger) | — | `success`, `message` — sends `S` and disables forwarding |

---

## 4. Topic Configuration

All ROS topic and service names are defined in a single YAML file:

**File**: `config/ros_topics.yaml`

This file is loaded at startup by `src/boxbunny_core/boxbunny_core/constants.py`, which exposes two classes:

- **`Topics`** -- All ROS topic name constants (e.g., `Topics.PUNCH_CONFIRMED`, `Topics.SESSION_STATE`).
- **`Services`** -- All ROS service name constants (e.g., `Services.START_SESSION`, `Services.GENERATE_LLM`).

The loader uses a `_t(section, key, default)` helper that reads from the YAML and falls back to a hardcoded default if the file is missing or a key is absent. This means renaming any topic requires editing only `ros_topics.yaml` -- no Python changes needed.

Additional constants defined in `constants.py`:

| Class | Purpose | Values |
|-------|---------|--------|
| `PunchType` | CV model output classes | jab, cross, left_hook, right_hook, left_uppercut, right_uppercut, block, idle |
| `PadLocation` | IMU pad identifiers + valid punch constraints | left, centre, right, head |
| `ImpactLevel` | Force classification | light (0.33), medium (0.66), hard (1.0) |
| `SessionState` | Session lifecycle states | idle, countdown, active, rest, complete |
| `TrainingMode` | All training modes | training, sparring, free, power, stamina, reaction |
| `Difficulty` | Difficulty levels | beginner, intermediate, advanced |
| `Speed` | Robot speed settings | slow, medium, fast |
| `MotorSpeed` | Motor speed presets (rad/s) | 8.0, 15.0, 25.0, max 30.0 |
| `DefenseType` | Defense classifications | block, slip, dodge, hit, unknown |
| `NavCommand` | Navigation commands + pad mapping | prev (left), next (right), enter (centre), back (head) |

---

## 5. Launch Files

All launch files are located in `src/boxbunny_core/launch/`.

### 5.1 boxbunny_dev.launch.py -- Development Mode

Runs with **Teensy Simulator** instead of real hardware. CV node uses CPU fallback with reduced inference interval.

**Nodes launched**:
- `teensy_simulator` (ExecuteProcess -- `tools/teensy_simulator.py`)
- `imu_node`
- `cv_node` (device=cpu, inference_interval=3)
- `punch_processor`
- `session_manager`
- `drill_manager`
- `llm_node`
- `boxbunny_gui`

**Not included**: robot_node, sparring_engine, free_training_engine, analytics_node (lightweight dev testing).

### 5.2 boxbunny_full.launch.py -- Full Production

Launches all nodes for the complete system with real hardware.

**Nodes launched**:
- `imu_node` (nav_debounce_ms=500, nav_global_debounce_ms=300, mode_transition_ms=200)
- `robot_node` (heartbeat_hz=10.0)
- `punch_processor` (fusion_window_ms=200, defense_window_ms=500)
- `session_manager` (respawn=True, respawn_delay=2.0, countdown_seconds=3, autosave_interval_s=10.0)
- `drill_manager`
- `sparring_engine`
- `free_training_engine`
- `analytics_node`
- `llm_node` (respawn=True, respawn_delay=5.0, n_gpu_layers=-1, n_ctx=2048)
- `boxbunny_gui`
- `gesture_node` (commented out by default -- uncomment to enable)

**Note**: `cv_node` is launched separately by `launch_system.sh` with custom PYTHONPATH for PyTorch.

### 5.3 headless.launch.py -- Headless Testing

Core processing nodes without GUI, sparring engine, or robot control. Suitable for automated testing and CI.

**Nodes launched**:
- `imu_node`
- `cv_node` (device=cpu)
- `punch_processor`
- `session_manager`
- `drill_manager`
- `analytics_node`
- `llm_node`

### 5.4 Simulator Launch Files

- **`teensy_simulator.launch.py`** -- Launches the Teensy IMU simulator standalone.
- **`imu_simulator.launch.py`** -- Launches the IMU simulator for testing IMU data flow.

### Build Command

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```
