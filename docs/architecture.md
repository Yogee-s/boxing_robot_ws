# BoxBunny System Architecture

## 1. System Overview

BoxBunny is an intelligent boxing training robot that combines real-time computer vision, inertial measurement, robotic arm control, and AI coaching to deliver structured and adaptive boxing training sessions. The system is designed as a standalone training station: the user stands in front of the robot, which detects their punches via a camera-based action prediction model, confirms strikes using IMU-equipped pads, controls a dual-arm robot for sparring exercises, and provides live AI coaching feedback.

### Hardware Platform

| Component | Model / Specification | Role |
|---|---|---|
| **Compute** | NVIDIA Jetson Orin Nano (JetPack 6.1, CUDA 12.6) | Central processing: runs CV inference, ROS 2 nodes, LLM, GUI |
| **Depth Camera** | Intel RealSense D435i | Synchronized RGB (960x540 @ 30fps) + Depth (848x480 @ 30fps) |
| **Microcontroller** | Teensy 4.1 | Motor control, IMU data acquisition, micro-ROS agent |
| **Arm Motors** | DYNAMIXEL servos (x4) | Two-arm robot: L1/L2 (left arm), R1/R2 (right arm) |
| **Height Motor** | DC motor with H-bridge | Adjusts robot height to match user via PWM commands |
| **Yaw Motor** | DC motor | Rotates robot torso to track user lateral position |
| **IMU Pads** | 4x IMU sensors (MPU-6050) on Teensy I2C buses | Centre, Left, Right, Head strike pads |
| **Touchscreen** | 7" or 10" display | PySide6 GUI for session control and feedback |
| **Phone** | Any smartphone via Wi-Fi AP | Vue.js 3 dashboard for remote monitoring and chat |

### Physical IMU Wiring

The Teensy 4.1 reads four IMU sensors across two I2C buses:

| Teensy Index | I2C Bus | Address | Pad Name (User Perspective) |
|---|---|---|---|
| 0 | Wire | 0x68 | Centre |
| 1 | Wire | 0x69 | Right |
| 2 | Wire1 | 0x68 | Left |
| 3 | Wire1 | 0x69 | Head |

Note the crossover: Teensy index 1 maps to the user's RIGHT pad, and index 2 maps to the user's LEFT pad. This mapping is configured in `config/boxbunny.yaml` under `fusion.imu_pad_map`.

---

## 2. Software Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Middleware** | ROS 2 Humble | Humble Hawksbill | Inter-node communication, topic pub/sub, services |
| **Language** | Python | 3.10 | All node logic, inference, GUI |
| **CV Model** | PyTorch + TensorRT | PyTorch 2.x, TRT 10.3 | Voxel-pose fusion model for action recognition |
| **Pose Estimation** | Ultralytics YOLO | yolo26n-pose (TRT FP16) | 7-joint upper-body pose from RGB |
| **Touchscreen GUI** | PySide6 | 6.x | On-robot touchscreen interface |
| **Phone Dashboard** | Vue.js 3 + Vite | 3.x | Mobile web dashboard over Wi-Fi AP |
| **Dashboard Server** | FastAPI | 0.100+ | REST API serving dashboard data |
| **AI Coach** | llama-cpp-python | Latest | On-device LLM (Qwen 2.5 3B, Q4_K_M GGUF) |
| **Database** | SQLite | 3.x | Session history, user accounts, punch statistics |
| **Micro-ROS** | micro-ROS Agent | Humble | Teensy 4.1 to ROS 2 bridge for motor commands |

### Build System

```bash
# Build the workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# Run tests
python3 -m pytest tests/ -v
```

---

## 3. Node Architecture

BoxBunny runs 10 ROS 2 nodes, each with a single responsibility. All nodes are defined as entry points in `src/boxbunny_core/setup.py`:

### Node Descriptions

| Node | Entry Point | Responsibility |
|---|---|---|
| **cv_node** | `boxbunny_core.cv_node:main` | Wraps the action prediction inference engine. Subscribes to RealSense RGB+depth, publishes punch detections, pose estimates, user tracking, and person direction. |
| **imu_node** | `boxbunny_core.imu_node:main` | Processes raw Teensy IMU data. Dual-mode: NAVIGATION (pad taps become GUI nav commands) vs TRAINING (pad impacts become punch events). Mode switches automatically on session state. |
| **robot_node** | `boxbunny_core.robot_node:main` | Bridges BoxBunny commands to the V4 Arm Control GUI. Translates RobotCommand messages to strike_command JSON for the V4 GUI, forwards person direction to yaw motor, handles height commands. |
| **punch_processor** | `boxbunny_core.punch_processor:main` | Fuses CV detections with IMU impacts using pad-constraint filtering. Produces confirmed punches. Manages defense detection windows when the robot attacks. Publishes session punch summaries. |
| **session_manager** | `boxbunny_core.session_manager:main` | Manages training session lifecycle: countdown, active rounds, rest periods, completion. Accumulates all session data (punches, defense, tracking, CV events, IMU strikes). Publishes SessionState, which is the central signal for the entire system. |
| **drill_manager** | `boxbunny_core.drill_manager:main` | Loads 50 combo drill definitions from YAML. Validates detected punch sequences against expected combos. Tracks accuracy, timing, and streak. Publishes drill progress events. |
| **sparring_engine** | `boxbunny_core.sparring_engine:main` | Generates robot attack sequences via Markov-chain transition matrices. Five boxing styles (Boxer, Brawler, Counter-Puncher, Pressure, Switch). Difficulty-scaled intervals, idle-surprise attacks, weakness-bias targeting, reactive counter-punches. |
| **analytics_node** | `boxbunny_core.analytics_node:main` | Computes per-session statistics: punch/pad/impact distributions, fatigue index, defense rate, movement analysis. Publishes results as JSON for the dashboard. |
| **llm_node** | `boxbunny_core.llm_node:main` | Hosts a local Qwen 2.5 3B LLM on the Jetson GPU. Generates real-time coaching tips every 18 seconds during sessions, post-session analysis, and responds to chat queries. Degrades gracefully to pre-written fallback tips if the model is unavailable. |
| **gesture_node** | `boxbunny_core.gesture_node:main` | Uses MediaPipe Hands to detect hand gestures from the camera for GUI navigation. Disabled by default. Publishes NavCommand on the same topic as imu_node. |

### ROS 2 Topic Graph (ASCII Diagram)

```
                     ┌──────────────┐
                     │  RealSense   │
                     │  D435i       │
                     └──────┬───────┘
                   /camera/ │ color + depth
                            ▼
┌────────────┐      ┌──────────────┐      ┌────────────────┐
│ gesture_   │◄─────│   cv_node    │─────►│ punch_         │
│ node       │color │              │detect│ processor      │
└─────┬──────┘      │  (inference  │──────│                │
      │             │   engine)    │track │  (CV+IMU       │
      │             └──┬───┬───┬───┘      │   fusion)      │
      │                │   │   │          └──┬──┬──┬───────┘
      │nav_event   pose│   │   │person_dir   │  │  │
      ▼                │   │   ▼             │  │  │
┌──────────────┐       │   │  ┌──────────┐   │  │  │
│   GUI /      │       │   │  │ robot_   │   │  │  │
│   gui_bridge │       │   │  │ node     │◄──┘  │  │
│              │◄──────┘   │  │          │ cmd  │  │
└──────────────┘   debug   │  └─────┬────┘      │  │
                   info    │     strike│cmd      │  │
                           │        ▼           │  │
                           │  ┌──────────┐      │  │
                           │  │ V4 Arm   │      │  │
                           │  │ Control  │      │  │
                           │  │ GUI      │      │  │
                           │  └────┬─────┘      │  │
                           │    motor│cmd       │  │
                           │       ▼           │  │
                           │  ┌──────────┐      │  │
                           │  │ Teensy   │      │  │
                           │  │ 4.1      │──────┘  │
                           │  └──────────┘  pad/   │
                           │               arm     │
                           ▼                       │
                    ┌──────────────┐                │
                    │ session_     │◄───────────────┘
                    │ manager     │   confirmed/defense/summary
                    └──┬──┬───────┘
                       │  │
              state ───┘  │
                       │  └── summary
       ┌───────────────┼──────────────┐
       ▼               ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ imu_node   │ │ drill_     │ │ llm_node   │
│            │ │ manager    │ │ (AI Coach) │
└────────────┘ └────────────┘ └────────────┘
       ▲               │              │
       │               │              │tip
  pad_impact      drill_event         ▼
  arm_strike      drill_progress  ┌────────────┐
       │               │          │ GUI coach  │
       │               ▼          │ tip bar    │
┌──────────────┐ ┌────────────┐   └────────────┘
│ Teensy /     │ │ sparring_  │
│ Simulator    │ │ engine     │
└──────────────┘ └──────┬─────┘
                        │robot_cmd
                        ▼
                  ┌──────────┐
                  │ robot_   │
                  │ node     │
                  └──────────┘
```

### Detailed Topic Wiring Per Node

**cv_node**
- Subscribes: `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/boxbunny/session/state`
- Publishes: `/boxbunny/cv/detection`, `/boxbunny/cv/pose`, `/boxbunny/cv/user_tracking`, `/boxbunny/cv/person_direction`, `/boxbunny/cv/debug_info`, `/boxbunny/cv/status`

**imu_node**
- Subscribes: `/boxbunny/imu/pad/impact`, `/boxbunny/imu/arm/strike`, `/boxbunny/session/state`, `/robot/strike_detected`
- Publishes: `/boxbunny/imu/punch_event`, `/boxbunny/imu/nav_event`, `/boxbunny/imu/arm_event`, `/boxbunny/imu/status`

**punch_processor**
- Subscribes: `/boxbunny/cv/detection`, `/boxbunny/imu/punch_event`, `/boxbunny/imu/arm_event`, `/boxbunny/robot/command`, `/boxbunny/cv/user_tracking`, `/boxbunny/session/state`
- Publishes: `/boxbunny/punch/confirmed`, `/boxbunny/punch/defense`, `/boxbunny/punch/session_summary`

**robot_node**
- Subscribes: `/boxbunny/robot/command`, `/boxbunny/robot/height`, `/boxbunny/robot/round_control`, `motor_feedback`, `/robot/strike_feedback`, `/boxbunny/cv/person_direction`
- Publishes: `/robot/strike_command`, `/robot/system_enable`, `/robot/height_cmd`, `/robot/yaw_cmd`, `/boxbunny/robot/status`, `/boxbunny/robot/strike_complete`

**session_manager**
- Subscribes: `/boxbunny/punch/confirmed`, `/boxbunny/punch/defense`, `/boxbunny/cv/user_tracking`, `/boxbunny/session/config`, `/boxbunny/cv/detection`, `/boxbunny/imu/punch_event`, `/boxbunny/cv/person_direction`, `/boxbunny/robot/command`
- Publishes: `/boxbunny/session/state`, `/boxbunny/punch/session_summary`, `/boxbunny/robot/height`
- Services: `/boxbunny/session/start` (server), `/boxbunny/session/end` (server)

**sparring_engine**
- Subscribes: `/boxbunny/session/state`, `/boxbunny/punch/confirmed`, `/boxbunny/imu/punch_event`
- Publishes: `/boxbunny/robot/command`

**drill_manager**
- Subscribes: `/boxbunny/punch/confirmed`, `/boxbunny/session/state`
- Publishes: `/boxbunny/drill/definition`, `/boxbunny/drill/event`, `/boxbunny/drill/progress`
- Services: `/boxbunny/drill/start` (server)

**llm_node**
- Subscribes: `/boxbunny/session/state`, `/boxbunny/punch/confirmed`, `/boxbunny/drill/event`, `/boxbunny/punch/session_summary`
- Publishes: `/boxbunny/coach/tip`
- Services: `/boxbunny/llm/generate` (server)

**analytics_node**
- Subscribes: `/boxbunny/punch/confirmed`, `/boxbunny/punch/defense`, `/boxbunny/punch/session_summary`, `/boxbunny/session/state`
- Publishes: `/boxbunny/analytics/session`

**gesture_node**
- Subscribes: `/camera/color/image_raw`, `/boxbunny/session/state`
- Publishes: `/boxbunny/imu/nav_event` (same topic as imu_node), `/boxbunny/gesture/status`

---

## 4. Data Flow Diagrams

### 4a. Punch Detection Flow

This is the primary data pipeline: from camera frame to confirmed punch.

```
Camera RGB+Depth (30fps)
         │
         ▼
   ┌──────────────────────────────────────────────────┐
   │ cv_node                                          │
   │                                                  │
   │  RGB ──► YOLO Pose ──► 7 upper-body keypoints    │
   │  Depth ──► Voxel Grid ──► 12x12x12 occupancy     │
   │                                                  │
   │  [keypoints + voxels] ──► FusionTransformerModel  │
   │                           (TensorRT FP16)        │
   │                                                  │
   │  Raw logits ──► EMA smoothing (alpha=0.35)       │
   │              ──► Hysteresis filter (margin=0.12)  │
   │              ──► State machine (enter/exit/hold)  │
   │              ──► Block consecutive filter         │
   │                                                  │
   │  Result: PunchDetection {type, confidence}       │
   └───────────────────┬──────────────────────────────┘
                       │ /boxbunny/cv/detection
                       ▼
   ┌──────────────────────────────────────────────────┐
   │ punch_processor                                  │
   │                                                  │
   │  CV buffer: stores recent non-idle predictions   │
   │  with timestamps (800ms window)                  │
   │                                                  │
   │  On IMU pad strike:                              │
   │    1. Lookup pad constraints                     │
   │    2. Search CV buffer for valid predictions     │
   │    3. Count frames per valid type                │
   │    4. Pick dominant prediction (most frames)     │
   │    5. Emit ConfirmedPunch                        │
   │                                                  │
   │  PAD CONSTRAINT TABLE:                           │
   │  ┌─────────┬──────────────────────────────────┐  │
   │  │  Pad    │  Valid Punch Types               │  │
   │  ├─────────┼──────────────────────────────────┤  │
   │  │ Centre  │ jab, cross                       │  │
   │  │ Left    │ left_hook, left_uppercut          │  │
   │  │ Right   │ right_hook, right_uppercut        │  │
   │  │ Head    │ all offensive punches             │  │
   │  └─────────┴──────────────────────────────────┘  │
   └───────────────────┬──────────────────────────────┘
                       │ /boxbunny/punch/confirmed
                       ▼
            ConfirmedPunch message:
            {timestamp, punch_type, pad, level, force_normalized,
             cv_confidence, imu_confirmed, cv_confirmed, accel_magnitude}
```

### 4b. Robot Arm Control Flow

Commands flow from the sparring engine through multiple bridge layers to the physical motors.

```
   ┌───────────────┐
   │ sparring_     │  Markov chain selects next punch
   │ engine        │  based on style, difficulty, weakness bias
   └───────┬───────┘
           │ RobotCommand {command_type="punch", punch_code="3", speed="medium"}
           │ /boxbunny/robot/command
           ▼
   ┌───────────────┐
   │ robot_node    │  Maps punch_code → slot, speed_string → rad/s
   │               │  Publishes JSON: {"slot": 3, "duration": 5.0, "speed": 15.0}
   └───────┬───────┘
           │ /robot/strike_command (String JSON)
           ▼
   ┌───────────────┐
   │ V4 Arm        │  FSM strike execution:
   │ Control GUI   │  alignment → windup → apex → snap-back
   │               │  Motor safety, calibration, current limiting
   └───────┬───────┘
           │ motor_commands (Float64MultiArray)
           │ [pos0-3, speed0-3, enable_flag] = 9 doubles
           ▼
   ┌───────────────┐
   │ Teensy 4.1    │  micro-ROS agent
   │ (firmware)    │  DYNAMIXEL motor control at 100Hz
   └───────┬───────┘
           │ motor_feedback (Float64MultiArray)
           │ [pos0-3, current0-3, can_rx, imu0_xyz...imu3_xyz] = 21 doubles
           ▼
   ┌───────────────┐
   │ robot_node    │  Receives feedback, updates connection state
   │               │  Publishes /boxbunny/robot/strike_complete
   └───────────────┘

   SIMULTANEOUSLY (defense detection):
   
   punch_processor sees the RobotCommand and opens a
   500ms defense window. During this window:
     - arm IMU events (contact/miss) are collected
     - CV block detections are collected  
     - user tracking displacements are collected
   At window close: classify_defense() determines:
     HIT, BLOCK, SLIP, DODGE, or UNKNOWN
```

### 4c. Session Data Flow

From real-time punches to persistent storage and mobile dashboard.

```
   ┌──────────────────┐     ┌──────────────────┐
   │ punch_processor  │     │ cv_node          │
   │ (confirmed +     │     │ (predictions,    │
   │  defense events) │     │  tracking, dir)  │
   └────────┬─────────┘     └────────┬─────────┘
            │                         │
            ▼                         ▼
   ┌──────────────────────────────────────────────┐
   │ session_manager                              │
   │                                              │
   │  Accumulates during active session:          │
   │  - Confirmed punches (type, pad, force, ts)  │
   │  - Defense events (type, arm, struck)         │
   │  - CV prediction events (grouped by type)     │
   │  - Raw IMU strikes (pad, level, accel)        │
   │  - Person direction changes (with duration)   │
   │  - User depth + lateral displacement          │
   │  - Defense reaction times (experimental)      │
   │                                              │
   │  Lifecycle: idle → countdown → active → rest  │
   │             → ... → active → complete → idle  │
   │                                              │
   │  On complete: build_summary() → publish       │
   └──────────┬───────────────────────────────────┘
              │ SessionPunchSummary
              │ /boxbunny/punch/session_summary
              ▼
   ┌──────────────────┐   ┌──────────────────┐
   │ analytics_node   │   │ llm_node         │
   │ (fatigue index,  │   │ (post-session    │
   │  distributions,  │   │  analysis)       │
   │  trends)         │   │                  │
   └────────┬─────────┘   └────────┬─────────┘
            │                       │
            ▼                       ▼
   ┌──────────────────────────────────────────────┐
   │ SQLite Database (data/boxbunny_main.db)      │
   │ - users, sessions, round_data, punch_log     │
   │ - per-user data in data/users/               │
   └──────────────────┬───────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────┐
   │ FastAPI Dashboard Server (:8080)             │
   │ - REST API: /api/sessions, /api/users, ...   │
   │ - WebSocket: live punch feed                  │
   │ - LLM chat endpoint via GenerateLlm service   │
   └──────────────────┬───────────────────────────┘
                      │ Wi-Fi AP (BoxBunny/boxbunny2026)
                      ▼
   ┌──────────────────────────────────────────────┐
   │ Vue.js 3 Phone Dashboard                     │
   │ - Session history with charts                 │
   │ - Live round monitoring                       │
   │ - AI Coach chat interface                     │
   │ - User profile and statistics                 │
   └──────────────────────────────────────────────┘
```

---

## 5. Configuration System

BoxBunny follows a strict configuration philosophy: **all tuneable values live in YAML files, never as magic numbers in code**. There are two configuration files and one Python module that ties them together.

### 5a. `config/boxbunny.yaml` -- Master Configuration

The master config file contains every tuneable parameter in the system, organized by subsystem. Each value has a comment explaining its purpose. The file maps to a hierarchy of Python dataclasses defined in `config_loader.py`:

```python
@dataclass
class BoxBunnyConfig:
    cv: CVConfig           # CV pipeline: confidence, smoothing, state machine
    fusion: FusionConfig   # CV+IMU fusion: windows, thresholds, pad constraints
    imu: IMUConfig         # IMU debounce, mode transition timing
    robot: RobotConfig     # Serial port, heartbeat, punch sequences
    llm: LLMConfig         # Model path, GPU layers, token limits
    height: HeightConfig   # Auto-adjustment: deadband, depth limits
    training: TrainingConfig  # Default rounds, work/rest times, speed
    free_training: FreeTrainingConfig  # Counter-punch mapping, cooldown
    network: NetworkConfig  # Wi-Fi AP, dashboard port
    database: DatabaseConfig  # SQLite paths, guest TTL
```

Key configuration sections:

| Section | Key Parameters | Default Values |
|---|---|---|
| `cv` | `min_confidence`, `ema_alpha`, `hysteresis_margin`, `min_hold_frames` | 0.4, 0.35, 0.12, 3 |
| `fusion` | `fusion_window_ms`, `imu_impact_threshold`, `defense_window_ms` | 500, 5.0, 500 |
| `imu` | `nav_debounce_ms`, `nav_global_debounce_ms`, `mode_transition_ms` | 300, 200, 200 |
| `robot` | `serial_port`, `baud_rate`, `heartbeat_hz` | /dev/ttyACM0, 115200, 10.0 |
| `llm` | `model_path`, `n_gpu_layers`, `n_ctx`, `max_tokens` | qwen2.5-3b Q4_K_M, -1, 2048, 128 |
| `training` | `default_rounds`, `default_work_time_s`, `default_rest_time_s` | 3, 180, 60 |
| `free_training` | `pad_counter_strikes`, `counter_cooldown_ms` | pad-to-punch map, 1500 |

### 5b. `config/ros_topics.yaml` -- ROS Topic Names

Every ROS topic and service name in the system is defined in this single file. The file is self-documenting: each entry includes the message type, publisher node, subscriber nodes, field descriptions, and behavioral notes.

The file is organized into sections:

| Section | Topics | Purpose |
|---|---|---|
| `imu` | 6 topics | Raw IMU data (from Teensy) and processed events (from imu_node) |
| `cv` | 6 topics | CV detections, pose, tracking, direction, status, debug |
| `punch` | 3 topics | Confirmed punches, defense events, session summary |
| `robot` | 12 topics | Commands, height, feedback, motor data, strike protocol |
| `session` | 2 topics | Session state and configuration |
| `drill` | 3 topics | Drill definition, events, progress |
| `coach` | 1 topic | AI coaching tips |
| `gesture` | 1 topic | Gesture control status |
| `camera` | 2 topics | RealSense RGB and depth streams |
| `services` | 6 services | Start/end session, start drill, set IMU mode, calibrate, LLM generate |

### 5c. `constants.py` -- Runtime Loader

The `constants.py` module loads `ros_topics.yaml` at import time and exposes all topic names, service names, and domain constants as Python class attributes:

```python
from boxbunny_core.constants import Topics, Services, PunchType, PadLocation

# Topic names are resolved from YAML with fallback defaults:
Topics.CV_DETECTION        # "/boxbunny/cv/detection"
Topics.PUNCH_CONFIRMED     # "/boxbunny/punch/confirmed"
Services.START_SESSION     # "/boxbunny/session/start"

# Domain constants:
PunchType.ALL_ACTIONS      # ["jab", "cross", "left_hook", ...]
PunchType.CODE_MAP         # {"1": "jab", "2": "cross", ...}
PadLocation.VALID_PUNCHES  # {"centre": ["jab", "cross"], ...}
ImpactLevel.FORCE_MAP      # {"light": 0.33, "medium": 0.66, "hard": 1.0}
```

This architecture means renaming a topic is a one-line YAML edit with zero Python changes. The fallback defaults in `_t()` ensure the system works even if the YAML file is missing.

---

## 6. Design Philosophy

### Single-Responsibility Nodes

Each ROS node does exactly one thing. The CV node does not know about IMU data. The IMU node does not know about the CV model. The punch_processor fuses them. The session_manager does not care how punches are detected -- it just consumes ConfirmedPunch messages. This makes the system testable in isolation: any node can be replaced with a mock publisher.

### Config-Driven Architecture

No magic numbers in code. Every threshold, interval, path, and mapping is in a YAML file. This is critical for a training robot where parameters need tuning based on user feedback, hardware changes, or different deployment environments. The `config_loader.py` dataclasses provide type safety and default values.

### Layered Data Processing

Data flows through well-defined layers of increasing abstraction:

1. **Raw**: Camera frames, IMU accelerometer readings, motor positions
2. **Detected**: CV predictions (per-frame), pad impacts (per-event)
3. **Fused**: Confirmed punches (CV + IMU agreement), defense events (arm + CV + tracking)
4. **Aggregated**: Session summaries (distributions, rates, trends)
5. **Analyzed**: AI coaching tips, fatigue index, weakness profiles

Each layer is a separate ROS topic, allowing any consumer to tap in at the appropriate level.

### Graceful Degradation

The system is designed to work with partial hardware:

- **No camera**: IMU-only punches (pad-inferred type, lower confidence)
- **No IMU pads**: CV-only detections (higher confidence threshold required)
- **No LLM model**: Fallback tips from `config/fallback_tips.json`
- **No Teensy**: Teensy Simulator provides identical ROS messages
- **No V4 GUI**: Teensy Simulator can auto-execute strikes with simulated feedback

### The SessionState Signal

`/boxbunny/session/state` is the single most important topic in the system. When it changes:

| Transition | Effect |
|---|---|
| idle -> countdown | IMU switches to TRAINING mode (200ms grace period). Height auto-adjustment triggers. |
| countdown -> active | Round timer starts. Sparring engine activates. Drill manager starts accepting punches. |
| active -> rest | Round ends. Punches stop counting. Rest timer starts. |
| rest -> countdown | Next round countdown begins. |
| active/rest -> complete | Session summary is published. Analytics computed. LLM generates post-session analysis. |
| complete -> idle | IMU switches back to NAVIGATION mode. Gesture control resumes. |

This event-driven design means adding a new node (e.g., a visualization tool) requires only subscribing to SessionState -- no modifications to existing nodes.

### Motor Safety

Motor speeds are capped at 30 rad/s for gear safety (defined in `MotorSpeed.MAX`). The V4 Arm Control GUI handles the FSM strike execution (alignment, windup, apex, snap-back) with current limiting. The `robot_node` bridge layer enforces the speed cap before forwarding commands. The round_control topic enables/disables motors at round boundaries.

---

## Appendix: File Structure

```
boxing_robot_ws/
├── config/
│   ├── boxbunny.yaml        # Master configuration
│   ├── ros_topics.yaml       # All ROS topic/service names
│   ├── drills.yaml           # 50 combo drill definitions
│   └── fallback_tips.json    # LLM fallback coaching tips
├── src/
│   └── boxbunny_core/
│       ├── setup.py          # Package definition + 10 entry points
│       ├── boxbunny_core/
│       │   ├── constants.py      # Topics, Services, PunchType, PadLocation, ...
│       │   ├── config_loader.py  # YAML → typed dataclasses
│       │   ├── cv_node.py        # Camera → inference → detections
│       │   ├── imu_node.py       # Teensy IMU → events
│       │   ├── robot_node.py     # Commands → V4 GUI bridge
│       │   ├── punch_processor.py # CV+IMU fusion
│       │   ├── punch_fusion.py   # Fusion helpers, ring buffer, defense classification
│       │   ├── session_manager.py # Session lifecycle
│       │   ├── drill_manager.py  # Combo drill validation
│       │   ├── sparring_engine.py # Markov-chain attack generation
│       │   ├── analytics_node.py # Session statistics
│       │   ├── llm_node.py       # On-device AI coaching
│       │   └── gesture_node.py   # MediaPipe hand gesture navigation
│       └── package.xml
├── action_prediction/        # Standalone CV model (see cv_pipeline.md)
│   ├── lib/
│   │   ├── fusion_model.py       # FusionVoxelPoseTransformerModel
│   │   ├── voxel_model.py        # Conv3DStem, PositionalEncoding
│   │   ├── voxel_features.py     # Depth → 3D voxel grid
│   │   ├── pose.py               # YOLO Pose wrapper
│   │   └── inference_runtime.py  # Headless inference engine for cv_node
│   └── model/
│       ├── best_model.pth        # Trained weights (v5, 96.6% val acc)
│       └── yolo26n-pose.engine   # TensorRT YOLO Pose
├── tools/
│   └── teensy_simulator.py   # Hardware simulator (see teensy_simulator.md)
├── notebooks/
│   └── scripts/
│       ├── run_with_ros.py       # CV inference GUI + ROS bridge
│       └── fusion_monitor.py     # IMU+CV fusion debug monitor
├── data/
│   ├── boxbunny_main.db      # SQLite database
│   └── users/                # Per-user data
└── docs/                     # This documentation
```
