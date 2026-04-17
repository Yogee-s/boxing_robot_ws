# BoxBunny System Architecture Reference

> Complete reference for recreating all system diagrams in any tool (draw.io, Figma, Lucidchart, etc.)

---

## 1. System Overview

- **12 ROS 2 nodes** on Jetson Orin NX (cv, imu, punch_processor, session_manager, drill_manager, sparring_engine, free_training_engine, analytics, llm, robot, gesture, ble_bridge)
- **22+ custom message types**, **8 services**, **45+ topics**
- **1 PySide6 GUI** (touchscreen) bridged to ROS via background thread
- **1 FastAPI dashboard** (phone) communicating via IPC files + 1 ROS service
- **Hardware:** RealSense D435i, Teensy 4.0 (arms — 4 pad IMUs, 2 arm IMUs, 4 Damiao motors), Arduino Uno R4 WiFi (base — rotation + height motors via BLE), 7" touchscreen

---

## 2. All ROS 2 Nodes

### 2.1 cv_node — Action Recognition
- **Subscribes:** RealSense camera (USB 3.0 direct, not ROS driver) — RGB 960x540 + Depth 640x480 @30fps
- **Publishes:**
  - `/boxbunny/cv/detection` (PunchDetection) → punch_processor, GUI
  - `/boxbunny/cv/pose` (PoseEstimate) → reaction test
  - `/boxbunny/cv/user_tracking` (UserTracking) → punch_processor, session_manager
  - `/boxbunny/cv/person_direction` (String) → robot_node, session_manager
  - `/boxbunny/cv/status` (String) → GUI
  - `/boxbunny/cv/debug_info` (String JSON) → GUI debug panel
- **Pipeline:** YOLO Pose (TensorRT FP16, ~6ms) → Voxel Grid 12^3 → Pose Features 42-dim → Temporal Window 12 frames (400ms) → FusionVoxelPose Transformer (TensorRT FP16, ~8ms) → EMA + Hysteresis + State Machine

### 2.2 imu_node — IMU Processing
- **Subscribes:**
  - `/boxbunny/imu/pad/impact` (PadImpact) ← Teensy (4 pads: centre, left, right, head)
  - `/boxbunny/imu/arm/strike` (ArmStrike) ← Teensy (2 arms)
  - `/boxbunny/session/state` (SessionState) ← session_manager (mode switching)
  - `/robot/strike_detected` (String JSON) ← V4 GUI
- **Publishes:**
  - `/boxbunny/imu/punch_event` (PunchEvent) → punch_processor *(TRAINING mode only)*
  - `/boxbunny/imu/nav_event` (NavCommand) → GUI *(NAVIGATION mode only)*
  - `/boxbunny/imu/arm_event` (ArmStrikeEvent) → punch_processor *(always)*
  - `/boxbunny/imu/status` (IMUStatus) → GUI
- **Services:**
  - Server: `/boxbunny/imu/set_mode` (SetImuMode)
  - Server: `/boxbunny/imu/calibrate` (CalibrateImuPunch)
- **Timers:** Status heartbeat 1 Hz, Transition check 20 Hz
- **Key params:** nav_debounce_ms=500, nav_global_debounce_ms=300, mode_transition_ms=200, threshold=5.0 m/s^2

### 2.3 punch_processor — CV + IMU Fusion
- **Subscribes:**
  - `/boxbunny/cv/detection` (PunchDetection) ← cv_node
  - `/boxbunny/imu/punch_event` (PunchEvent) ← imu_node
  - `/boxbunny/imu/arm_event` (ArmStrikeEvent) ← imu_node
  - `/boxbunny/robot/command` (RobotCommand) ← sparring/free_training (opens defense window)
  - `/boxbunny/cv/user_tracking` (UserTracking) ← cv_node (slip/dodge detection)
  - `/boxbunny/session/state` (SessionState) ← session_manager
- **Publishes:**
  - `/boxbunny/punch/confirmed` (ConfirmedPunch) → session_manager, GUI, analytics, drill_mgr, llm, sparring
  - `/boxbunny/punch/defense` (DefenseEvent) → session_manager, GUI, analytics
  - `/boxbunny/punch/session_summary` (SessionPunchSummary) → session_manager, analytics, llm
- **Timer:** Expiry check 20 Hz
- **Fusion params:** fusion_window=500ms, defense_window=500ms, slip_lateral=40px, slip_depth=0.15m, dodge_lateral=20px, dodge_depth=0.08m, block_cv_min=0.3, cv_only: >=3 frames >=0.7 conf (penalty -0.3), imu_only: default conf 0.3

### 2.4 session_manager — Session Lifecycle
- **Subscribes:**
  - `/boxbunny/punch/confirmed` (ConfirmedPunch) ← punch_processor
  - `/boxbunny/punch/defense` (DefenseEvent) ← punch_processor
  - `/boxbunny/cv/user_tracking` (UserTracking) ← cv_node
  - `/boxbunny/session/config` (SessionConfig) ← GUI
  - `/boxbunny/cv/detection` (PunchDetection) ← cv_node
  - `/boxbunny/imu/punch_event` (PunchEvent) ← imu_node
  - `/boxbunny/cv/person_direction` (String) ← cv_node
  - `/boxbunny/robot/command` (RobotCommand) ← sparring/free_training
- **Publishes:**
  - `/boxbunny/session/state` (SessionState) → ALL nodes, GUI — states: idle, countdown, active, rest, complete
  - `/boxbunny/punch/session_summary` (SessionPunchSummary) → analytics, llm
  - `/boxbunny/robot/height` (HeightCommand) → robot_node (auto-adjust during countdown)
  - `/boxbunny/session/config_json` (String) → sparring_engine, free_training_engine
- **Services:**
  - Server: `/boxbunny/session/start` (StartSession)
  - Server: `/boxbunny/session/end` (EndSession)
- **Timers:** Main tick 1 Hz, Autosave 10s, State publish 2 Hz
- **Database:** Writes session data to per-user SQLite DB

### 2.5 drill_manager — Combo Validation
- **Subscribes:**
  - `/boxbunny/punch/confirmed` (ConfirmedPunch) ← punch_processor
  - `/boxbunny/session/state` (SessionState) ← session_manager
- **Publishes:**
  - `/boxbunny/drill/definition` (DrillDefinition) → GUI
  - `/boxbunny/drill/event` (DrillEvent) → session_manager, GUI, llm
  - `/boxbunny/drill/progress` (DrillProgress) → GUI
- **Services:**
  - Server: `/boxbunny/drill/start` (StartDrill)
- **Timer:** Timeout check 2 Hz
- **Config:** 50+ drills from config/drills.yaml

### 2.6 sparring_engine — Sparring AI
- **Subscribes:**
  - `/boxbunny/session/state` (SessionState) ← session_manager
  - `/boxbunny/punch/confirmed` (ConfirmedPunch) ← punch_processor
  - `/boxbunny/imu/punch_event` (PunchEvent) ← imu_node (reactive counters)
  - `/boxbunny/session/config_json` (String) ← session_manager
  - `/robot/strike_feedback` (String JSON) ← V4 GUI
- **Publishes:**
  - `/boxbunny/robot/command` (RobotCommand) → robot_node
- **Timer:** Main tick 10 Hz
- **AI:** 5 Markov-chain boxing styles (boxer, brawler, counter-puncher, pressure, switch), 3 difficulty tiers (easy 2s, medium 1.2s, hard 0.6s intervals), counter-punch probability (easy:0.3, medium:0.5, hard:0.8)

### 2.7 free_training_engine — Reactive Mode
- **Subscribes:**
  - `/boxbunny/session/state` (SessionState) ← session_manager
  - `/boxbunny/imu/punch_event` (PunchEvent) ← imu_node (trigger counter)
  - `/robot/strike_feedback` (String JSON) ← V4 GUI
  - `/boxbunny/session/config_json` (String) ← session_manager
- **Publishes:**
  - `/boxbunny/robot/command` (RobotCommand) → robot_node
- **Counter mappings:** centre→jab/cross, left→left_hook/left_uppercut, right→right_hook/right_uppercut, head→jab/cross
- **Params:** counter_cooldown_ms=300, idle_return_s=5.0

### 2.8 analytics_node — Statistics
- **Subscribes:**
  - `/boxbunny/punch/confirmed` (ConfirmedPunch) ← punch_processor
  - `/boxbunny/punch/defense` (DefenseEvent) ← punch_processor
  - `/boxbunny/punch/session_summary` (SessionPunchSummary) ← punch_processor
  - `/boxbunny/session/state` (SessionState) ← session_manager
- **Publishes:**
  - `/boxbunny/analytics/session` (String JSON) → GUI, dashboard
- **Timer:** Periodic publish 5 Hz
- **Stats:** Punch distribution, fatigue index, defense rate, movement, trends, personal records

### 2.9 llm_node — AI Coach
- **Subscribes:**
  - `/boxbunny/session/state` (SessionState) ← session_manager
  - `/boxbunny/punch/confirmed` (ConfirmedPunch) ← punch_processor
  - `/boxbunny/drill/event` (DrillEvent) ← drill_manager
  - `/boxbunny/punch/session_summary` (SessionPunchSummary) ← punch_processor
- **Publishes:**
  - `/boxbunny/coach/tip` (CoachTip) → GUI (every ~18s during active)
- **Services:**
  - Server: `/boxbunny/llm/generate` (GenerateLlm) — used by GUI chat + dashboard chat API
- **Model:** Qwen 2.5 3B Q4 (GGUF), GPU accelerated, 2048 context, 200 max tokens, temp 0.7, timeout 20s
- **Fallback:** Pre-written tips from config/fallback_tips.json

### 2.10 robot_node — Motor Bridge
- **Subscribes:**
  - `/boxbunny/robot/command` (RobotCommand) ← sparring_engine, free_training_engine, GUI
  - `/boxbunny/robot/height` (HeightCommand) ← session_manager, GUI
  - `/boxbunny/robot/round_control` (RoundControl) ← session_manager
  - `/motor_feedback` (Float64MultiArray) ← Teensy
  - `/robot/strike_feedback` (String JSON) ← V4 GUI
  - `/boxbunny/cv/person_direction` (String) ← cv_node (yaw tracking)
- **Publishes:**
  - `/robot/strike_command` (String JSON) → V4 GUI
  - `/robot/system_enable` (String) → V4 GUI
  - `/robot/height_cmd` (String) → Teensy
  - `/robot/yaw_cmd` (String) → Teensy
  - `/boxbunny/robot/status` (String JSON) → GUI
  - `/boxbunny/robot/strike_complete` (String JSON) → GUI
- **Timer:** Status 1 Hz
- **Hardware:** Dynamixel servos (4 motors), height motor, yaw motor — all via Teensy

### 2.12 ble_bridge_node — Base Arduino BLE Bridge
- **Subscribes:**
  - `/boxbunny/robot/height` (HeightCommand) ← GUI settings, phone dashboard (translates to `HUP:/HDOWN:/HSTOP`)
  - `/boxbunny/cv/person_direction` (String) ← cv_node (forwards as `L:/R:/S` when tracking enabled)
  - `/boxbunny/cv/user_tracking` (UserTracking) ← cv_node (depth gating for tracking)
- **Publishes:**
  - `/boxbunny/ble/status` (String JSON, 1 Hz) → GUI (connected/tracking/last_cmd state)
- **Services:**
  - Server: `/boxbunny/ble/tracking_start` (std_srvs/Trigger) — enable CV→BLE rotation forwarding
  - Server: `/boxbunny/ble/tracking_stop` (std_srvs/Trigger) — send `S` and disable forwarding
- **Timers:** Status 1 Hz, height deadman 10 Hz
- **Transport:** BLE (bleak async client on a daemon thread) to Arduino Uno R4 WiFi advertising as "BoxBunny Base"
- **Near-range gating:** `near_range_m=0.8` with `hysteresis_m=0.15` (both in `config/boxbunny.yaml`)
- **Safety:** 500 ms Jetson deadman on height, paired with 250 ms firmware-side refresh-window watchdog in `ble_control.ino`
- **Single-owner:** classic BLE peripheral accepts one central at a time. Do not run `notebooks/scripts/test_base_tracking.py` alongside this node.

### 2.11 gesture_node — Hand Gesture Recognition (disabled by default)
- **Subscribes:**
  - `/camera/color/image_raw` (Image BGR8) ← cv_node/camera
  - `/boxbunny/session/state` (SessionState) ← session_manager
- **Publishes:**
  - `/boxbunny/imu/nav_event` (NavCommand) → GUI (same topic as pad nav)
  - `/boxbunny/gesture/status` (String) → GUI
- **Timer:** Status 1 Hz
- **Gestures:** Open palm→enter, Thumbs up→enter, Peace→back, Swipe left/right→prev/next
- **Params:** enabled=False, hold_duration=0.7s, cooldown=1.5s, min_confidence=0.7

---

## 3. Non-ROS Components

### 3.1 BoxBunny GUI (PySide6)
- **ROS Bridge:** Background QThread runs ROS 2 node `boxbunny_gui`
- **Subscribes (via bridge):**
  - `/boxbunny/punch/confirmed`, `/boxbunny/punch/defense`
  - `/boxbunny/drill/progress`, `/boxbunny/session/state`
  - `/boxbunny/coach/tip`, `/boxbunny/imu/nav_event`
  - `/boxbunny/imu/status`, `/boxbunny/robot/strike_complete`
  - `/boxbunny/cv/detection`, `/boxbunny/cv/debug_info`
  - `/boxbunny/cv/user_tracking` (gates reaction test against background walkers)
  - `/boxbunny/ble/status` (enables tracking-button when link is up)
  - `/boxbunny/robot/command` (counter-punch tag — sparring UI uses this for the COUNTERS counter)
- **Publishes (via bridge):**
  - `/boxbunny/robot/command` (RobotCommand)
  - `/boxbunny/robot/height` (HeightCommand)
- **Service clients:**
  - `/boxbunny/session/start` (StartSession)
  - `/boxbunny/session/end` (EndSession)
  - `/boxbunny/llm/generate` (GenerateLlm)
  - `/boxbunny/ble/tracking_start` + `/boxbunny/ble/tracking_stop` (Trigger)
  - `/boxbunny/llm/generate` (GenerateLlm)
- **Input:** Touch, IMU pad navigation, keyboard fallback
- **Display:** 1024x600, 24 pages (home, training, sparring, results, settings, etc.)
- **DB access:** Direct SQLite reads for auth, presets, history

### 3.2 FastAPI Dashboard
- **Port:** 8080, WiFi AP "BoxBunny"
- **Frontend:** Vue 3 SPA (Pinia stores, Tailwind CSS)
- **API routes:** /api/auth, /api/sessions, /api/chat, /api/remote, /api/gamification, /api/presets, /api/export, /api/coach
- **WebSocket:** /ws (live session data)
- **ROS connection:** Service client for `/boxbunny/llm/generate` (chat only)
- **IPC to GUI:** Writes JSON to /tmp/ files, GUI polls every 100ms
  - `/tmp/boxbunny_gui_command.json` (remote control)
  - `/tmp/boxbunny_gui_login.json` (phone login notification)
  - `/tmp/boxbunny_height_cmd.json` (height adjustment)
- **DB access:** Direct SQLite reads/writes for auth, sessions, gamification, presets

---

## 4. Complete Topic List (40+ Topics)

### IMU Topics
| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/boxbunny/imu/pad/impact` | PadImpact | Teensy (micro-ROS) | imu_node |
| `/boxbunny/imu/arm/strike` | ArmStrike | Teensy (micro-ROS) | imu_node |
| `/boxbunny/imu/punch_event` | PunchEvent | imu_node | punch_processor, sparring, free_training |
| `/boxbunny/imu/nav_event` | NavCommand | imu_node, gesture_node | GUI |
| `/boxbunny/imu/arm_event` | ArmStrikeEvent | imu_node | punch_processor |
| `/boxbunny/imu/status` | IMUStatus | imu_node | GUI |

### CV Topics
| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/boxbunny/cv/detection` | PunchDetection | cv_node | punch_processor, session_manager, GUI |
| `/boxbunny/cv/pose` | PoseEstimate | cv_node | reaction test |
| `/boxbunny/cv/user_tracking` | UserTracking | cv_node | punch_processor, session_manager |
| `/boxbunny/cv/person_direction` | String | cv_node | robot_node, session_manager |
| `/boxbunny/cv/status` | String | cv_node | GUI |
| `/boxbunny/cv/debug_info` | String JSON | cv_node | GUI |

### Punch / Fusion Topics
| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/boxbunny/punch/confirmed` | ConfirmedPunch | punch_processor | session_manager, GUI, analytics, drill_mgr, llm, sparring |
| `/boxbunny/punch/defense` | DefenseEvent | punch_processor | session_manager, GUI, analytics |
| `/boxbunny/punch/session_summary` | SessionPunchSummary | punch_processor, session_manager | analytics, llm |

### Robot Topics
| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/boxbunny/robot/command` | RobotCommand | sparring, free_training, GUI | robot_node, punch_processor |
| `/boxbunny/robot/height` | HeightCommand | session_manager, GUI | robot_node |
| `/boxbunny/robot/round_control` | RoundControl | session_manager | robot_node |
| `/boxbunny/robot/status` | String JSON | robot_node | GUI |
| `/boxbunny/robot/strike_complete` | String JSON | robot_node | GUI |
| `/motor_feedback` | Float64MultiArray | Teensy | robot_node |
| `/robot/strike_command` | String JSON | robot_node | V4 GUI |
| `/robot/strike_feedback` | String JSON | V4 GUI | robot_node, sparring, free_training |
| `/robot/height_cmd` | String | robot_node | Teensy |
| `/robot/yaw_cmd` | String | robot_node | Teensy |
| `/robot/system_enable` | String | robot_node | V4 GUI |
| `/robot/strike_detected` | String JSON | V4 GUI | imu_node |

### Session Topics
| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/boxbunny/session/state` | SessionState | session_manager | ALL nodes, GUI |
| `/boxbunny/session/config` | SessionConfig | GUI | session_manager |
| `/boxbunny/session/config_json` | String JSON | session_manager | sparring, free_training |

### Drill Topics
| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/boxbunny/drill/definition` | DrillDefinition | drill_manager | GUI |
| `/boxbunny/drill/event` | DrillEvent | drill_manager | session_manager, GUI, llm |
| `/boxbunny/drill/progress` | DrillProgress | drill_manager | GUI |

### Coach / Analytics Topics
| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/boxbunny/coach/tip` | CoachTip | llm_node | GUI |
| `/boxbunny/analytics/session` | String JSON | analytics_node | GUI, dashboard |

### Camera Topics
| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/camera/color/image_raw` | Image BGR8 | camera driver | gesture_node |

---

## 5. Services

| Service | Type | Server | Clients |
|---|---|---|---|
| `/boxbunny/session/start` | StartSession | session_manager | GUI, dashboard |
| `/boxbunny/session/end` | EndSession | session_manager | GUI, dashboard |
| `/boxbunny/drill/start` | StartDrill | drill_manager | session_manager |
| `/boxbunny/imu/set_mode` | SetImuMode | imu_node | (manual tools) |
| `/boxbunny/imu/calibrate` | CalibrateImuPunch | imu_node | GUI calibration |
| `/boxbunny/llm/generate` | GenerateLlm | llm_node | GUI, dashboard chat API |

---

## 6. Hardware Connections

| Hardware | Interface | Connected Node | Details |
|---|---|---|---|
| Intel RealSense D435i | USB 3.0 (PyRealsense2, no ROS driver) | cv_node | RGB 960x540 + Depth 640x480 @30fps |
| Teensy 4.1 | micro-ROS serial bridge (115200 baud) | imu_node, robot_node | 4 pad IMUs, 2 arm IMUs, motors |
| 4x Pad Accelerometers | Teensy I2C | imu_node | Centre, Left, Right, Head @100Hz |
| 2x Arm Accelerometers | Teensy I2C | imu_node | Left arm, Right arm (contact detect) |
| 4x Dynamixel Servos | Teensy PWM via V4 GUI | robot_node | 2 joints x 2 arms |
| Height Motor | Teensy PWM | robot_node | Linear actuator |
| Yaw Motor | Teensy PWM | robot_node | Base rotation |
| 7" Touchscreen (1024x600) | HDMI + USB touch | GUI (PySide6) | Direct Qt rendering |
| Mobile Phone | WiFi AP → HTTP :8080 | dashboard (FastAPI) | Vue 3 SPA in browser |

---

## 7. State Machine (Session States)

```
IDLE ──StartSession──→ COUNTDOWN ──3s──→ ACTIVE ──round timer──→ REST ──rest timer──→ COUNTDOWN
                                          │                                              │
                                          ├──final round──→ COMPLETE ──auto 3s──→ IDLE   │
                                          └──EndSession───→ COMPLETE                     │
                                                                                         │
                                                            (loop back for multi-round)──┘
```

**Mode switching:** IDLE/COMPLETE → IMU NAVIGATION mode | COUNTDOWN/ACTIVE → IMU TRAINING mode (200ms grace)

---

## 8. Data Flow Summary (for diagram creation)

### Primary Pipeline (top to bottom):
```
RealSense D435i ──USB──→ cv_node ──/cv/detection──→ punch_processor ──/punch/confirmed──→ session_manager
Teensy 4.1 ──micro-ROS──→ imu_node ──/imu/punch_event──→ punch_processor                    │
                                                                                              ↓
                                                          drill_manager ←── /punch/confirmed  │
                                                          sparring_engine ←── /punch/confirmed │
                                                          analytics_node ←── /session_summary  │
                                                          llm_node ←── /punch/confirmed        │
                                                                                              ↓
                                                          GUI ←── /session/state, /coach/tip, /drill/progress
```

### Robot Control Loop:
```
sparring_engine ──/robot/command──→ robot_node ──/robot/strike_command──→ V4 GUI ──→ Teensy ──→ Robot Arm
free_training    ──/robot/command──→ robot_node         ↑                                          │
                                                        └──/robot/strike_feedback──────────────────┘
```

### Defense Detection:
```
robot_node receives /robot/command → punch_processor opens 500ms defense window
  cv_node: /cv/user_tracking (slip/dodge detection)
  imu_node: /imu/arm_event (contact detection)
  Result: /punch/defense (block|slip|dodge|hit)
```

### GUI Communication:
```
Phone ──HTTP/WS──→ FastAPI ──/tmp JSON──→ GUI Command Poller (100ms)
                      │                         │
                      └──/llm/generate──→ llm_node (ROS service)
                      │
                      └──SQLite──→ Main DB + User DBs

GUI ──ROS bridge──→ All ROS nodes (pub/sub + services)
GUI ──SQLite──→ Main DB (auth, presets)
```

---

## 9. Message Type Fields

### ConfirmedPunch
```
timestamp, punch_type, pad, level, force_normalized, cv_confidence, imu_confirmed, cv_confirmed, accel_magnitude
```

### SessionState
```
state ("idle"|"countdown"|"active"|"rest"|"complete"), mode ("training"|"sparring"|"drill"|"free"|...), username
```

### PunchDetection
```
timestamp, punch_type, confidence, raw_class, consecutive_frames
```

### PunchEvent
```
timestamp, pad, level, force_normalized, accel_magnitude
```

### DefenseEvent
```
timestamp, arm, robot_punch_code, struck (bool), defense_type ("block"|"slip"|"dodge"|"hit"|"none")
```

### RobotCommand
```
command_type, punch_code, speed, source
```

### CoachTip
```
timestamp, tip_text, tip_type ("technique"|"encouragement"|"correction"|"suggestion"), trigger, priority
```

### DrillProgress
```
timestamp, combos_completed, combos_remaining, overall_accuracy, current_streak, best_streak
```

### UserTracking
```
timestamp, bbox_centre_x/y, bbox_top_y, bbox_width/height, depth, lateral_displacement, depth_displacement, user_detected
```

### SessionPunchSummary
```
20 fields: punch distributions, stats, movement, timing, fatigue index, defense breakdown
```

