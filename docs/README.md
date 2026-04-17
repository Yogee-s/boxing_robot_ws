# BoxBunny Documentation

Everything you need to pick this project up from scratch. Read this file first — it tells you where to go for what.

---

## How this documentation is organised

```
docs/
├── README.md                     <- YOU ARE HERE (reading guide + index)
├── system/                       <- architecture, ROS graph, integration, deep-dive
├── hardware/                     <- Arduino base, Teensy arms, BLE bridge
├── gui/                          <- PySide6 desktop app (touchscreen)
├── dashboard/                    <- FastAPI + Vue 3 phone web app
├── training/                     <- training modes (sparring, free, drills, tests)
├── data/                         <- SQLite schema, session storage
├── node_graphs/                  <- ROS topic/service quick-reference cards
├── deployment.md                 <- build, launch, hardware setup
├── testing.md                    <- test suites + how to run them
├── new_changes.md                <- recent major changes (LLM upgrade, BLE, etc.)
└── _archive/                     <- old docs kept for reference, do not rely on
```

Rule: if you're unsure whether a doc is current, check the git log on that file (`git log -n 3 -- docs/...`). Anything in `_archive/` is out of date by design.

---

## Recommended reading order by role

### "I'm brand new — where do I start?"

1. Root [README.md](../README.md) — one-screen elevator pitch + quick-start commands.
2. [system/architecture.md](system/architecture.md) — the big picture: ROS nodes, topics, data flow.
3. [node_graphs/SYSTEM_ARCHITECTURE_REFERENCE.md](node_graphs/SYSTEM_ARCHITECTURE_REFERENCE.md) — every node on one map.
4. [deployment.md](deployment.md) — how to actually get it running on real hardware.
5. Poke around [gui/architecture.md](gui/architecture.md) or [dashboard/backend.md](dashboard/backend.md) depending on which surface you'll work on.

### "I need to touch the robot hardware"

1. [hardware/boxing_arm_control.md](hardware/boxing_arm_control.md) — Teensy + Damiao motors + V4 arm-control GUI.
2. [hardware/ble_bridge.md](hardware/ble_bridge.md) — base-rotation Arduino + height motor + BLE protocol + Jetson bridge.
3. `Boxing_Arm_Control/README.md` — the submodule's own readme (kinematics, strike library, calibration flow).
4. `Boxing_Arm_Control/ros2_ws/ble_control/README.md` — Arduino firmware specifics.

### "I need to change a training mode or add a new one"

1. [training/modes.md](training/modes.md) — every mode explained with its backing node(s).
2. [gui/architecture.md](gui/architecture.md) — page router, bridge signals, how a page is wired.
3. [node_graphs/SESSION_STATE_MACHINE_REFERENCE.md](node_graphs/SESSION_STATE_MACHINE_REFERENCE.md) — idle → countdown → active → rest → complete.
4. Code: `src/boxbunny_core/boxbunny_core/session_manager.py`, `sparring_engine.py`, `free_training_engine.py`, `drill_manager.py`.

### "I need to debug CV / IMU detection"

1. [node_graphs/CV_IMU_FUSION_PIPELINE_REFERENCE.md](node_graphs/CV_IMU_FUSION_PIPELINE_REFERENCE.md) — exact fusion path.
2. [system/technical-deep-dive.md](system/technical-deep-dive.md) — camera FPS, voxel features, model architecture.
3. Code: `notebooks/scripts/run_with_ros.py` (CV model runner), `src/boxbunny_core/boxbunny_core/imu_node.py`, `punch_processor.py`.

### "I'm the hand-off developer — what changed recently?"

1. [new_changes.md](new_changes.md) — curated log of the bigger changes (LLM model upgrade, BLE bridge, base tracking, sparring counter fix, etc.). Read top-to-bottom.
2. `git log --oneline -30` — the last month of commits.
3. `CLAUDE.md` in the repo root — non-negotiable project rules (no deletes, YAML configs, no magic numbers).

---

## Quick index (every doc, one line each)

### System
- [system/architecture.md](system/architecture.md) — ROS nodes, message types, topics, services, launch configs.
- [system/integration.md](system/integration.md) — GUI ↔ Dashboard ↔ Core communication, remote commands, auth flow.
- [system/technical-deep-dive.md](system/technical-deep-dive.md) — hardware specs, CV FPS analysis, LLM pipeline, GPU sharing.

### Hardware
- [hardware/boxing_arm_control.md](hardware/boxing_arm_control.md) — arm hardware (Teensy + Damiao motors + V4 GUI).
- [hardware/ble_bridge.md](hardware/ble_bridge.md) — base Arduino (rotation + height), BLE protocol, Jetson bridge node.

### GUI (touchscreen desktop app)
- [gui/architecture.md](gui/architecture.md) — PySide6 app structure, page system, ROS bridge, navigation, sound.
- [gui/design-system.md](gui/design-system.md) — colours, typography, spacing, button/card styles, touch-target sizes.

### Dashboard (phone web app)
- [dashboard/backend.md](dashboard/backend.md) — FastAPI server, auth, REST endpoints, WebSocket, DB access.
- [dashboard/frontend.md](dashboard/frontend.md) — Vue 3 SPA, router, Pinia stores, real-time updates.

### Training
- [training/modes.md](training/modes.md) — combo drills, sparring AI, free training, performance tests, coach station.

### Data
- [data/schema.md](data/schema.md) — SQLite schemas (main + per-user), session summaries, gamification, benchmarks.

### ROS graph quick-reference cards
- [node_graphs/SYSTEM_ARCHITECTURE_REFERENCE.md](node_graphs/SYSTEM_ARCHITECTURE_REFERENCE.md) — every node + what it publishes/subscribes.
- [node_graphs/GUI_DASHBOARD_COMMUNICATION_REFERENCE.md](node_graphs/GUI_DASHBOARD_COMMUNICATION_REFERENCE.md) — GUI ↔ dashboard message paths.
- [node_graphs/CV_IMU_FUSION_PIPELINE_REFERENCE.md](node_graphs/CV_IMU_FUSION_PIPELINE_REFERENCE.md) — detection → fusion → UI path.
- [node_graphs/SESSION_STATE_MACHINE_REFERENCE.md](node_graphs/SESSION_STATE_MACHINE_REFERENCE.md) — session lifecycle state transitions.

### Ops
- [testing.md](testing.md) — unit tests, integration tests, notebook-based smoke tests.
- [deployment.md](deployment.md) — hardware bring-up, build, launch configs, troubleshooting.

---

## Key files outside `docs/` you should know

| Purpose | File |
|---------|------|
| Master project rules | [../CLAUDE.md](../CLAUDE.md) |
| Master config | [../config/boxbunny.yaml](../config/boxbunny.yaml) |
| Every ROS topic/service name | [../config/ros_topics.yaml](../config/ros_topics.yaml) |
| Main SQLite DB | `data/boxbunny_main.db` |
| Per-user SQLite DB | `data/users/{username}/boxbunny.db` |
| DB schemas (source of truth) | `data/schema/main_schema.sql`, `data/schema/user_schema.sql` |
| Full-stack launch | [../src/boxbunny_core/launch/boxbunny_full.launch.py](../src/boxbunny_core/launch/boxbunny_full.launch.py) |
| Runner notebook (recommended entry point) | [../notebooks/boxbunny_runner.ipynb](../notebooks/boxbunny_runner.ipynb) |

---

## One-minute architecture primer

```
   Touchscreen (PySide6 GUI)           Phone browser (Vue 3 SPA)
                   │                              │
             GuiBridge (QThread)          FastAPI + WebSocket
                   │                              │
                   └──────── ROS 2 graph ─────────┘
                                  │
  ┌───────────────┬──────────────┼──────────────┬───────────────┐
  │               │              │              │               │
  cv_node     imu_node    session_manager   robot_node   ble_bridge_node
  (YOLO +     (pad +      (state machine)   (→ V4 GUI)   (→ Arduino base
  fusion      arm IMUs                                    via BLE:
  model)      via Teensy)                                 height + rotation)
```

- **cv_node** — runs the punch model + YOLO pose, publishes detections, pose, and primary-user tracking.
- **imu_node** — forwards Teensy pad impacts and arm strikes, switches between NAV / TRAINING modes based on session state.
- **session_manager** — the brain of the session: idle ↔ countdown ↔ active ↔ rest ↔ complete. Publishes `SessionState` that every other node listens to.
- **punch_processor** — fuses CV predictions + IMU impacts inside ±200 ms, emits `ConfirmedPunch`. Also detects blocks/slips via pose + IMU-miss.
- **sparring_engine** — runs in sparring mode; Markov-chain attack patterns + reactive counter-punches.
- **free_training_engine** — runs in free mode; robot only counter-punches when a pad is hit.
- **drill_manager** — progresses combo drills, scores per-combo accuracy.
- **llm_node** — on-device coaching LLM (Gemma-3n E2B via llama.cpp).
- **robot_node** — translates BoxBunny `RobotCommand` into strike commands for the V4 Arm Control GUI.
- **ble_bridge_node** — owns the single BLE link to the base-rotation Arduino; multiplexes height-motor commands and live person-tracking rotation commands.

---

## Conventions every contributor must follow

From [../CLAUDE.md](../CLAUDE.md):

1. **Never delete files** — archive to `_archive/`.
2. **Never touch files outside** `boxing_robot_ws/`.
3. **Do not modify** `action_prediction/lib/fusion_model.py`, `pose.py`, `voxel_features.py`, `voxel_model.py` — these are the model definition.
4. All configurable values live in YAML configs under [config/](../config/). No magic numbers in code.
5. All ROS topic/service names live in [config/ros_topics.yaml](../config/ros_topics.yaml) and are loaded by `constants.py`.
6. No `print()` — use the `logging` module.
7. Max ~300 lines per file, type hints on all function signatures.
8. Production code requires: docstrings, specific exception handling, structured logging.

---

## If you only have 10 minutes

Read, in order:

1. [../README.md](../README.md) (root) — project pitch + quick-start.
2. This file — where everything lives.
3. [node_graphs/SYSTEM_ARCHITECTURE_REFERENCE.md](node_graphs/SYSTEM_ARCHITECTURE_REFERENCE.md) — the ROS graph at a glance.

That's enough to know what questions to ask next.
