# BoxBunny

**AI-powered boxing training robot with real-time punch detection, coaching, and performance analytics.**

BoxBunny is a production boxing training system built on the Jetson Orin NX platform. It combines computer vision, IMU sensors, a robot arm, and a local AI coach to deliver personalized boxing training sessions — no internet required.

---

## Features

### Training Modes
- **Combo Drills** — 50 progressive combos (Beginner → Advanced) with real-time accuracy tracking
- **Sparring** — 5 boxing styles (Boxer, Brawler, Counter-Puncher, Pressure, Switch) with Markov chain attack sequences and defense tracking
- **Free Training** — Open-ended pad work with punch counting and force tracking
- **Performance Tests** — Power (IMU force), Stamina (timed endurance), Reaction Time (pose estimation)

### AI Coach
- Local LLM (Qwen2.5-3B on Jetson GPU) — no cloud APIs, no internet needed
- Real-time coaching tips during sessions
- Post-session AI analysis with personalized drill suggestions
- Chat interface on phone dashboard for boxing Q&A

### Punch Detection & Fusion
- **CV Model**: FusionVoxelPoseTransformerModel — 8-class punch detection (jab, cross, hooks, uppercuts, block, idle) from depth voxels + pose features
- **IMU Sensors**: 4 pad IMUs (force classification) + 2 arm IMUs (defense tracking)
- **Fusion Engine**: CV + IMU correlation within ±200ms window with pad-location constraints to filter false positives
- **Derived Slip Detection**: arm miss + no CV block + depth/lateral displacement = slip

### Mobile Dashboard
- Vue 3 SPA served over WiFi AP — scan QR code to connect
- Training history with charts and trends
- Gamification: XP, ranks (Novice → Elite), streaks, achievements, session scores
- Population benchmarks: percentile rankings by age/gender from sports science data
- Coach mode: station management for group training sessions
- PDF/CSV data export

### Robot GUI (Touchscreen)
- PySide6 dark theme, 1024x600, 60px touch targets for gloved hands
- IMU pad navigation (left=prev, right=next, centre=enter, head=back)
- Pattern lock authentication (no typing with gloves)
- Developer mode overlay showing live pad/arm activity and CV predictions
- Optional gesture control (MediaPipe Hands)

---

## Hardware

| Component | Details |
|-----------|---------|
| **Compute** | Jetson Orin NX 16GB, Ubuntu 22.04, ROS 2 Humble |
| **Display** | 10.1" 1024x600 touchscreen |
| **Camera** | Intel RealSense D435i (RGB 960x540, Depth 848x480 @ 30fps) |
| **IMU Sensors** | 6x MPU6050 via Teensy (4 pads + 2 arms) |
| **Robot Arm** | 4-motor arm via Teensy serial (8 punch sequences) |
| **Height Motor** | Auto-adjusts to user height via YOLO pose detection |

---

## Quick Start

### 1. Bootstrap (first time)
```bash
cd boxing_robot_ws
bash scripts/setup.sh
```

### 2. Build
```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 3. Run (development mode with IMU simulator)
```bash
ros2 launch boxbunny_core boxbunny_dev.launch.py
```

### 4. Run (full production)
```bash
ros2 launch boxbunny_core boxbunny_full.launch.py
```

### 5. Phone Dashboard
Open browser to `http://<jetson-ip>:8080` or scan the QR code on the robot screen.

### Demo Users
| User | Password | Level | Sessions | Rank |
|------|----------|-------|----------|------|
| alex | boxing123 | Beginner (M, 22) | 8 | Novice |
| maria | boxing123 | Intermediate (F, 28) | 35 | Fighter |
| jake | boxing123 | Advanced (M, 31) | 120 | Champion |
| sarah | coaching123 | Coach (F, 35) | 3 coaching | — |

---

## Architecture

```
boxing_robot_ws/
├── action_prediction/          # Trained CV model (preserved, do not modify)
│   ├── lib/                    # Inference engine (10 modules)
│   └── model/                  # Model weights (.pth, .onnx, .trt, YOLO)
│
├── src/
│   ├── boxbunny_msgs/          # 21 custom messages + 6 services
│   ├── boxbunny_core/          # 10 ROS 2 processing nodes
│   ├── boxbunny_gui/           # PySide6 touchscreen GUI (23 pages, 11 widgets)
│   └── boxbunny_dashboard/     # FastAPI server + Vue 3 SPA
│
├── config/                     # All configuration in one place
│   ├── boxbunny.yaml           # Master system config
│   ├── ros_topics.yaml         # All ROS topic & service names
│   ├── drills.yaml             # 50 combo drill definitions
│   ├── sparring.yaml           # 5 sparring style transition matrices
│   ├── fallback_tips.json      # 54 AI coach fallback tips
│   ├── llm_system_prompt.txt   # LLM persona configuration
│   └── llm_models.yaml         # Available LLM models
│
├── data/
│   ├── benchmarks/             # Population performance norms (age/gender)
│   ├── boxing_knowledge/       # 18 RAG documents for AI coach
│   ├── punch_sequences/        # 8 robot punch waypoint files
│   └── schema/                 # SQLite schema files
│
├── models/
│   └── llm/                    # Qwen2.5-3B-Instruct (2GB GGUF)
│
├── tools/                      # Development utilities
│   ├── imu_simulator.py        # 6-button IMU simulator GUI
│   ├── llm_chat_gui.py         # Standalone LLM chat interface
│   └── demo_data_seeder.py     # Create demo users with training history
│
├── notebooks/
│   └── boxbunny_runner.ipynb   # Master notebook (28 sections, 75 cells)
│
├── tests/                      # pytest suite (171 tests)
├── scripts/                    # setup.sh, download_models.sh, asset generators
└── _archive/                   # Archived old code (gitignored)
```

### ROS 2 Nodes

| Node | Role |
|------|------|
| `cv_node` | Wraps action prediction model, publishes punch detections + user tracking |
| `imu_node` | Processes Teensy IMU data, dual mode (navigation vs training) |
| `robot_node` | Serial interface to robot arm + height motor |
| `punch_processor` | CV+IMU fusion, defense event pipeline, slip detection |
| `session_manager` | Session lifecycle, timers, data accumulation |
| `drill_manager` | Combo drill validation, mastery scoring |
| `sparring_engine` | Markov chain attack sequences, reactive behaviors |
| `analytics_node` | Statistics computation, trend analysis |
| `llm_node` | Local LLM coaching tips + analysis |
| `gesture_node` | MediaPipe hand gesture navigation (optional) |

### Data Flow
```
Camera → cv_node → PunchDetection ─┐
                                    ├→ punch_processor → ConfirmedPunch → session_manager → DB
Teensy → imu_node → PunchEvent ────┘                  → DefenseEvent
                  → NavCommand → gui_bridge → GUI

session_manager → SessionState → imu_node (mode switch)
                               → cv_node, drill_manager, sparring_engine
```

---

## Configuration

All system configuration lives in `config/`. No magic numbers in code.

| File | Purpose |
|------|---------|
| `boxbunny.yaml` | Thresholds, timeouts, model paths, network settings |
| `ros_topics.yaml` | Every ROS topic and service name (fully documented) |
| `drills.yaml` | 50 combo definitions with timing and difficulty params |
| `sparring.yaml` | 5 style transition matrices + difficulty scaling |
| `fallback_tips.json` | 54 coaching tips used when LLM is unavailable |

**To rename a ROS topic**: edit `config/ros_topics.yaml` → restart. All nodes load from there.

---

## Testing

```bash
# Run all 171 tests (no hardware needed)
python3 -m pytest tests/ -v

# Open the master notebook for interactive testing
jupyter notebook notebooks/boxbunny_runner.ipynb
```

The notebook includes cells for:
- Building, seeding data, launching subsystems
- Sound playback tests
- QR code generation for phone dashboard
- IMU simulator with ROS topic monitoring
- CV model live test with camera
- Developer mode overlay test
- Demo user dashboard previews
- Benchmark percentile calculations

---

## Dependencies

- **Python 3.10+**, **ROS 2 Humble**, **CUDA** (Jetson)
- **PyTorch** + **YOLO** (action prediction), **PySide6** (GUI), **FastAPI** (dashboard)
- **llama-cpp-python** with CUDA (local LLM), **MediaPipe** (gesture control)
- **Vue 3** + **Tailwind CSS** + **Chart.js** (phone dashboard frontend)

Install everything:
```bash
pip install -r requirements.txt
pip install -r requirements-jetson.txt  # Jetson-specific
bash scripts/download_models.sh         # Download LLM model (2GB)
```

---

## License

Proprietary. All rights reserved.
