# BoxBunny

**AI-powered boxing training robot with real-time punch detection, coaching, and performance analytics.**

BoxBunny is a production boxing training system built on the Jetson Orin NX platform. It combines computer vision, IMU sensors, a robot arm, and a local AI coach to deliver personalized boxing training sessions — no internet required.

---

## Features

### Training Modes
- **Combo Drills** — 50 progressive combos (Beginner → Advanced) with real-time accuracy tracking and mastery scoring
- **Sparring** — 5 boxing styles (Boxer, Brawler, Counter-Puncher, Pressure, Switch) with Markov chain attack sequences, reactive counter-punches (30/50/80% by difficulty), and defense tracking
- **Free Training** — Dynamic sparring where the robot only counter-punches when you hit a pad — user controls the pace
- **Performance Tests** — Power (IMU force), Stamina (timed endurance), Reaction Time (pose estimation)

### AI Coach
- Local LLM (Qwen2.5-3B-Instruct on Jetson GPU) — no cloud APIs, no internet needed
- Real-time coaching tips during sessions (every 18s, context-aware)
- Post-session AI analysis with personalized drill suggestions
- Chat interface on phone dashboard for boxing Q&A
- 54 fallback tips (technique, encouragement, correction, suggestion) when LLM is unavailable
- Standalone LLM Chat GUI for testing (`tools/llm_chat_gui.py`)

### Punch Detection & Fusion
- **CV Model**: FusionVoxelPoseTransformerModel — 8-class punch detection (jab, cross, left_hook, right_hook, left_uppercut, right_uppercut, block, idle) from depth voxels + pose features
- **IMU Sensors**: 4 pad IMUs (force classification: light/medium/hard) + 2 arm IMUs (defense tracking)
- **Fusion Engine**: CV predictions buffered over 0.8s window; when IMU pad strike fires, searches buffer for the most frequent valid prediction per pad constraint. Filters false positives using pad-punch rules (centre=jab/cross, left=left_hook/left_uppercut, right=right_hook/right_uppercut, head=any)
- **Data Collection**: Per-punch fusion records + grouped CV prediction events (>50% conf) + raw IMU strikes + person direction timeline + experimental defense reaction timing
- **Person Tracking**: YOLO bounding box centre drives left/right/centre direction with 20px hysteresis for robot yaw motor tracking

### Mobile Dashboard
- Vue 3 + Tailwind CSS SPA served over local network or public tunnel (localhost.run)
- Scan QR code from notebook to connect from any phone
- **Authentication**: Login with password or Android-style pattern lock
- **Onboarding**: 6-question proficiency survey to determine skill level on signup
- **Profile**: Selectable avatars (8 themed icons), display name, stats
- **Dashboard**: XP progress, weekly goals, training heatmap, peer comparison percentiles, AI coach tips
- **Training history** with session cards, grades, and trend charts
- **Gamification**: XP system, 6 ranks (Novice → Elite), streaks, 12 achievements with SVG badges
- **Population benchmarks**: percentile rankings by age/gender/level from sports science data
- **AI Chat**: Full conversation interface with the boxing coach
- **Coach mode**: Station management for group training sessions
- **Presets**: Save and load custom training configurations
- **Data export**: CSV export with date range filtering
- 7 API modules (auth, sessions, gamification, chat, coach, presets, export)
- WebSocket for real-time session updates

### Robot GUI (Touchscreen)
- PySide6 dark theme, 1024x600, 60px touch targets for gloved hands
- 24 pages, 11 reusable widgets
- IMU pad navigation (left=prev, right=next, centre=enter, head=back)
- Pattern lock authentication (no typing with gloves)
- Developer mode overlay showing live pad/arm activity and CV predictions
- Optional gesture control (MediaPipe Hands — open palm, thumbs up, peace sign, swipe)

---

## Hardware

| Component | Details |
|-----------|---------|
| **Compute** | Jetson Orin NX 16GB, Ubuntu 22.04, ROS 2 Humble |
| **Display** | 10.1" 1024x600 touchscreen |
| **Camera** | Intel RealSense D435i (RGB 960x540, Depth 848x480 @ 30fps) |
| **IMU Sensors** | 6x MPU6050 via Teensy 4.1 (4 pads + 2 arms) |
| **Robot Arm** | 4-motor arm via Teensy serial (6 punch types, IK-safe Bezier paths) |
| **Height Motor** | Auto-adjusts to user height via YOLO pose detection; manual UP/DOWN from GUI or phone |
| **Yaw Motor** | Tracks user position (left/right/centre) via CV person direction |

---

## Quick Start

### 1. Bootstrap (first time)
```bash
cd boxing_robot_ws
bash scripts/setup.sh
```

### 2. Build & seed demo data
```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
python3 tools/demo_data_seeder.py
```

### 3. Run (development mode with Teensy simulator)
```bash
ros2 launch boxbunny_core boxbunny_dev.launch.py
```

### 4. Run (full production with hardware)
```bash
ros2 launch boxbunny_core boxbunny_full.launch.py
```

### 5. Phone Dashboard
```bash
# From the notebook (includes public tunnel + QR code):
jupyter notebook notebooks/boxbunny_runner.ipynb
# → Run the "Phone Dashboard" cell

# Or manually:
python3 -m boxbunny_dashboard.server
# Open http://<jetson-ip>:8080 on your phone
```

### Demo Users
| User | Password | Level | Profile | Sessions | Rank |
|------|----------|-------|---------|----------|------|
| alex | boxing123 | Beginner | M, 22, 175cm, 72kg | 8 | Novice |
| maria | boxing123 | Intermediate | F, 28, 165cm, 58kg | 35 | Fighter |
| jake | boxing123 | Advanced | M, 31, 183cm, 85kg | 120 | Champion |
| sarah | coaching123 | Coach | F, 35 | 3 coaching | — |

---

## Architecture

```
boxing_robot_ws/
├── action_prediction/          # Trained CV model (DO NOT MODIFY)
│   ├── lib/                    # Inference engine (fusion_model, pose, voxel)
│   ├── model/                  # Weights (.pth, .onnx, .trt, YOLO)
│   └── run.py                  # Standalone inference script
│
├── src/
│   ├── boxbunny_msgs/          # 21 messages + 6 services
│   ├── boxbunny_core/          # 10 ROS 2 nodes + 4 launch files
│   ├── boxbunny_gui/           # PySide6 GUI (24 pages, 11 widgets)
│   │   └── assets/sounds/      # 18 sound effects
│   └── boxbunny_dashboard/     # FastAPI + Vue 3 SPA
│       ├── boxbunny_dashboard/ # Python backend (7 API modules)
│       ├── frontend/           # Vue 3 source (10 views, 8 components)
│       └── static/dist/        # Built SPA
│
├── config/                     # All configuration (no magic numbers in code)
│   ├── boxbunny.yaml           # Master system config
│   ├── ros_topics.yaml         # All ROS topic & service names
│   ├── drills.yaml             # 50 combo drill definitions
│   ├── sparring.yaml           # 5 sparring style transition matrices
│   ├── fallback_tips.json      # 54 AI coach fallback tips
│   ├── llm_system_prompt.txt   # LLM persona configuration
│   └── llm_models.yaml         # Available LLM models
│
├── data/
│   ├── boxbunny_main.db        # Main SQLite database (users, sessions, auth)
│   ├── users/                  # Per-user SQLite databases (XP, streaks, records)
│   ├── benchmarks/             # Population performance norms (age/gender)
│   ├── boxing_knowledge/       # 9 RAG documents for AI coach context
│   ├── punch_sequences/        # 8 robot punch waypoint files (JSON)
│   └── schema/                 # SQLite schema files (main + user)
│
├── models/
│   └── llm/                    # Qwen2.5-3B-Instruct Q4_K_M (2GB GGUF)
│
├── tools/
│   ├── teensy_simulator.py      # Teensy hardware simulator GUI (pads, arms, height, tracking)
│   ├── llm_chat_gui.py         # Standalone LLM chat interface (PySide6)
│   └── demo_data_seeder.py     # Create demo users with training history
│
├── notebooks/
│   ├── boxbunny_runner.ipynb   # Master notebook (12 sections)
│   └── scripts/                # Extracted notebook helper scripts
│
├── tests/                      # pytest suite (171 tests, 4 files)
├── scripts/                    # setup.sh, download_models.sh
└── _archive/                   # Archived old code (not used at runtime)
```

### ROS 2 Nodes

| Node | Package | Role |
|------|---------|------|
| `cv_node` | boxbunny_core | Wraps action prediction model, publishes punch detections + user tracking |
| `imu_node` | boxbunny_core | Processes Teensy IMU data, dual mode (navigation vs training) |
| `robot_node` | boxbunny_core | Bridge to V4 GUI: arm commands, height motor, yaw motor |
| `punch_processor` | boxbunny_core | CV+IMU fusion, defense event pipeline, slip detection |
| `session_manager` | boxbunny_core | Session lifecycle, round management, comprehensive data collection (fusion + CV + IMU + tracking) |
| `drill_manager` | boxbunny_core | Combo drill validation, mastery scoring, progression |
| `sparring_engine` | boxbunny_core | Markov chain attacks + reactive counter-punches, 5 styles, free mode |
| `analytics_node` | boxbunny_core | Statistics computation, trend analysis |
| `llm_node` | boxbunny_core | Local LLM coaching tips + post-session analysis |
| `gesture_node` | boxbunny_core | MediaPipe hand gesture navigation (optional) |

### Launch Files

| Launch File | Description |
|-------------|-------------|
| `boxbunny_dev.launch.py` | All nodes + Teensy simulator + GUI (development) |
| `boxbunny_full.launch.py` | All nodes with real hardware (production) |
| `headless.launch.py` | Processing nodes only, no GUI |
| `teensy_simulator.launch.py` | Teensy simulator standalone |

### Data Flow
```
Camera → cv_node → PunchDetection ─┐
                 → UserTracking     ├→ punch_processor → ConfirmedPunch → session_manager → DB
                 → PersonDirection  │                  → DefenseEvent
Teensy → imu_node → PunchEvent ────┘
                  → NavCommand → gui_bridge → GUI

cv_node → PersonDirection → robot_node → /robot/yaw_cmd → Teensy (turning motor)
gui_bridge → HeightCommand → robot_node → /robot/height_cmd → Teensy (height motor)

session_manager → SessionState → imu_node (mode switch)
                               → cv_node, drill_manager, sparring_engine
               ← PunchDetection (CV event grouping)
               ← PunchEvent (raw IMU strikes)
               ← PersonDirection (direction timeline)
               ← RobotCommand (defense reaction timing)

sparring_engine → RobotCommand (timer attacks + reactive counters)
               ← PunchEvent (free mode pad reactions)

llm_node ← GenerateLlm service ← gui_bridge / dashboard chat API
         → CoachTip → GUI overlay
```

### Dashboard API

| Module | Prefix | Endpoints |
|--------|--------|-----------|
| `auth` | `/api/auth` | Login, signup, pattern login, session, profile update, set pattern, logout |
| `sessions` | `/api/sessions` | Current session, history, detail, trends, raw data (CV/IMU/direction) |
| `gamification` | `/api/gamification` | XP/rank profile, achievements, leaderboard, benchmarks |
| `chat` | `/api/chat` | Send message, chat history |
| `coach` | `/api/coach` | Load config, start/end station, live participants |
| `presets` | `/api/presets` | CRUD for training presets, favorites |
| `remote` | `/api/remote` | GUI commands, presets, height control |
| `export` | `/api/export` | CSV export by session or date range |
| WebSocket | `/ws` | Real-time session state updates |

---

## Configuration

All system configuration lives in `config/`. No magic numbers in code.

| File | Purpose |
|------|---------|
| `boxbunny.yaml` | Thresholds, timeouts, model paths, network settings |
| `ros_topics.yaml` | Every ROS topic and service name (loaded by `constants.py`) |
| `drills.yaml` | 50 combo definitions with timing, sequence, and difficulty params |
| `sparring.yaml` | 5 style transition matrices + difficulty scaling curves |
| `fallback_tips.json` | 54 coaching tips (4 categories) used when LLM is unavailable |
| `llm_system_prompt.txt` | LLM persona and behavior instructions |
| `llm_models.yaml` | Available LLM model paths and configurations |

**To rename a ROS topic**: edit `config/ros_topics.yaml` → restart. All nodes load from there via `constants.py`.

---

## Notebook

The master notebook (`notebooks/boxbunny_runner.ipynb`) provides all essential operations:

| # | Section | Description |
|---|---------|-------------|
| 1 | Build & Setup | `colcon build` + seed demo data |
| 2 | Run Tests | Full pytest suite (171 tests) |
| 3 | System Check | Hardware, dependencies, and model status |
| 4 | Launch System | Start all ROS nodes + Teensy simulator + GUI |
| 5 | Stop System | Kill all BoxBunny processes |
| 6 | GUI Test | Launch touchscreen GUI for visual inspection |
| 7 | Phone Dashboard | Server + public tunnel + QR code for phone access |
| 8 | CV Model Live Test | Camera feed with pose skeleton + action labels |
| 9 | LLM Coach Test | Interactive AI coach chat GUI |
| 10 | Build Vue Frontend | Rebuild phone dashboard SPA after changes |
| 11 | Sound Test | Play all 18 sound effects |
| 12 | Demo Profiles & Benchmarks | User profile cards + percentile rankings |

---

## Testing

```bash
# Run all 171 tests (no hardware needed)
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 -m pytest tests/ -v

# Or use the notebook:
jupyter notebook notebooks/boxbunny_runner.ipynb
# → Run cell 2 (Run Tests)
```

---

## Dependencies

| Category | Packages |
|----------|----------|
| **Platform** | Python 3.10, ROS 2 Humble, CUDA (Jetson), Ubuntu 22.04 |
| **CV / ML** | PyTorch, YOLO (ultralytics), ONNX Runtime, TensorRT |
| **GUI** | PySide6 |
| **Dashboard backend** | FastAPI, uvicorn, bcrypt, PyJWT |
| **Dashboard frontend** | Vue 3, Pinia, Vue Router, Tailwind CSS, Chart.js |
| **LLM** | llama-cpp-python (with CUDA), Qwen2.5-3B-Instruct GGUF |
| **Sensors** | pyrealsense2 (D435i), pyserial (Teensy) |
| **Optional** | MediaPipe (gesture control), qrcode (QR generation), pyngrok (tunneling) |

```bash
pip install -r requirements.txt
pip install -r requirements-jetson.txt  # Jetson-specific wheels
bash scripts/download_models.sh         # Download LLM model (2GB)
```

---

## Documentation

Detailed technical documentation is in the `docs/` folder:

| Document | Contents |
|----------|----------|
| [architecture.md](docs/architecture.md) | System overview, node architecture, ROS topic graph, data flows, design philosophy |
| [cv_pipeline.md](docs/cv_pipeline.md) | CV model, inference pipeline, CV+IMU fusion algorithm, person tracking |
| [teensy_simulator.md](docs/teensy_simulator.md) | Simulator GUI, execute/auto mode, hardware forwarding, combo system |
| [data_collection.md](docs/data_collection.md) | Data sources, collection strategy, session summary, database schema |
| [dashboard_and_llm.md](docs/dashboard_and_llm.md) | Phone dashboard, API endpoints, AI coach, LLM reliability, height control |
| [sparring_and_training.md](docs/sparring_and_training.md) | Training modes, sparring engine, free mode, robot arm control, person tracking |
| [gui_and_user_experience.md](docs/gui_and_user_experience.md) | Touchscreen GUI pages, UX design (gloved hands, pattern lock, IMU nav), phone dashboard views, calibration workflow |
| [testing_and_notebook.md](docs/testing_and_notebook.md) | Notebook sections, pytest suite (171 tests), integration tests, manual testing guide, demo data |

---

## Developer Guidelines

See `CLAUDE.md` for full development rules. Key points:

- **Never delete files** — archive to `_archive/`
- **Never modify** `action_prediction/lib/` model files
- All configurable values in YAML configs — no magic numbers
- All ROS topic names in `config/ros_topics.yaml`
- Use `logging` module, not `print()`
- Max ~300 lines per file, type hints on all function signatures
- Specific exception handling, structured logging

---

## Sound Assets

18 sound effects in `src/boxbunny_gui/assets/sounds/`:

| Sound | Source | License |
|-------|--------|---------|
| `bell_start.wav` | [Envato Elements](https://elements.envato.com/bell-QVQMGF2) (3x rapid dings) | Envato Elements |
| `bell_end.wav` | Same source (single ding) | Envato Elements |
| `impact.wav` | [Mixkit #2149](https://mixkit.co/free-sound-effects/punch/) | Mixkit License (free) |
| `hit_confirm.wav` | [Mixkit #2150](https://mixkit.co/free-sound-effects/punch/) | Mixkit License (free) |
| `miss.wav` | [Mixkit #1491](https://mixkit.co/free-sound-effects/whoosh/) | Mixkit License (free) |
| `combo_complete.wav` | [Mixkit #2018](https://mixkit.co/free-sound-effects/) | Mixkit License (free) |
| `session_complete.wav` | [Mixkit #2000](https://mixkit.co/free-sound-effects/) | Mixkit License (free) |
| `countdown_beep.wav` | [Mixkit #916](https://mixkit.co/free-sound-effects/countdown/) | Mixkit License (free) |
| `countdown_go.wav` | [Mixkit #2575](https://mixkit.co/free-sound-effects/) | Mixkit License (free) |
| `countdown_tick.wav` | Trimmed from countdown_beep | Mixkit License (free) |
| `rest_start.wav` | [Mixkit #933](https://mixkit.co/free-sound-effects/bell/) | Mixkit License (free) |
| `stimulus.wav` | [Mixkit #586](https://mixkit.co/free-sound-effects/bell/) | Mixkit License (free) |
| `reaction_stimulus.wav` | Copy of stimulus | Mixkit License (free) |
| `coach_notification.wav` | [Mixkit #2869](https://mixkit.co/free-sound-effects/) | Mixkit License (free) |
| `error.wav` | [Mixkit #950](https://mixkit.co/free-sound-effects/) | Mixkit License (free) |
| `button_click.wav` | [Mixkit #2568](https://mixkit.co/free-sound-effects/) | Mixkit License (free) |
| `btn_press.wav` | Trimmed from button_click | Mixkit License (free) |
| `nav_tick.wav` | Trimmed from button_click | Mixkit License (free) |

## Visual Assets

**Dashboard Avatars** — 8 SVG icons in `src/boxbunny_dashboard/frontend/public/avatars/`:
boxer, tiger, eagle, wolf, flame, lightning, shield, crown

**Achievement Badges** — 12 SVG icons in `src/boxbunny_dashboard/frontend/public/achievements/`:
first_blood, century, fury, thousand_fists, speed_demon, weekly_warrior, consistent, iron_chin, marathon, centurion, well_rounded, perfect_round

**Rank Badges** — 6 SVG icons in `src/boxbunny_dashboard/frontend/public/ranks/`:
novice, contender, fighter, warrior, champion, elite

---

## License

Proprietary. All rights reserved.
