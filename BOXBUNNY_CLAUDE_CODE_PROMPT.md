# BoxBunny — Full Build Prompt for Claude Code

## Context

BoxBunny is a boxing training robot product built on a **Jetson Orin NX 16GB** (Ubuntu 22.04) with a 10.1-inch 1024×600 touchscreen. It has a robot arm that throws punches at the user via padded targets. An Intel RealSense D435i depth camera (RGB 960×540 @ 30fps, Depth 848×480 @ 30fps) runs a trained CV model that detects punches the user throws. The system is currently a working PySide6 prototype with combo drills, sparring, performance tests, user management, and basic AI coaching (currently via Anthropic API — to be replaced with a local LLM, as everything must run locally with no internet dependency).

**This is a ground-up rebuild, not a refactor.** Build everything from scratch with production quality. The old prototype code should be archived to `_archive/` (never deleted), but the new codebase is written fresh. Only the trained CV model weights and inference library (`action_prediction/`) are preserved and wrapped in a ROS 2 node. The system uses **ROS 2 Humble** as the communication backbone between all hardware components and software modules. The current prototype uses PySide6 (Python/Qt), but the rebuild should use whatever framework is best for this product. Evaluate the tradeoffs and recommend the best stack.

**Workspace path**: `/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/`

**Developer priorities**: Clean project structure, modularity, testability. Ability to test individual subsystems independently. Comprehensive test/demo infrastructure (simulators, demo data seeders, smoke tests, hardware check tools). Production-quality code for market entry.

**Hardware:**
- **Jetson Orin NX 16GB**, Ubuntu 22.04, ROS 2 Humble
- 10.1-inch 1024×600 touchscreen
- Intel RealSense D435i depth camera (RGB 960×540 @ 30fps, Depth 848×480 @ 30fps) running a custom trained CV model that outputs detected punch classes: **jab, cross, left_hook, right_hook, left_uppercut, right_uppercut, block, idle**. Note: the model does NOT detect slips — slip detection is derived from depth data + bounding box displacement + arm IMU miss (see Section 4).
- YOLO26n-pose model for pose estimation (reaction time measurement) — same YOLO version used for the CV punch detection model
- Robot arm controlled via serial (Teensy), already has a working stub interface
- **Height adjustment motor** — controlled via Teensy, adjusts the vertical position of the robot/pads to match the user's height. Auto-calibrated using the YOLO pose model bounding box (see Section 4).
- **6 × IMU sensors** connected via a Teensy to the Jetson. The Teensy publishes data over ROS topics. Layout:
  
  **4 pad IMUs (on padded targets the user punches):**
  - **Left pad** — IMU on the left torso target
  - **Centre pad** — IMU on the centre torso target
  - **Right pad** — IMU on the right torso target
  - **Head pad** — IMU on the head target
  
  **2 arm IMUs (on the robot's striking arms):**
  - **Left arm** — IMU on the robot's left striking arm
  - **Right arm** — IMU on the robot's right striking arm
  - These detect whether the robot's punch successfully struck the user — tracked in ALL training modes, not just sparring

- **IMU data format from Teensy:**
  - **Pad IMUs**: The Teensy processes raw accelerometer data on-board and sends a classified impact level: `light`, `medium`, or `hard` (not raw values). This simplifies the data pipeline — the Jetson receives clean categorized impacts, not noisy raw acceleration data.
  - **Arm IMUs**: Binary — the Teensy sends `struck` or `miss` for each robot arm movement. Did the arm make contact with the user or not?

- The pad IMUs serve **dual purpose**:
  1. **Punch detection** — detect impacts on each pad (force level, location, timing) for training data
  2. **GUI navigation** — Left/Right pads = navigate between options, Centre pad = select/enter, Head pad = go back. This lets users navigate with gloves on by tapping the pads. Centre pad tap can also start a session (triggers the 3-second countdown).
  
- The arm IMUs **continuously collect data in all training modes** (combo drills, sparring, free training — not just sparring):
  - Every time the robot throws a punch, the arm IMU records whether it made contact with the user
  - In combo drills: tracks if the robot's demonstration punches are making unintended contact
  - In sparring: tracks the user's defensive performance (blocked/dodged vs hit)
  - In free training: tracks any contact events
  - All arm IMU data is saved to the session record and shown in the post-session analytics on the phone dashboard

**IMU Simulator (for development without hardware):**
- Since the physical IMU Teensy is not yet available, build a small separate GUI window that simulates all 6 IMUs
- The simulator shows 6 buttons arranged to match the physical layout:
  - 4 pad buttons: left, centre, right in a row, head above centre (same as physical pad layout)
  - 2 arm buttons: positioned on the sides to represent the robot's left and right striking arms
- Clicking a pad button publishes the same ROS topic message that the real Teensy would publish, with a selectable impact level (light/medium/hard)
- Clicking an arm button publishes a `struck` event (simulating the robot's arm hitting the user)
- This allows full development and testing of all IMU-dependent features without the hardware
- The simulator should be a standalone launchable tool that publishes to the same ROS topics

**Key design philosophy:** This is a product going to market. It needs to serve two distinct user types — individuals training solo and coaches integrating the robot into group/circuit training sessions. Every feature should be polished, intuitive, and feel professional.

---

## Existing Codebase Summary

The current codebase is a PySide6 app (`main_gui.py`) using a QStackedWidget with 44 pages. Key existing modules:

```
GUI/
├── main_gui.py                    # Monolithic — all 44 page classes + MainWindow
├── core/                          # config.py (AppState, TrainingConfig), constants.py (PageIndex, ButtonStyle), navigation.py, tooltip.py
├── sparring/                      # spar_pages.py (8 pages), combo_pools.py (Markov matrices), sequence_generator.py, robot_interface.py, sparring_database.py
├── proficiency/                   # Proficiency assessment after signup (now served via phone dashboard, not robot GUI)
├── combo_curriculum/              # SQLite-backed combo progression system (50 combos across 3 levels)
├── power/                         # power_runner.py — punch force measurement via IMU
├── stamina/                       # stamina_runner.py — timed punch endurance
├── reaction_time/                 # reaction_time_runner.py — YOLO pose + camera
├── utils/                         # user_management.py (CSV-based user CRUD)
├── setup/                         # DB schema creation and combo population scripts
├── data/combos.db                 # Shared combo template DB
├── users/<username>/              # Per-user dirs: combos.db, performance_history.db
└── training_history/              # CSV session logs
```

**What works well (preserve the concepts, reimplement in chosen framework):**
- Stacked page navigation model
- Sparring Markov chain sequence generation with boxing style matrices and weakness bias
- Combo curriculum progression system (50 combos, mastery scoring)
- Performance test flows (stamina, reaction time — power test will need IMU)
- Per-user SQLite databases for history
- Robot interface (set_speed, send_punch, send_round_start, send_round_stop) — to be wrapped in a ROS robot_node
- CV punch detection model and pose estimation model (inference logic, to be wrapped in ROS cv_node)

**What needs to change:**
- `main_gui.py` is monolithic — break into proper module structure
- AI coaching currently uses Anthropic API — switch to local LLM hosted on the Jetson
- UI needs visual polish, sound effects, and better UX for a consumer product
- Need to add phone-based auth, preset system, coach mode, QR-based mobile dashboard, gesture control
- File-based CV pipeline (trigger JSON → poll for output) must be replaced with real-time ROS topic architecture
- Need CV + IMU fusion pipeline for confirmed punch detection with force data
- User management should move from CSV to SQLite
- All inter-component communication must go through ROS 2 topics
- Evaluate whether PySide6/Python is still the best framework for the robot GUI or if something else would be better for this product

### Trained CV Model — action_prediction/ (KEEP)

This is the trained action prediction model. **Do not modify the model architecture or retrain.** Wrap it in a ROS 2 node (`cv_node`).

**Architecture**: `FusionVoxelPoseTransformerModel` (PyTorch)
- **Input**: Sequence of 12 frames, concatenated voxel features (3,456 dims) + pose features (42 dims) = 3,498 dims total
- **Voxel Branch**: `Conv3DStem` (3D convolutions), 2-channel 12×12×12 voxel grids → 192-dim embedding
  - Channel 0: fast delta (4 frames, ~67ms)
  - Channel 1: slow delta (8 frames, ~267ms)
- **Pose Branch**: `PoseEncoder` MLP, 42-dim → 64-dim with confidence gating
- **Fusion Layer**: Concat voxel(192) + pose(64) → 256-dim → Linear → 192-dim + LayerNorm
- **Temporal**: Sinusoidal PositionalEncoding, TransformerEncoder (4 layers, 8 heads, d_model=192, FFN=576, causal mask)
- **Classification Head**: Mean+Max pooling → 384-dim → LayerNorm → Linear(384→96) → GELU → Linear(96→8)
- **Output**: 8 action classes: jab, cross, left_hook, right_hook, left_uppercut, right_uppercut, block, idle
- **Inference**: Uses YOLO26n-pose for skeleton detection on RGB, D435i depth stream for voxel construction

**Existing library structure** (preserve this, wrap in ROS node):
```
action_prediction/
├── lib/
│   ├── __init__.py
│   ├── inference_runtime.py    # Production inference engine — THIS IS THE MAIN ENTRY POINT
│   ├── feature_buffer.py       # Feature extraction buffer (voxel + pose)
│   ├── state_machine.py        # Action state machine (idle↔active transitions)
│   ├── checkpoint_utils.py     # Model loading utilities
│   └── prediction_utils.py     # Prediction post-processing & smoothing
├── model/
│   ├── best_model.pth          # FusionVoxelPoseTransformer trained weights
│   └── yolo26n-pose.pt         # YOLO pose model weights
└── run.py                      # Standalone inference launcher (for testing without ROS)
```

The `cv_node` ROS node should wrap `inference_runtime.py` — initialise the model, feed it camera frames, and publish detections to `/boxbunny/cv/detection`. The existing inference pipeline handles frame buffering, voxel construction, pose extraction, and temporal smoothing internally.

### Old Prototype Structure (for reference — archive, don't delete)

The old workspace at `/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/` contains:

---

## Architecture Requirements

### Module Structure

This is a ROS 2 Humble workspace. All code lives in `src/` as proper ROS 2 packages, plus the preserved `action_prediction/` library and development tools.

```
boxing_robot_ws/
├── action_prediction/              # PRESERVED — trained CV model (DO NOT MODIFY)
│   ├── lib/                        # Inference library
│   ├── model/                      # Model weights (best_model.pth, yolo26n-pose.pt)
│   └── run.py                      # Standalone test launcher
│
├── src/                            # ROS 2 packages
│   │
│   ├── boxbunny_msgs/              # Custom message & service definitions
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── msg/                    # All .msg files (see ROS Topics section)
│   │   └── srv/                    # All .srv files
│   │       ├── StartSession.srv
│   │       ├── EndSession.srv
│   │       ├── StartDrill.srv
│   │       ├── SetImuMode.srv      # Switch IMU between nav/training mode
│   │       ├── CalibrateImuPunch.srv
│   │       └── GenerateLlm.srv     # LLM text generation request
│   │
│   ├── boxbunny_core/              # Main processing nodes
│   │   ├── package.xml
│   │   ├── setup.py
│   │   ├── boxbunny_core/
│   │   │   ├── __init__.py
│   │   │   ├── cv_node.py          # Wraps action_prediction/lib/inference_runtime.py, publishes detections + user tracking
│   │   │   ├── imu_node.py         # IMU data processing, mode switching (nav vs training)
│   │   │   ├── robot_node.py       # Robot arm serial interface + height motor control
│   │   │   ├── punch_processor.py  # CV + IMU fusion, defense event derivation, slip detection
│   │   │   ├── session_manager.py  # Session lifecycle (start/pause/stop/data collection)
│   │   │   ├── analytics_node.py   # Statistics computation from raw session data
│   │   │   ├── drill_manager.py    # Combo drill management, sequence validation
│   │   │   ├── sparring_engine.py  # Markov chain sparring sequence generation
│   │   │   └── llm_node.py         # Local LLM AI Coach (wraps Ollama, handles prompts)
│   │   ├── config/
│   │   │   ├── cv.yaml             # CV pipeline parameters (confidence thresholds, smoothing windows)
│   │   │   ├── imu.yaml            # IMU parameters (impact thresholds, nav debounce)
│   │   │   ├── fusion.yaml         # Fusion parameters (time window, pad constraints)
│   │   │   ├── robot.yaml          # Robot serial config, speed mappings, height params
│   │   │   ├── drills.yaml         # Drill definitions (all combo sequences, difficulty mappings)
│   │   │   └── sparring.yaml       # Sparring style matrices, weakness bias parameters
│   │   └── launch/
│   │       ├── boxbunny_full.launch.py    # Launch everything: all nodes + GUI + dashboard
│   │       ├── boxbunny_dev.launch.py     # Launch with IMU simulator instead of real hardware
│   │       ├── imu_simulator.launch.py
│   │       └── headless.launch.py         # Nodes only, no GUI (for testing)
│   │
│   ├── boxbunny_gui/               # PySide6 GUI (robot touchscreen)
│   │   ├── package.xml
│   │   ├── setup.py
│   │   ├── boxbunny_gui/
│   │   │   ├── __init__.py
│   │   │   ├── app.py              # MainWindow setup, page registration
│   │   │   ├── theme.py            # Colors, styles, dark theme
│   │   │   ├── sound.py            # Sound effects manager
│   │   │   ├── gui_bridge.py       # ROS 2 ↔ Qt event bridge (ROS callbacks → Qt signals)
│   │   │   ├── nav/
│   │   │   │   ├── router.py       # Page navigation/routing
│   │   │   │   └── imu_nav_handler.py  # IMU pad navigation (left/right/centre/head)
│   │   │   ├── widgets/
│   │   │   │   ├── big_button.py       # Touch-friendly large button (min 60x60px)
│   │   │   │   ├── timer_display.py    # Round/rest timer with progress ring
│   │   │   │   ├── stat_card.py        # Stat display card
│   │   │   │   ├── punch_counter.py    # Live punch counter
│   │   │   │   ├── combo_display.py    # Visual combo sequence with punch icons
│   │   │   │   ├── coach_tip_bar.py    # AI Coach tip bar (collapsible, top of screen)
│   │   │   │   ├── qr_widget.py        # QR code display (WiFi + URL encoded)
│   │   │   │   ├── account_picker.py   # User account grid with search-as-you-type
│   │   │   │   ├── pattern_lock.py     # Android-style pattern lock grid
│   │   │   │   └── preset_card.py      # Clickable preset card for quick-start
│   │   │   └── pages/
│   │   │       ├── auth/               # Startup, account picker, pattern lock, guest assessment
│   │   │       ├── home/               # Homepage (individual + guest + coach variants)
│   │   │       ├── training/           # Combo drill flow
│   │   │       ├── sparring/           # Sparring flow
│   │   │       ├── performance/        # Power, Stamina, Reaction tests
│   │   │       ├── history/            # All history views
│   │   │       ├── presets/            # Preset management
│   │   │       ├── coach/              # Station mode, session planner, session summary
│   │   │       └── settings/           # Settings page
│   │   └── assets/
│   │       ├── sounds/                 # WAV files (bell, countdown, click, etc.)
│   │       ├── icons/                  # Punch type icons, navigation icons
│   │       └── fonts/
│   │
│   └── boxbunny_dashboard/         # FastAPI mobile dashboard server
│       ├── package.xml
│       ├── setup.py
│       ├── boxbunny_dashboard/
│       │   ├── __init__.py
│       │   ├── server.py           # FastAPI app, WiFi AP management
│       │   ├── websocket.py        # WebSocket manager (phone ↔ robot sync)
│       │   ├── api/
│       │   │   ├── auth.py         # Login, signup, pattern setup, session tokens, guest claim
│       │   │   ├── sessions.py     # Session data, history, stats
│       │   │   ├── presets.py      # Preset CRUD
│       │   │   ├── gamification.py # XP, rank, PRs, streaks, achievements
│       │   │   ├── coach.py        # Coach control (load config, manage station)
│       │   │   ├── chat.py         # AI Coach chat (proxies to llm_node)
│       │   │   └── export.py       # PDF/CSV report generation
│       │   ├── db/
│       │   │   └── manager.py      # SQLite database manager
│       │   ├── templates/          # HTML for mobile dashboard (Jinja2 or static SPA)
│       │   └── static/             # CSS/JS assets
│       └── data/
│           ├── schema.sql          # Main database schema
│           ├── schema_user.sql     # Per-user database schema
│           ├── seed_combos.sql     # 50 combo curriculum seed data
│           └── seed_achievements.sql  # Gamification achievements seed data
│
├── tools/                          # Development & testing utilities
│   ├── imu_simulator.py            # Standalone IMU simulator (6 buttons, publishes ROS topics)
│   ├── hardware_check.py           # Hardware connectivity checker (camera, IMU, robot, LLM)
│   ├── topic_monitor.sh            # ROS topic monitoring script
│   ├── ros_topic_list.py           # List all BoxBunny ROS topics with status
│   ├── db_migrate.py               # Database migration tool (CSV → SQLite)
│   └── demo_data_seeder.py         # Seed databases with demo data for testing
│
├── data/
│   ├── boxing_knowledge/           # Boxing knowledge corpus for LLM RAG
│   └── combos.db                   # Combo template database
│
├── models/                         # Downloaded models (auto-setup)
│   ├── llm/                        # Local LLM model files
│   └── mediapipe/                  # MediaPipe hand tracking models
│
├── scripts/
│   ├── setup.sh                    # One-command bootstrap for fresh Jetson
│   ├── build_knowledge_base.py     # Scrape/build boxing knowledge corpus
│   └── download_models.sh          # Download required models
│
├── _archive/                       # Archived old code (NEVER deleted, added to .gitignore)
│
└── tests/                          # pytest test suite mirroring src/ structure
```

### Database Schema (SQLite — unified)

Move away from CSV for users. Single shared database (`boxbunny_main.db`) for users and shared data, plus per-user database files for training data.

**Users table** (in shared `boxbunny_main.db`):
- id, username, password_hash (bcrypt, not SHA-256), pattern_hash (bcrypt hash of the pattern sequence — nullable, set via phone dashboard), display_name, user_type (individual/coach), level (beginner/intermediate/advanced), created_at, last_login, settings_json, proficiency_answers_json (nullable — from signup questionnaire or guest assessment)
- Coach accounts are created through initial setup or coach invitation — not through regular signup

**Auth sessions table** (in shared db):
- id, user_id (nullable for guests), session_token, device_type (phone/robot), created_at, expires_at, is_active
- Used for phone login sessions and for linking phone auth to robot GUI state

**Guest sessions table** (in shared db):
- id, guest_session_token, created_at, claimed_by_user_id (nullable), claimed_at (nullable), expires_at (7 days after creation)
- Tracks unclaimed guest training data so it can be linked to an account later

**Presets table** (in shared db):
- id, user_id, name, description, preset_type (training/sparring/performance/circuit), config_json (stores all parameters), is_favorite, tags (comma-separated: warmup/cardio/technique/cool-down), created_at, updated_at, use_count

**Coaching sessions table** (in shared db, coach accounts only):
- id, coach_user_id, station_config_preset_id, started_at, ended_at, total_participants, notes

**Coaching participants table** (in shared db):
- id, coaching_session_id, participant_number (auto-increment per session), participant_name (nullable — auto-numbered if not provided), session_data_json (punch counts, distribution, reaction time, fatigue index, round stats, etc.), created_at

**Per-user tables** (in `users/<username>/boxbunny.db`):
- combo_progress (existing schema, keep)
- power_tests (existing schema, keep)
- stamina_tests (existing schema, keep)
- reaction_tests (existing schema, keep)
- sparring_sessions (existing schema, augment with filtered punch data)
- sparring_weakness_profile (existing schema, keep)
- training_sessions (replaces CSV logs — structured records)
- session_events (timestamped event log: round_start, punch_detected, rest_start, etc.)

---

## Feature Specifications

### 1. Two User Modes — Individual vs Coach

#### Authentication & Access Model

**The robot's 10.1-inch touchscreen is not ideal for typing.** All text-heavy interactions (login, signup, chat, detailed stats) happen on the user's phone via the mobile dashboard. The robot GUI is optimized for big-button, gloves-on interaction.

**How it works — WiFi Access Point:**
The Jetson runs its own WiFi hotspot (AP mode) at all times — e.g., network name "BoxBunny". This means phones can connect to the robot directly regardless of what gym WiFi is available, whether there's internet, or whether the user has mobile data. The QR code displayed on the robot screen encodes both the WiFi credentials and the dashboard URL, so scanning it auto-connects the phone and opens the dashboard in one step. No configuration needed from the user — scan and go.

**Robot GUI startup screen — four paths:**

1. **"Start Training" (Guest Mode)** — Large primary button. No login, no typing. Tapping this (or pressing the centre pad) takes the user to a quick skill assessment: 3-4 simple multiple-choice questions shown as big tappable buttons on the touchscreen (e.g., "Have you boxed before? Yes/No", "What's your goal? Fitness / Learn Boxing / Improve Skills", "Preferred intensity? Light / Medium / Hard"). Based on answers, the system suggests a difficulty level and a recommended first drill. The user can accept or tap "Custom" to configure their own session. Session data is stored temporarily under a guest session ID. At the end of the session, the results screen shows a QR code: "Scan to save your session." If they scan, they can create an account on their phone (or log into an existing one) and the guest session data gets linked to their account retroactively. If they don't scan, the data is kept locally for a configurable period (e.g., 7 days) then cleaned up.

2. **"Log In"** — Tapping this opens the **account picker screen**:
   - **Account list**: Shows all existing user accounts as large, tappable avatar/name cards in a scrollable grid (e.g., 2-3 columns). Each card shows the user's display name and level badge. If there are ≤6 accounts, all are visible at once — no scrolling needed. Tap an account → goes straight to the pattern lock screen for that user.
   - **Search filter** (for gyms/shared robots with many accounts): a text field at the top with an on-screen keyboard. As the user types characters, the account list filters in real-time (autocomplete/typeahead). Even typing 1-2 characters should narrow it down significantly. The keyboard should be optional — if there are few accounts, the user never needs to type.
   - **"New? Scan QR to sign up"** link at the bottom — shows a QR code for phone-based account creation.
   - Once an account is selected → **pattern lock screen**: Android-style dot grid. The user draws their pattern with a finger or taps the dots with gloves. If the pattern matches → logged in, robot transitions to homepage with their presets and history loaded.
   - **"Use QR instead"** button on the pattern screen — for users who prefer to authenticate via phone scan instead of pattern. Displays a QR code; scanning it on the phone logs them in and the robot auto-detects.
   - If a user hasn't set a pattern yet (new account created via phone but no pattern configured) → prompt them to either set a pattern now (on the touchscreen) or scan QR to log in. Pattern setup is also available in the phone dashboard settings.

3. **"Coach Login"** — A smaller, secondary button (not hidden, but clearly separate from the individual paths). Opens a similar account picker but filtered to coach accounts only. Tap account → pattern lock → station mode. **Only coach accounts appear here** — it's admin-gated. Individual users cannot enter coach mode even if they try. Coach accounts are created through an initial setup process or by an existing coach inviting a new one (not through the regular signup flow).

**Why this works for a product:**
- Walk-up friendly: anyone can start training in seconds (guest mode with quick assessment, no barrier)
- Account picker: tap your name → draw pattern → training in 5 seconds. No username typing for returning users.
- Search filter: scales to gyms with dozens of users without getting cluttered
- Pattern login: fast, no typing, works with gloves — ideal for returning users
- QR scan: for when you also want the phone dashboard open, or for first-time account creation
- Coach mode is clearly separated — no accidental access, no confusion
- Works in any environment (gym, home, outdoor) because the robot is its own network

**Individual Mode (logged in via phone):**
- Homepage shows: Quick Start (presets), Training, Sparring, Free Training, Performance, History, Settings
- Users see their own stats, history, presets loaded from their account
- After training: QR code to view detailed session dashboard on phone (gamification, AI analysis, export)
- Preset system: save configurations they like, mark favorites, one-tap start on the robot
- Settings and profile management happen on the phone dashboard, not the robot GUI

**Individual Mode (guest — not logged in):**
- After the quick skill assessment, homepage shows training modes: Training, Sparring, Free Training, Performance
- Difficulty is set based on their assessment answers (not hardcoded to Beginner)
- The assessment also generates a "Recommended for you" drill card at the top of the homepage — one-tap start
- No presets (since there's no account), no history
- After each session: results shown on robot screen + QR code to save/claim the session
- If they create an account via the QR at any point, all guest sessions from that visit are linked to their new account, and their skill assessment answers are saved to their profile

**Coach Mode:**
- Coach authenticates on their phone → robot GUI switches to station mode
- **Coach's phone becomes the primary control interface:**
  - Session Planner: create/select station configs (presets) from the phone — more convenient than the small touchscreen for browsing and configuring
  - Start/stop station, switch configs, end session — all controllable from the phone AND the robot touchscreen (both work, coach chooses what's convenient)
  - Live session monitor on phone: see current participant's live stats, participant count, elapsed time
- **Robot screen in coach mode shows the simplified station interface:**
  - Ready state: Large **"Start"** button + current drill name + "Participant #N" label
  - Active state: timer, live punch count, round counter, combo display
  - Completed state: brief results → **"Next Person"** button → resets to ready state
  - Participants are auto-numbered (Participant 1, 2, 3...). Optional: quick name entry via the robot touchscreen (short nickname only, not full signup)
  - The station repeats the loaded config for every participant until the coach ends the session or switches configs
- **Mid-Session Config Switch**: Coach can swap configs from their phone or the robot screen without ending the overall session
- **End of Session Summary**: When the coach ends the coaching session:
  - Robot screen: aggregate dashboard showing total participants and key stats
  - Coach's phone: full detailed report — all participants listed with individual breakdowns, side-by-side comparison, AI Coach analysis, export options
  - Data persists under the coach's account for future review
- **Circuit Training Workflow** (typical use):
  1. Coach scans QR on startup screen → logs in on phone → robot enters station mode
  2. Coach selects a station config on phone (e.g., "Beginner combo drill, 2 rounds × 45s, medium speed")
  3. Participant 1 steps up → taps Start on robot → does the drill → results shown → Next Person
  4. Participant 2 steps up → taps Start → drill → results → Next Person
  5. ... repeat for all participants ...
  6. Coach taps "End Session" (on phone or robot) → robot shows summary, phone shows full report
  7. Coach can review this session later from their phone dashboard under past coaching sessions

### 2. Training Modes

#### 2a. Combo Drills (existing — enhance)
Keep the existing combo curriculum system (50 combos, Beginner/Intermediate/Advanced/Self-Select). Enhance with:
- Visual punch sequence display using icons/animations (not just text)
- Sound cues synced with the combo display (optional setting)
- The robot arm mirrors the combo so the user can follow along
- CV model runs during the session to detect what the user throws
- Post-session: show accuracy (detected punches vs expected combo), timing analysis
- AI Coach can analyze technique and suggest which combos to work on next

#### 2b. Sparring (existing — enhance)
Keep the existing Markov chain sequence generator with 5 boxing styles and weakness bias. Enhance with:
- The robot attacks unpredictably based on the style matrix
- When CV detects the user is idle for too long, the robot should attack to catch them off guard
- When CV detects a block, the robot can react (e.g., follow up with a different angle)
- **Defense tracking via arm IMUs + CV + depth**: when the robot throws a punch, the arm IMU detects if it made contact. Combined with CV block detection and D435i depth/bounding box displacement, the system classifies how the user defended: block, slip, dodge, or hit taken (see Section 4 for full slip derivation logic).
- Post-session: detailed punch breakdown (user's offensive punches via pad IMU + CV) AND defense breakdown (robot punches landed vs defended, defense type), round-by-round analysis
- Difficulty scaling: Easy (slower, more predictable), Medium, Hard (fast, reactive, exploits weaknesses)

Note: arm IMU contact tracking runs in ALL training modes (combo drills, sparring, free training), not just sparring. Any time the robot moves an arm, contact data is recorded. This ensures complete data collection regardless of mode.

#### 2c. Free Training (new)
- Open-ended mode: robot holds pads in position, user throws whatever they want
- CV tracks and records all punches thrown
- Timer optional (can be untimed "open gym" or timed rounds)
- Good for warmup or cool-down
- Minimal UI — just timer + live punch count

### 3. Performance Tests

#### 3a. Power Test (functional with IMU)
- User throws 10 punches at the pads
- IMU measures peak acceleration and force magnitude for each punch
- Displays: peak power (strongest punch), average power across all 10, power per punch bar chart
- If IMU is unavailable (not connected or simulator not running): show "IMU required for power test — connect hardware or launch simulator"
- Historical trend: are they hitting harder session over session?

#### 3b. Stamina Test (existing — enhance)
- Timed punch endurance test (2 minutes default, configurable)
- User punches different pads sequentially as indicated by the GUI
- Count punches via CV detection from the D435i camera
- Track punch rate over time, compute fatigue curve
- Show real-time punch count, rate, and remaining time
- Historical trend: are they improving session over session?

#### 3c. Reaction Time Test (existing — enhance)
- Uses pose estimation (YOLO26n-pose) via the D435i camera to measure how quickly the user's pose changes after a stimulus
- GUI displays a signal (visual flash on screen + optional sound) telling user to punch
- Pose estimator measures time delta between signal and detected motion
- We want RELATIVE reaction time, not absolute — so a tier system:
  - Lightning (top tier) / Fast / Average / Developing / Slow
  - Show trend over time — is the user getting faster?
- Run multiple trials (e.g., 10) and show distribution, best, worst, average
- Random delay between trials to prevent anticipation (1-4 seconds randomized)

### 4. Punch Detection, Depth Sensing & Fusion Pipeline

#### CV Model Output Classes

The custom YOLO CV model detects exactly these classes:
- **Offensive punches**: `jab`, `cross`, `left_hook`, `right_hook`, `left_uppercut`, `right_uppercut`
- **Defensive/neutral**: `block`, `idle`

**The model does NOT detect slips.** Slip detection is derived (see below).

#### Depth Data from D435i

The D435i is a depth camera — use the depth stream in addition to RGB:

- **User depth**: distance from camera to user, measured at the centre of the YOLO pose bounding box. Tracked per frame.
- **Bounding box tracking**: the YOLO pose model provides a bounding box around the user. Track the centre point (x, y) and the top of the bounding box across frames.
- **This data enables**:
  1. **Slip detection** (see below)
  2. **Height auto-adjustment** (see below)
  3. **Analytics**: depth/distance data over a session can show movement patterns — how much the user moves forward/backward, lateral movement, etc. Display in the phone dashboard session summary.

#### Derived Slip Detection

Since the CV model doesn't detect slips, derive them from multiple signals:

A **slip** is detected when ALL of the following are true within a time window:
1. The robot threw a punch (RobotCommand was published)
2. The arm IMU reports `miss` (no contact with user)
3. The CV model does NOT detect `block` (so the user didn't block — they moved)
4. The user's bounding box centre shifted significantly from its baseline position (lateral movement detected via YOLO pose bounding box displacement) OR the user's depth changed meaningfully (they leaned/ducked away from camera)

If conditions 1+2 are true but 3 shows `block` → defense type = "block" (not slip).
If conditions 1+2+3+4 are all true → defense type = "slip".
If conditions 1+2 are true, no block detected, but no significant movement detected → defense type = "dodge" or "unknown".

Publish as `DefenseEvent` with `defense_type = "slip"`, `"block"`, `"dodge"`, or `"unknown"`.

#### Height Auto-Adjustment

The robot needs to match the user's height so the pads are at the right level:

- When a user steps up to the robot (detected by YOLO pose model seeing a person), measure the top of their bounding box relative to the camera frame
- Compute the required height adjustment to centre the pads at the user's torso/head level
- Publish a height command via ROS topic `/boxbunny/robot/height` to the Teensy controlling the height motor
- The adjustment happens automatically during the ready/countdown phase before a session starts
- Can also be manually overridden from the GUI settings or the coach's phone

#### CV Processing Pipeline (within cv_node)

The raw CV model output can be noisy. Apply filtering before publishing:

```
Raw CV detections (per-frame) → Temporal smoothing → Punch event extraction → Publish to ROS topic
```

**Temporal smoothing:**
- Use a sliding window (e.g., 5-10 frames) and require N consecutive frames of the same class to register a detection
- Filter out single-frame spikes (noise)
- Handle transitions between classes (e.g., idle → jab → idle = one jab punch)

**Punch event extraction:**
- Detect the start and end of each punch action
- Extract: punch_type, start_frame, end_frame, confidence (average confidence across frames)
- Compute duration of each punch

#### Fusion (within punch_processor node)

The `punch_processor` ROS node subscribes to CV detections, pad IMU events, arm IMU events, and robot commands. It fuses them in real-time:

**User punch fusion (CV + pad IMU):**
- Match CV detections with pad IMU impacts using a time window (±200ms)
- Apply pad-location constraints to validate or correct CV classification (see Section 14)
- Publish `ConfirmedPunch` messages with full data: type, pad, force level, CV confidence, IMU-confirmed flag

**Defense event fusion (robot command + arm IMU + CV + depth):**
- When a robot punch command is published, open a detection window
- Collect arm IMU contact data, CV block detection, and bounding box displacement within the window
- Derive defense type (block/slip/dodge/unknown) using the logic above
- Publish `DefenseEvent` — this happens in ALL training modes, not just sparring

**Session-level reconstruction (within session_manager):**
- Accumulate confirmed punches and defense events over the session
- Build the ordered event sequence with timing data
- Track depth/distance data over time for movement analytics
- For combo drills: align detected sequence with expected combo to compute accuracy
- At session end: publish `SessionPunchSummary` with aggregated stats

**Output data model (stored in session database):**
```python
@dataclass
class ConfirmedPunchRecord:
    punch_type: str          # jab, cross, left_hook, right_hook, left_uppercut, right_uppercut
    pad: str                 # left, centre, right, head
    timestamp: float         # seconds into the round
    duration: float          # how long the punch action lasted
    level: str               # "light", "medium", "hard" (empty if no IMU)
    force_normalized: float  # 0.33/0.66/1.0 (0.0 if no IMU)
    cv_confidence: float     # 0.0-1.0 CV model confidence
    imu_confirmed: bool      # True if pad IMU impact matched

@dataclass
class DefenseRecord:
    timestamp: float
    robot_punch_code: str    # what punch the robot threw
    arm: str                 # which robot arm
    struck: bool             # True = robot hit user
    defense_type: str        # "block", "slip", "dodge", "unknown"
    user_depth: float        # user's depth at time of event
    bbox_displacement: float # bounding box centre movement magnitude

@dataclass  
class SessionPunchData:
    punch_events: list[ConfirmedPunchRecord]
    defense_events: list[DefenseRecord]
    total_punches: int
    punch_distribution: dict[str, int]
    force_distribution: dict[str, float]   # average force per punch type
    pad_distribution: dict[str, int]       # punches per pad
    average_confidence: float
    peak_force_level: str                  # highest registered impact level
    imu_confirmation_rate: float           # % of CV detections confirmed by IMU
    # Defense stats
    robot_punches_thrown: int
    robot_punches_landed: int
    defense_rate: float                    # defended / thrown
    defense_type_breakdown: dict[str, int] # {"block": 5, "slip": 3, "dodge": 1, ...}
    # Movement analytics
    avg_depth: float                       # average distance from camera
    depth_range: float                     # max - min depth (movement range)
    lateral_movement: float                # total lateral bounding box displacement
```

### 5. Local LLM AI Coach

**Do NOT use the Anthropic API.** Host a local LLM on the Jetson for AI coaching features.

**Setup:**
- Use llama.cpp or Ollama to serve a quantized model locally on the Jetson (the Jetson has a GPU)
- Model recommendation: a small but capable model (e.g., Phi-3-mini, TinyLlama, or Mistral 7B Q4 — whatever fits in Jetson memory). Make model configurable in settings
- The LLM server starts when the app launches and stays running in the background
- Health check: if the LLM server is down, AI features degrade gracefully (show "AI Coach unavailable" instead of crashing)

**AI Coach features:**

*Real-time AI Coach Bar (top of screen during training):*
- A persistent bar at the top of the training/sparring session screen
- During the session: shows contextual tips based on what's happening (e.g., "Try doubling up on your jab" or "Keep your guard up between combos")
- Tips rotate every 15-20 seconds or when triggered by events
- Can be minimized/dismissed

*Post-Session Analysis:*
- After any training session, the AI Coach analyzes the collected data:
  - Punch distribution and accuracy
  - Reaction times
  - Fatigue patterns (if stamina data available)
  - Comparison with previous sessions
- **On the robot results screen**: a brief 1-2 sentence AI summary (e.g., "Strong cross game today. Your jab accuracy dropped in round 3 — try pacing yourself. Check your phone for the full breakdown.")
- **On the phone dashboard session summary**: full 2-3 paragraph natural language analysis, specific drill suggestions, and an "AI Coach Suggest" button that generates a recommended next training session based on recent performance
- The AI Coach is aware of the user's gamification progress and can reference it naturally (e.g., "You're close to breaking your punch count PR — try pushing hard this round" or "One more session this week hits your weekly goal")

*AI Coach Chat (on phone dashboard):*
- Dedicated chat page on the mobile dashboard where the user can type questions and get responses
- The LLM has access to a boxing knowledge base (see section 6)
- Context includes the user's recent training history and performance data
- Users can ask things like:
  - "How do I improve my jab speed?"
  - "What combos should I practice for my level?"
  - "Analyze my last 5 sparring sessions"
  - "Give me a 30-minute training plan for today"

*Coach Mode AI:*
- After a coaching session, the AI can:
  - Analyze aggregate participant performance and highlight common weaknesses
  - Suggest what drills or focus areas to prioritize in the next session
  - Generate a session report summary that the coach can review on their phone
- Coaches can also ask the AI for general coaching advice:
  - Drill recommendations for specific skill levels
  - Training plans for specific goals (competition prep, fitness, technique)
  - How to structure a circuit training class

**System prompts** should establish the LLM as an expert boxing coach who:
- Knows proper technique for all basic punches, combinations, footwork, defense
- Can analyze training data and give specific, actionable feedback
- Adjusts communication style based on user level (simpler language for beginners)
- Focuses on safety (proper form to prevent injury)
- Is encouraging and motivational but honest about areas needing work

### 6. Boxing Knowledge Base

Build a knowledge base from open-source boxing data for the LLM to reference (RAG-style):

**Data to scrape/collect:**
- Boxing technique guides (proper form for each punch type)
- Common combinations and when to use them
- Training methodologies (periodization, progressive overload for boxing)
- Footwork patterns and drills
- Defensive techniques (blocking, slipping, rolling, parrying)
- Conditioning and fitness for boxing
- Common mistakes and corrections for each technique
- Round/session structuring (how to structure a training session)
- Beginner through advanced progression paths

**Implementation:**
- Store as structured text documents in `data/boxing_knowledge/`
- Use a simple embedding + vector search or even keyword-based retrieval
- When the user asks a question or the AI analyzes a session, retrieve relevant knowledge chunks and include them in the LLM prompt context
- Keep it simple — don't over-engineer. A well-organized set of markdown files with a basic search is fine for V1

### 7. Mobile Dashboard & Phone Integration

**This is a central pillar of the product, not an afterthought.** The phone handles everything the small touchscreen isn't good at: login, typing, detailed stats, gamification, AI chat, and coach control. The robot screen handles everything the phone isn't good at: large training displays visible at arm's length, big buttons for gloved hands, real-time session feedback.

#### Network Architecture — WiFi Access Point

The Jetson runs its own WiFi access point (AP mode) using `hostapd` or `nmcli`. This is always on when the robot is powered.

- **SSID**: "BoxBunny" (or "BoxBunny-XXXX" where XXXX is a short unique device ID, for gyms with multiple robots)
- **Security**: WPA2 with a preconfigured password (printed on the robot, changeable in settings)
- **No internet required**: Everything runs locally. The phone connects to the robot's WiFi, accesses the dashboard, and that's it. The phone doesn't need mobile data or gym WiFi.
- **QR code encoding**: The QR code displayed on the robot screen encodes a WiFi network config + URL. On iOS and Android, scanning this QR code auto-prompts the user to join the BoxBunny WiFi and then opens the dashboard URL in the browser. One scan, two actions.

**Why AP mode instead of requiring same-network:**
- Works in any environment — gyms, homes, parks, warehouses — no dependency on existing WiFi infrastructure
- No IT setup needed at a gym (no asking for WiFi passwords, no firewall issues)
- Consistent, predictable connection — the robot IS the network
- Phones auto-connect and disconnect naturally (when you walk away, phone falls back to normal WiFi/data)

#### Server Implementation

- **FastAPI** running on the Jetson in a background thread/process (lightweight, async, good for REST + WebSocket)
- Serves a **mobile-first responsive web app** (HTML/CSS/JS — no native app needed, no app store)
- **REST API endpoints** for:
  - Authentication (login, signup, session token management)
  - User profile (settings, level, preferences)
  - Session data (current session, history, stats)
  - Presets (CRUD, favorites)
  - Gamification (XP, rank, PRs, streaks, achievements)
  - AI Coach chat (send message → LLM → response)
  - Coach controls (load config, start/stop station, end session)
  - Export (generate PDF report, CSV download)
- **WebSocket** for real-time sync between phone and robot:
  - Login on phone → robot GUI updates immediately
  - Coach changes config on phone → robot switches immediately
  - Live session stats pushed to coach's phone during active training
  - Session completion → phone dashboard auto-refreshes with new data
- **URL structure**:
  - `http://boxbunny.local:8080/` — dashboard home (mDNS for friendly URL)
  - `http://boxbunny.local:8080/login` — login/signup
  - `http://boxbunny.local:8080/session/<id>` — specific session summary
  - `http://boxbunny.local:8080/coach` — coach control panel
  - `http://boxbunny.local:8080/chat` — AI Coach chat
  - `http://boxbunny.local:8080/api/...` — REST API

#### Phone Dashboard Pages

**For Individual Users:**
- **Dashboard Home**: Current rank with XP progress bar to next rank, training streak counter, weekly goal progress, recent session score, any new PRs or milestones unlocked
- **Session Summary**: Detailed stats from the most recent (or any past) session — punch breakdown, accuracy, reaction times, fatigue curve, session score, XP earned, any PRs broken, AI coach analysis
- **Training History**: Timeline of all sessions with session scores, trends, and progress charts
- **Performance Trends**: Charts showing power/stamina/reaction time improvement over time
- **Progress & Achievements**: Full rank progression timeline, all unlocked milestones as badges, personal records list with dates and progression, training streak history
- **AI Coach Chat**: Full chat interface — type questions, get responses from the local LLM. Context-aware: the LLM knows your training history, recent sessions, weaknesses, and level. Can ask for drill suggestions, technique tips, training plans, or just general boxing knowledge. Chat history is saved per user.
- **Profile & Settings**: Change display name, password, level, weekly training goal, notification preferences. All profile management happens here, not on the robot screen.
- **Presets Manager**: Create, edit, delete, and reorder presets. Mark favorites (which appear as quick-start cards on the robot homepage when logged in). More convenient than configuring presets on the touchscreen.
- **Export**: Download any session or date range as a PDF report or CSV file

**For Coaches (additional pages):**
- **Coach Control Panel**: The primary tool during a coaching session
  - Preset library with search/filter by tags
  - Load/switch active station config
  - Start/stop station and end session — mirrors the robot screen controls
  - Live view: current participant number, timer, live punch count
  - Quick actions: pause, skip to next person, add note
- **Session Report**: After ending a coaching session — all participants listed with individual stats and session scores, optional leaderboard view, AI analysis of the group, export/share the report
- **Past Sessions**: History of all coaching sessions with date, participant count, drill used, and aggregate stats. Tap into any past session to see the full report again.
- **AI Coach Chat** (coach variant): Same chat interface but the LLM is primed with coaching knowledge — can ask for class planning advice, drill progressions, how to structure a circuit, training recommendations for specific skill levels

#### Guest Session Claiming

When a guest finishes training and scans the QR code:
1. Phone opens the dashboard showing their session results (no login needed to view)
2. Banner at the top: "Create an account to save this session and track your progress"
3. If they sign up or log in: the guest session (and any other guest sessions from this visit) are retroactively linked to their account. XP is calculated, PRs are checked, streaks begin.
4. If they don't sign up: they can still view the current session stats on their phone for that visit. Data stays on the Jetson for 7 days (configurable) in case they come back and want to claim it.

#### Offline / Disconnected Behaviour

- The phone dashboard is only accessible while connected to the BoxBunny WiFi (since the server is on the Jetson)
- If the phone disconnects mid-session, no data is lost — everything is stored on the Jetson. Phone can reconnect and pick up where it left off.
- Export feature allows users to download PDF/CSV to their phone, which they can then view offline or share over their normal network
- Future consideration: optional cloud sync if internet is available (out of scope for V1)

### 8. GUI Input Modes

**Primary: IMU pad navigation (default, always available):**
- The 4 pad IMUs are the primary navigation method when wearing gloves
- Left pad = previous option, Right pad = next option, Centre pad = select/enter, Head pad = back
- Centre pad tap on the ready screen starts the session (triggers countdown)
- Navigation mode is automatically disabled during active training sessions (all taps become punch data)
- This is the default and always-on input mode — no setting needed to enable it

**Secondary: Gesture control (toggleable in settings, default: off):**
- For users who take off their gloves and want to navigate without touching the screen
- Toggle in Settings → Display → "Gesture Navigation: On/Off"
- When enabled, IMU navigation still works simultaneously — gestures are additive
- Use MediaPipe Hands or a similar lightweight hand landmark model
- Run gesture detection on the D435i camera feed (RGB stream)
- Define a small set of clear, unambiguous gestures:
  - **Open palm / stop sign**: Pause/resume the session
  - **Thumbs up**: Confirm / OK / Start
  - **Swipe left/right** (detected via hand movement): Navigate between pages
  - **Fist**: currently not a gesture (too similar to boxing stance)
  - **Two fingers up (peace sign)**: Open quick menu / settings
- Show a small gesture indicator icon in the corner of the screen when gesture control is active
- Gestures should require the hand to be held in position for 0.5-1 second to prevent accidental triggers

**Tertiary: Touchscreen (always available):**
- The touchscreen always works for direct tapping — all buttons are tappable
- This is the fallback and the only input method if both IMU and gesture are unavailable

### 9. Preset System

**Individual Presets:**
- User can save any training configuration as a preset
- Preset stores: mode (training/sparring/performance), all parameters (combo, rounds, time, rest, speed, difficulty, style), and a user-given name
- Presets appear on the homepage as quick-start cards
- "Favorites" — mark up to 3 presets as favorites, these appear prominently
- When starting from a preset: one tap → countdown → session starts. No configuration pages needed
- Presets are synced to the user's profile and persist across logins

**Coach Presets:**
- Coaches can create and manage a library of station configs (presets)
- Before a coaching session: coach selects which preset to load as the active station config
- Can prepare multiple presets in advance and switch between them mid-session
- Presets can be tagged (e.g., "warmup", "cardio", "technique", "cool-down") for organization
- Coaches can share presets with other coach accounts (export/import as JSON)

### 10. Sound Effects

Add audio feedback throughout the app:

| Event | Sound |
|-------|-------|
| Round start | Boxing ring bell (ding ding ding) |
| Round end | Single bell ring |
| Countdown tick (3, 2, 1) | Sharp beep |
| Countdown "GO" | Louder/different beep or air horn |
| Button press | Subtle click |
| Session complete | Achievement chime |
| Punch detected (optional) | Subtle impact sound |
| Reaction time stimulus | Sharp alert tone |
| Rest period start | Gentle tone |
| AI Coach tip appears | Subtle notification |

- All sounds should be toggleable in settings (master volume + individual toggles)
- Use a low-latency audio library appropriate for the chosen GUI framework (e.g., `QSoundEffect` for Qt, `pygame.mixer`, `sounddevice`, or similar)
- Sound files should be small WAV files for minimal latency

### 11. UI/UX Design Requirements

**General:**
- Dark theme (existing) — refine with a consistent color system:
  - Background: very dark gray/near-black (#0D0D0D or #121212)
  - Primary accent: vibrant green (#00E676 or similar — "go" / "active" color)
  - Secondary accent: orange/amber for warnings, rest timers
  - Error/danger: red for stop buttons, warnings
  - Text: white primary, light gray secondary
  - Cards/panels: slightly lighter than background (#1E1E1E)
- All interactive elements must be large enough for sweaty hands with gloves
  - Minimum touch target: 60x60px
  - Spacing between buttons: minimum 16px
- Font: clear, bold, highly legible at arm's length (the user is standing 2-3 feet from the screen)
  - Minimum body text: 18px
  - Headers: 28-36px
  - Timer/counter displays: 72-120px
- Animations: subtle and purposeful (page transitions, progress bars, countdown)
- Loading states: always show feedback when something is processing

**Startup Screen (replaces old login page):**
- Clean, branded screen with BoxBunny logo
- Three clear paths:
  - **"Start Training"** — largest button, centred, primary colour. Goes to quick skill assessment (for guests) or homepage (if returning guest session detected).
  - **"Log In"** — opens account picker (grid of user name cards, search filter, then pattern lock for the selected account). QR fallback available.
  - **"Coach Login"** — smaller, positioned separately. Same account picker flow filtered to coach accounts only. Admin-gated.
- All buttons should be navigable via IMU pads (left/right to move between buttons, centre to select)
- No text fields or on-screen keyboard on THIS screen — the keyboard only appears on the account picker screen if the user taps the search filter

**Homepage (Individual — logged in):**
- Top bar: User name + level badge, AI Coach status indicator, settings gear icon
- Main area: 
  - Quick Start section with favorite preset cards (up to 3)
  - Mode buttons: Training | Sparring | Free Training | Performance
  - Recent session summary card
- Bottom: History button, QR Code button (shows QR to open phone dashboard)

**Homepage (Guest — not logged in):**
- Top bar: "Guest" label, QR code icon (tap to show "Scan to save progress" QR)
- Main area:
  - Mode buttons: Training | Sparring | Free Training | Performance (same as logged in, no presets section)
  - Difficulty defaults to Beginner
- Bottom: History button greyed out (no history without account)

**Homepage (Coach):**
- Top bar: Coach name, session status (idle / station active), session timer (if active)
- Main area:
  - If no active session: Session Planner — select a preset to load as station config, create new preset, or view preset library
  - If station active: large status display showing current config name, participant count so far, and controls (Pause Station / Switch Config / End Session)
- Bottom: Past Sessions (history of coaching sessions), Settings

**Station Mode (full screen during active station):**
- Ready state: Large "Start" button centered, current config name displayed, optional name entry field for participant
- Active state: Timer, round counter, live punch count, combo display — same layout as individual training session
- Completed state: Quick results summary (3-4 key stats), large "Next Person" button, smaller "End Session" button

**During Session:**
- Full screen dedicated to the session
- Timer prominently displayed (large, center-top)
- Current combo/punch sequence displayed clearly
- Round counter (e.g., "Round 2 of 5")
- Live punch count (if CV is running)
- AI Coach tip bar (collapsible, top)
- Stop button always accessible (large, red, corner)
- Rest screen: countdown timer, brief stats from the round just completed

**Results Pages (on robot screen):**
- Clean stat cards with large numbers
- Punch distribution as a visual chart (bar chart or radial)
- Brief AI Coach summary (1-2 sentences — full analysis is on phone)
- QR code: "Scan for full breakdown, AI analysis & progress" (opens session summary on phone dashboard)
- "Save as Preset" button if this was a custom configuration
- "Train Again" and "Home" buttons

### 12. Analytics Engine

Compute and display these analytics:

**Per-Session:**
- Total punches thrown (filtered, confirmed by CV+IMU fusion)
- Punch distribution (count and percentage per type)
- Punch distribution by pad (left/centre/right/head)
- Impact level distribution (light/medium/hard — from IMU)
- Punches per round
- Punches per minute (rate)
- Accuracy (combo drills only — detected vs expected)
- Average reaction time (if applicable)
- Fatigue index (punch rate first 30s vs last 30s of each round)
- Session duration
- Rounds completed
- **Defense stats (all training modes)**: defense rate (% of robot punches avoided), hits taken vs defended, defense type breakdown (blocks vs slips vs dodges — derived from CV + depth + arm IMU)
- **Movement analytics**: average depth from camera, depth range (how much the user moves forward/backward), total lateral displacement (how much they move side to side)

**Historical/Trend:**
- Total sessions, total training time, total punches (all-time)
- Rolling averages (last 7 sessions, last 30 days)
- Personal records (highest punch count, fastest reaction, hardest punch, best defense rate, etc.)
- Progress through combo curriculum (mastery %)
- Improvement trends (charts showing key metrics over time)
- Weakness profile evolution (how it's changed over sessions)
- Defense improvement trend (sparring defense rate over time)
- Power progression (average impact level trend over time)
- Consistency tracking (sessions per week, streaks)

**Coach Analytics:**
- Per-coaching-session: total participants, aggregate stats, individual participant breakdowns
- Per-participant: punch count, distribution, reaction time, fatigue index (all auto-collected)
- Cross-participant: ranked view (highest punch count, fastest reaction, etc.), common weaknesses
- Historical: past coaching sessions with date, participant count, config used, and key aggregates
- AI analysis: LLM-generated summary of the session and recommendations for next class

### 13. Gamification — Training Progression System (Mobile Dashboard Only)

**Important: All gamification UI lives on the mobile phone dashboard, NOT on the robot's touchscreen GUI.** The robot screen stays focused purely on training — clean, distraction-free, no badges or XP counters. The gamification is the reward users discover when they scan the QR code and check their phone. This creates a natural separation: the robot is where you work, the phone is where you see your progress.

The gamification should feel like a natural extension of real boxing training progression — not arcade-style points and flashy animations. Think of it like a martial arts belt system or a strength training log: it tracks your journey, rewards consistency and improvement, and gives you concrete goals to chase.

**Training XP & Rank System:**
- Every training activity earns XP based on effort and quality, not just participation:
  - Combo drill completed: base XP scaled by difficulty level and accuracy (detected punches vs expected combo)
  - Sparring round completed: XP scaled by rounds, duration, and punch output
  - Performance test completed: XP based on percentile improvement over personal baseline
  - Free training: modest XP based on duration and punch volume
- XP accumulates into a visible rank that maps to boxing progression:
  - Novice → Contender → Fighter → Warrior → Champion → Elite
  - Each rank requires progressively more XP (exponential curve so early ranks come fast, higher ranks take real commitment)
- Rank is displayed on the mobile dashboard profile page alongside the user's training history
- Rank should feel earned. A user who trains 3x/week for a month should feel meaningfully different from someone on their second session

**Personal Records (PRs):**
- Automatically track and surface personal bests:
  - Highest punch count in a single round
  - Fastest reaction time
  - Longest training streak (consecutive days/weeks)
  - Highest combo accuracy
  - Most punches in a stamina test
  - Highest power reading (from IMU force measurement)
- When a PR is broken: it's highlighted on the mobile dashboard session summary (e.g., "New PR! 87 punches in one round — beat your previous 82")
- PR history is viewable on the mobile dashboard — shows when each PR was set and the progression over time

**Training Streaks:**
- Track consecutive training days and weekly consistency
- Display current streak on the mobile dashboard home (e.g., "🔥 12-day streak" or "4 weeks consistent")
- Streak milestones: hitting 7 days, 30 days, 100 days, etc. gets acknowledged with a small badge
- Weekly goal: configurable target (e.g., "train 3 times this week") with a simple progress indicator
- If a streak breaks, don't punish — just reset cleanly and show the previous best streak as motivation

**Session Scoring:**
- Every session gets a simple overall score (e.g., out of 100 or a letter grade: S/A/B/C/D)
- Score is computed from a weighted blend of:
  - Volume (punches thrown relative to time available)
  - Accuracy (combo drills: detected vs expected; sparring: clean punch percentage)
  - Consistency (punch rate stability — not dropping off heavily in later rounds)
  - Improvement (better than your recent average = bonus)
- The score is shown on the mobile dashboard session summary — gives users a quick single-number "how was that session?" without needing to analyze every stat
- Historical trend: chart of session scores over time on the mobile dashboard

**Milestones & Achievements:**
- Unlock milestones for meaningful training accomplishments (not trivial ones):
  - "First Blood" — complete your first training session
  - "Century" — throw 100 punches in a single session
  - "Iron Chin" — complete 10 sparring sessions
  - "Speed Demon" — achieve a "Lightning" tier reaction time
  - "Curriculum Master" — complete all 50 combo drills
  - "Marathon" — 50 total training sessions
  - "Consistent" — maintain a 30-day training streak
  - "Well-Rounded" — complete at least one session of every mode (training, sparring, power, stamina, reaction, free training)
- Keep the list focused: 15-20 meaningful milestones, not 200 trivial ones
- Milestones appear on the mobile dashboard profile as badges. Newly unlocked ones are highlighted on the session summary page
- No locked/greyed-out badges visible — only show what's been earned. Upcoming milestones can be hinted at in the AI Coach suggestions ("You're 3 sessions away from unlocking Iron Chin!")

**Coach Mode Gamification:**
- During a coaching session with multiple participants, the station mode can optionally show a **session leaderboard** at the end:
  - Ranks participants by session score, punch count, or reaction time
  - Keeps it friendly — show ranks but emphasize personal effort over competition
  - Coach can toggle this on/off (some coaches may not want competitive pressure)
- The leaderboard is also visible on the coach's QR mobile dashboard after the session

**What to avoid:**
- No gamification UI on the robot's touchscreen — the robot GUI is purely for training. All progress, badges, XP, scores, streaks, and achievements live on the mobile phone dashboard only
- No loot boxes, no virtual currency, no cosmetic unlocks — this is a training tool, not a mobile game
- No notifications that interrupt training sessions on the robot screen
- No punishing mechanics (losing XP, rank demotion) — progress should only go forward
- No social comparison for individual users — your stats are your own. Leaderboards only exist in coach mode where it's a shared group context
- Achievements should never feel arbitrary or padding — every milestone should represent a genuinely meaningful training accomplishment

**Database additions:**
- Add to per-user `boxbunny.db`:
  - `user_xp` table: total_xp, current_rank, rank_history (JSON array of rank-up dates)
  - `personal_records` table: record_type, value, achieved_at, previous_value
  - `achievements` table: achievement_id, unlocked_at
  - `streaks` table: current_streak, longest_streak, last_training_date, weekly_goal, weekly_progress

### 14. IMU Integration & ROS Architecture

#### IMU Hardware Layout

6 IMU sensors (accelerometer + gyroscope) connected to a Teensy microcontroller which communicates with the Jetson via micro-ROS (or a serial bridge node). The Teensy does on-board processing of raw accelerometer data and sends classified events (not raw values).

```
                [HEAD PAD]                  ← IMU 4 (head target)
                    |
[LEFT ARM] ← [LEFT] [CENTRE] [RIGHT] → [RIGHT ARM]
  IMU 5       IMU 1   IMU 2    IMU 3       IMU 6
(robot arm)  (torso targets, L to R)    (robot arm)
```

**Two categories of IMU:**

**Pad IMUs (4) — on the targets the user punches:**
- Detect user punch impacts on the pads
- Teensy classifies each impact as `light`, `medium`, or `hard` — the Jetson receives this classification, not raw acceleration data
- Also used for GUI navigation when not in a training session

**Arm IMUs (2) — on the robot's striking arms:**
- Detect whether the robot's punch made contact with the user
- Teensy sends a binary signal: `struck` (contact detected) or `miss` (no contact)
- Used during sparring to track the user's defensive performance

#### IMU Functions

**1. Pad IMU — Punch Detection (during training session):**
- Each pad IMU detects impacts — Teensy sends classified level (light/medium/hard), pad location, and timestamp
- Combined with CV data for confirmed, classified punches (see Fusion Logic below)
- Impact level maps to force for analytics: light=0.33, medium=0.66, hard=1.0 (normalized)

**2. Pad IMU — GUI Navigation (outside training session):**
- Users can navigate the GUI by tapping the pads with their gloves — no touchscreen needed
- **Left pad tap** → navigate to previous option (like pressing Left/Up)
- **Right pad tap** → navigate to next option (like pressing Right/Down)
- **Centre pad tap** → select / confirm / enter
- **Head pad tap** → go back / cancel
- Navigation mode is active when NOT in an active training session. During a training session, all pad taps are treated as punch data, not navigation.
- The transition between navigation mode and training mode must be clean — session_manager signals when a session starts/stops, and imu_interface switches interpretation accordingly.

**3. Arm IMU — Sparring Defense Tracking (during sparring session):**
- When the robot throws a punch (via `robot_node`), the system starts a short detection window
- If the arm IMU detects contact within that window → `struck` → user failed to defend
- If no contact detected → `miss` → user successfully blocked, slipped, or dodged
- This data feeds into sparring analytics: defense rate (% of robot punches avoided), hits taken, improvement over time
- The CV model's block/slip detection can be cross-referenced with arm IMU data for richer defensive analysis

#### IMU Data Models

```python
# hardware/imu_interface.py

from dataclasses import dataclass
from enum import Enum

class PadLocation(Enum):
    LEFT = "left"
    CENTRE = "centre"
    RIGHT = "right"
    HEAD = "head"

class ArmSide(Enum):
    LEFT = "left"
    RIGHT = "right"

class ImpactLevel(Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HARD = "hard"

class IMUEventType(Enum):
    PAD_IMPACT = "pad_impact"           # User punched a pad (during session)
    NAV_TAP = "nav_tap"                 # User tapped a pad (outside session, for navigation)
    ARM_STRIKE = "arm_strike"           # Robot arm struck the user
    ARM_MISS = "arm_miss"               # Robot arm did not make contact

@dataclass
class PadImpactEvent:
    """Processed impact event from a pad IMU. Teensy has already classified the force level."""
    timestamp: float
    pad: PadLocation
    level: ImpactLevel                  # light, medium, hard — classified by Teensy
    force_normalized: float             # 0.33 (light), 0.66 (medium), 1.0 (hard)

@dataclass
class ArmStrikeEvent:
    """Strike detection event from a robot arm IMU."""
    timestamp: float
    arm: ArmSide                        # which robot arm
    contact: bool                       # True = struck user, False = missed/blocked

@dataclass
class IMUNavEvent:
    """Navigation command derived from a pad tap outside training."""
    timestamp: float
    pad: PadLocation                    # LEFT=prev, RIGHT=next, CENTRE=enter, HEAD=back
    
@dataclass
class IMUSessionData:
    """All IMU data collected during a training session."""
    pad_events: list[PadImpactEvent]
    arm_events: list[ArmStrikeEvent]
    total_pad_impacts: int
    impacts_by_pad: dict[PadLocation, int]
    impacts_by_level: dict[ImpactLevel, int]     # how many light/medium/hard punches
    average_force: float
    peak_level: ImpactLevel                       # highest registered impact
    # Sparring defense stats (only populated in sparring mode):
    robot_punches_thrown: int
    robot_punches_landed: int                     # arm IMU detected contact
    robot_punches_defended: int                   # arm IMU detected miss
    defense_rate: float                           # defended / thrown (0.0-1.0)
```

#### CV + IMU Fusion Logic (in punch_processor.py)

The CV model and pad IMUs provide complementary data. Fusing them gives high-confidence, fully classified punches:

- **Pad IMU tells us**: a punch landed, which pad it hit, how hard (light/medium/hard), and when
- **CV tells us**: what type of punch it was (jab, cross, hook, uppercut, etc.)
- **Together**: punch type + pad location + force level + timing = complete punch record

**Pad fusion rules:**
- A confirmed punch requires BOTH a CV detection AND a pad IMU impact within a short time window (e.g., ±200ms)
- If pad IMU detects an impact but CV doesn't classify a punch → record as "unclassified impact" on that pad (still counts for punch count and force)
- If CV detects a punch but no pad IMU impact → lower confidence detection (possible shadow boxing or miss — flag it but still record)
- The pad location from IMU constrains which punches are valid (reduces CV misclassification):
  - **Centre pad**: jab (1), cross (2), left_uppercut (5), right_uppercut (6) — no hooks
  - **Left pad**: all punches except right_hook (4) and right_uppercut (6)
  - **Right pad**: all punches except left_hook (3) and left_uppercut (5)
  - **Head pad**: all punches valid
- If CV says "left_hook" but pad IMU says "centre pad" → reclassify based on constraints (likely a jab or cross)

**Arm strike processing (sparring only):**
- When `robot_node` publishes a punch command, `punch_processor` opens a detection window (e.g., 500ms)
- If an `ArmStrikeEvent` with `contact=True` arrives within that window → mark as "robot punch landed"
- If the window closes with no contact → mark as "robot punch defended"
- Cross-reference with CV: if CV detected `block` or `slip` during the same window, we know HOW the user defended (blocked vs dodged)

**Degraded modes:**
- If IMU is unavailable (not connected or simulator not running): fall back to CV-only mode with temporal filtering. All features still work, just without force data, pad confirmation, and defense tracking.
- If CV is unavailable (camera disconnected): fall back to IMU-only mode — punch counts and force levels are available per pad, but no punch type classification. Arm strike detection still works.
- If both unavailable: training still runs (robot throws combos, timer works), but no punch data is recorded.

#### ROS Topics

All inter-component communication uses ROS topics. This is the complete topic map for the system:

**IMU Topics (published by Teensy via micro-ROS / serial bridge):**

The Teensy does on-board processing and publishes classified events, not raw sensor data.

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|----------------|-------------|
| `/boxbunny/imu/pad/impact` | `boxbunny_msgs/PadImpact` | Teensy | imu_node | Classified pad impact: which pad, level (light/medium/hard) |
| `/boxbunny/imu/arm/strike` | `boxbunny_msgs/ArmStrike` | Teensy | imu_node | Arm contact detection: which arm, struck (true/false) |
| `/boxbunny/imu/status` | `boxbunny_msgs/IMUStatus` | Teensy | imu_node, GUI settings | Connection status of all 6 IMUs |

**Processed IMU Topics (published by imu_node on Jetson):**

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|----------------|-------------|
| `/boxbunny/imu/punch_event` | `boxbunny_msgs/PunchEvent` | imu_node | punch_processor, gui_bridge | Processed pad punch: pad, level, force_normalized, timestamp |
| `/boxbunny/imu/nav_event` | `boxbunny_msgs/NavCommand` | imu_node | gui_bridge | Navigation tap (only published when session state = idle) |
| `/boxbunny/imu/arm_event` | `boxbunny_msgs/ArmStrikeEvent` | imu_node | punch_processor | Robot arm contact event: arm, contact (bool) |

**CV Topics (published by cv_node on Jetson):**

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|----------------|-------------|
| `/boxbunny/cv/detection` | `boxbunny_msgs/PunchDetection` | cv_node | punch_processor | Per-frame punch class detection: type, confidence |
| `/boxbunny/cv/pose` | `boxbunny_msgs/PoseEstimate` | cv_node | reaction_test, session_manager | Pose estimation keypoints for reaction time |
| `/boxbunny/cv/user_tracking` | `boxbunny_msgs/UserTracking` | cv_node | punch_processor, session_manager | User bounding box, depth, centre displacement — for slip detection & height adjustment |
| `/boxbunny/cv/status` | `std_msgs/String` | cv_node | GUI settings | CV pipeline status |

**Fused/Processed Topics (published by punch_processor):**

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|----------------|-------------|
| `/boxbunny/punch/confirmed` | `boxbunny_msgs/ConfirmedPunch` | punch_processor | session_manager, GUI | Fused CV+IMU confirmed punch: type, pad, force, confidence |
| `/boxbunny/punch/defense` | `boxbunny_msgs/DefenseEvent` | punch_processor | session_manager, GUI | Robot punch defense result: struck or defended, defense type if CV detected |
| `/boxbunny/punch/session_summary` | `boxbunny_msgs/SessionPunchSummary` | punch_processor | session_manager | End-of-session punch summary |

**Robot Arm Topics:**

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|----------------|-------------|
| `/boxbunny/robot/command` | `boxbunny_msgs/RobotCommand` | session_manager | robot_node, punch_processor | Punch command to robot arm (punch_processor listens to open arm strike detection window) |
| `/boxbunny/robot/height` | `boxbunny_msgs/HeightCommand` | session_manager | robot_node | Height adjustment command: target height based on user's YOLO pose bounding box |
| `/boxbunny/robot/round_control` | `boxbunny_msgs/RoundControl` | session_manager | robot_node | Round start/stop signals |
| `/boxbunny/robot/status` | `std_msgs/String` | robot_node | GUI settings | Robot arm connection and height motor status |

**Session Control Topics:**

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|----------------|-------------|
| `/boxbunny/session/state` | `boxbunny_msgs/SessionState` | session_manager | imu_node, cv_node, GUI, drill_manager | Session state changes: idle, countdown, active, rest, complete |
| `/boxbunny/session/config` | `boxbunny_msgs/SessionConfig` | GUI | session_manager | Session configuration when starting a drill |

**Drill Topics (published by drill_manager):**

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|----------------|-------------|
| `/boxbunny/drill/definition` | `boxbunny_msgs/DrillDefinition` | drill_manager | GUI, session_manager | Active drill definition: name, expected combo sequence, difficulty, parameters |
| `/boxbunny/drill/event` | `boxbunny_msgs/DrillEvent` | drill_manager | session_manager, GUI | Drill events: combo_started, combo_completed, combo_missed, round_score |
| `/boxbunny/drill/progress` | `boxbunny_msgs/DrillProgress` | drill_manager | GUI | Real-time drill progress: current combo index, accuracy so far, combos completed/remaining |

**AI Coach Topics:**

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|----------------|-------------|
| `/boxbunny/coach/tip` | `boxbunny_msgs/CoachTip` | llm_node | GUI (coach_tip_bar) | Real-time coaching tip to display on the robot screen during training |

**Custom Message Definitions (boxbunny_msgs):**

```
# boxbunny_msgs/msg/PadImpact.msg
# Published by Teensy — classified on-board, not raw data
float64 timestamp
string pad              # "left", "centre", "right", "head"
string level            # "light", "medium", "hard"

# boxbunny_msgs/msg/ArmStrike.msg
# Published by Teensy — binary contact detection
float64 timestamp
string arm              # "left", "right"
bool contact            # true = struck user, false = missed

# boxbunny_msgs/msg/PunchEvent.msg
# Processed pad impact (published by imu_node)
float64 timestamp
string pad              # "left", "centre", "right", "head"
string level            # "light", "medium", "hard"
float32 force_normalized  # 0.33 (light), 0.66 (medium), 1.0 (hard)

# boxbunny_msgs/msg/ArmStrikeEvent.msg
# Processed arm strike (published by imu_node)
float64 timestamp
string arm              # "left", "right"
bool contact            # true = struck user, false = missed

# boxbunny_msgs/msg/NavCommand.msg
float64 timestamp
string command           # "prev", "next", "enter", "back"

# boxbunny_msgs/msg/PunchDetection.msg
float64 timestamp
string punch_type        # "jab", "cross", "left_hook", "right_hook", "left_uppercut", "right_uppercut", "block", "idle"
float32 confidence       # 0.0-1.0
string raw_class         # raw model output class name

# boxbunny_msgs/msg/ConfirmedPunch.msg
float64 timestamp
string punch_type        # classified punch type
string pad               # which pad was hit
string level             # "light", "medium", "hard" (empty if IMU unavailable)
float32 force_normalized # 0.33/0.66/1.0 (0.0 if IMU unavailable)
float32 cv_confidence    # CV model confidence
bool imu_confirmed       # true if pad IMU impact matched
bool cv_confirmed        # true if CV detection matched

# boxbunny_msgs/msg/DefenseEvent.msg
float64 timestamp
string arm               # which robot arm threw the punch
string robot_punch_code  # "1"-"6" — what punch the robot threw
bool struck              # true = robot hit user, false = user defended
string defense_type      # "block", "slip", "dodge", "unknown" (from CV if available)

# boxbunny_msgs/msg/PoseEstimate.msg
float64 timestamp
float32[] keypoints      # flattened [x1,y1,conf1, x2,y2,conf2, ...] 
float32 movement_delta   # magnitude of pose change from previous frame

# boxbunny_msgs/msg/SessionState.msg
string state             # "idle", "countdown", "active", "rest", "complete"
string mode              # "training", "sparring", "free", "power", "stamina", "reaction"
string username          # current user (or "guest")

# boxbunny_msgs/msg/SessionConfig.msg
string mode
string difficulty
string combo_sequence    # JSON-encoded punch sequence
int32 rounds
int32 work_time_sec
int32 rest_time_sec
string speed             # "slow", "medium", "fast"
string style             # sparring style name

# boxbunny_msgs/msg/RobotCommand.msg
string command_type      # "punch", "set_speed"
string punch_code        # "1"-"6", "3b", "2b"
string speed             # "slow", "medium", "fast"

# boxbunny_msgs/msg/RoundControl.msg
string action            # "start", "stop"

# boxbunny_msgs/msg/HeightCommand.msg
float32 target_height_px   # target top-of-bbox y-coordinate in camera frame
float32 current_height_px  # current detected top-of-bbox y-coordinate
string action              # "adjust", "calibrate", "manual_up", "manual_down", "stop"

# boxbunny_msgs/msg/UserTracking.msg
float64 timestamp
float32 bbox_centre_x      # bounding box centre x (pixels)
float32 bbox_centre_y      # bounding box centre y (pixels)
float32 bbox_top_y         # top of bounding box y (pixels) — used for height adjustment
float32 bbox_width         # bounding box width (pixels)
float32 bbox_height        # bounding box height (pixels)
float32 depth              # distance from camera to user at bbox centre (meters, from D435i depth stream)
float32 lateral_displacement  # change in bbox_centre_x from baseline (pixels) — for slip detection
float32 depth_displacement    # change in depth from baseline (meters) — for slip detection
bool user_detected         # false if no person detected in frame

# boxbunny_msgs/msg/IMUStatus.msg
bool left_pad_connected
bool centre_pad_connected
bool right_pad_connected
bool head_pad_connected
bool left_arm_connected
bool right_arm_connected
bool is_simulator        # true if running from simulator, not real hardware

# boxbunny_msgs/msg/DrillDefinition.msg
string drill_name        # e.g., "jab-cross-hook", "defensive-slip-counter"
string difficulty        # "beginner", "intermediate", "advanced"
string[] combo_sequence  # ordered list of punch codes: ["1", "2", "3"] for jab-cross-hook
int32 total_combos       # how many times the combo repeats in this drill
float32 target_speed     # expected execution speed

# boxbunny_msgs/msg/DrillEvent.msg
float64 timestamp
string event_type        # "combo_started", "combo_completed", "combo_missed", "combo_partial"
int32 combo_index        # which combo repetition (1-based)
float32 accuracy         # 0.0-1.0 for this combo attempt
float32 timing_score     # 0.0-1.0 how close to target tempo
string[] detected_punches  # what the CV+IMU actually detected
string[] expected_punches  # what was expected

# boxbunny_msgs/msg/DrillProgress.msg
float64 timestamp
int32 combos_completed   # how many combos done so far
int32 combos_remaining
float32 overall_accuracy # running accuracy across all combos
float32 current_streak   # consecutive correct combos
int32 best_streak        # best streak this session

# boxbunny_msgs/msg/CoachTip.msg
float64 timestamp
string tip_text          # the coaching tip to display
string tip_type          # "technique", "encouragement", "correction", "suggestion"
string trigger           # what triggered this tip: "low_accuracy", "fatigue", "idle", "periodic", "pr_close"
int32 priority           # 0=low, 1=normal, 2=high (high = display immediately)
```

**Custom Service Definitions (boxbunny_msgs/srv):**

```
# boxbunny_msgs/srv/StartSession.srv
string mode              # "training", "sparring", "free", "power", "stamina", "reaction"
string difficulty        # "beginner", "intermediate", "advanced"
string config_json       # JSON-encoded full session config
string username          # user or "guest"
---
bool success
string session_id
string message

# boxbunny_msgs/srv/EndSession.srv
string session_id
---
bool success
string summary_json      # JSON-encoded session summary
string message

# boxbunny_msgs/srv/StartDrill.srv
string drill_name
string difficulty
int32 rounds
int32 work_time_sec
int32 rest_time_sec
string speed             # "slow", "medium", "fast"
---
bool success
string drill_id
string message

# boxbunny_msgs/srv/SetImuMode.srv
string mode              # "navigation", "training"
---
bool success
string current_mode

# boxbunny_msgs/srv/CalibrateImuPunch.srv
string pad               # "left", "centre", "right", "head" or "all"
---
bool success
string message

# boxbunny_msgs/srv/GenerateLlm.srv
string prompt
string context_json      # JSON with user history, session data, etc.
string system_prompt_key # "drill_feedback", "session_analysis", "technique_tips", "drill_suggestions", "general"
---
bool success
string response
float32 generation_time_sec
```

#### IMU Simulator

A standalone tool for development and testing without the physical Teensy + IMU hardware:

```
tools/imu_simulator.py
```

**Implementation:**
- Small GUI window (can use any lightweight framework — Tkinter is fine since it's a dev tool, not user-facing)
- Layout mirrors the physical hardware:
  ```
              [HEAD]
  [L ARM]  [LEFT] [CENTRE] [RIGHT]  [R ARM]
  ```
- **Pad buttons** (Left, Centre, Right, Head): each click publishes a `PadImpact` message on `/boxbunny/imu/pad/impact`. Include a force level selector (3 radio buttons or toggle: Light / Medium / Hard) so the developer can simulate different impact strengths.
- **Arm buttons** (L Arm, R Arm): each click publishes an `ArmStrike` message on `/boxbunny/imu/arm/strike` with `contact=True`. Hold Shift+click to publish `contact=False` (simulating a miss).
- Publishes an `IMUStatus` message with `is_simulator = true` and all pads/arms marked as connected
- Launchable via: `ros2 launch boxbunny_core imu_simulator.launch.py`
- Can run alongside the main application — the main app doesn't know or care whether the IMU data is real or simulated (same topics, same message format)

#### ROS 2 Launch Files

Located in `src/boxbunny_core/launch/`:
```
├── boxbunny_full.launch.py        # Launch everything: GUI + all ROS nodes + dashboard server
├── boxbunny_dev.launch.py         # Launch with IMU simulator instead of real hardware
├── imu_simulator.launch.py        # Launch just the IMU simulator
└── headless.launch.py             # Launch ROS nodes without the GUI (for headless testing)
```

### 15. Settings

Settings are split between the robot GUI (hardware/system stuff that needs to be configured on-site) and the phone dashboard (user profile and preferences).

**Robot GUI Settings Page** (accessible from homepage gear icon):
- **Hardware**:
  - D435i camera status, RGB stream preview, depth stream status
  - Robot arm connection status and serial port config
  - Height motor status and manual override (up/down buttons + auto-calibrate button)
  - IMU status: per-sensor connection indicator (Left Pad ✓/✗, Centre Pad ✓/✗, Right Pad ✓/✗, Head Pad ✓/✗, Left Arm ✓/✗, Right Arm ✓/✗), simulator mode badge if running from simulator
  - ROS node health: status of each ROS node (imu_node, cv_node, robot_node)
- **AI Coach**:
  - Enable/disable AI features
  - LLM model selection (if multiple installed)
  - LLM server status (running/stopped/error)
  - Test button to verify LLM is working
- **CV Pipeline**:
  - Enable/disable CV
  - Confidence threshold slider
  - Filtering sensitivity (loose/normal/strict)
- **Sound**:
  - Master volume slider
  - Individual sound toggles
  - Test button for each sound
- **Display**:
  - Gesture control toggle
  - Screen brightness (if controllable)
  - Animation speed
- **Network**:
  - WiFi AP status (SSID, connected clients count)
  - Dashboard server status
  - Show QR code for connecting
  - WiFi AP password change
- **System**:
  - Database maintenance
  - Guest data cleanup
  - Software version, hardware info
  - Factory reset option (behind confirmation)

**Phone Dashboard Settings** (on the mobile web app):
- **Profile**: Change display name, password, level override, weekly training goal
- **Preferences**: Notification preferences, default difficulty, preferred training modes
- **Data**: Export all user data, view storage usage, delete account
- **About**: Software version, credits, help/FAQ

### 16. Technical Implementation Notes

**Threading / Process Model:**
- GUI runs on the main thread (event loop — Qt, GTK, or equivalent)
- **ROS nodes run as separate processes** (standard ROS node architecture):
  - `imu_node`: subscribes to raw IMU topics, publishes processed punch/nav events
  - `cv_node`: runs camera capture + YOLO inference, publishes detections and pose estimates
  - `robot_node`: subscribes to robot commands, writes serial to Teensy
  - `punch_processor`: subscribes to both CV and IMU topics, publishes fused confirmed punches
- `gui_bridge`: bridges ROS callbacks into the GUI event loop (runs a ROS spinner in a background thread, emits GUI-native signals/events when ROS messages arrive)
- LLM server: separate process (managed by `subprocess` or systemd)
- LLM inference requests: background thread with async callback/signal for results
- Sound playback: non-blocking (use platform-appropriate audio library)
- Dashboard server: separate process (FastAPI with uvicorn — runs independently of the GUI)
- WebSocket manager: part of the dashboard server process, handles real-time phone↔robot sync
- WiFi AP: managed by the OS (hostapd/NetworkManager) — started on boot, not by the app
- IMU simulator (dev only): separate process, publishes to same ROS topics as real hardware

**CV + IMU Pipeline (ROS-based):**
- Replaces the old file-based trigger/poll approach with real-time ROS topics
- `cv_node` publishes `/boxbunny/cv/detection` per frame → `punch_processor` subscribes
- `imu_node` publishes `/boxbunny/imu/punch_event` per impact → `punch_processor` subscribes
- `punch_processor` fuses both streams in real-time → publishes `/boxbunny/punch/confirmed`
- `gui_bridge` subscribes to `/boxbunny/punch/confirmed` → emits GUI signal to update live punch counter
- `session_manager` subscribes to confirmed punches → accumulates session data → publishes summary at session end
- The file-based spar_trigger.json / spar_cv_output.txt system is no longer needed — everything flows through ROS topics

**Error Handling:**
- Every hardware interface must have a graceful fallback
- No feature crash should crash the app
- Use Qt signal/slot for all async operations
- Log all errors to a file (`logs/boxbunny.log`) with rotation

**Performance on Jetson:**
- YOLO inference should use TensorRT for maximum speed on Jetson GPU
- LLM inference is GPU-bound — may need to time-share GPU between CV and LLM
- If both can't run simultaneously: run CV during sessions, LLM between sessions
- Profile and optimize for consistent 30fps on the camera feed during sessions
- Lazy-load heavy modules (YOLO, LLM) — don't load at startup, load on first use

---

## Implementation Priority

Build in this order:

**Phase 1 — Core Refactor, ROS Foundation & Connectivity:**
1. Restructure codebase into the module structure above
2. **ROS workspace setup**: create `boxbunny_msgs` package with all custom message definitions (including UserTracking, HeightCommand, DefenseEvent), build the message types
3. **ROS nodes**: implement `imu_node`, `cv_node` (with depth stream and user tracking), `robot_node` (with height motor control), `punch_processor`, and `gui_bridge`
4. **IMU simulator**: standalone GUI with 6 buttons (4 pads + 2 arms) that publishes fake IMU data to the same ROS topics as real hardware
5. **ROS 2 launch files**: `boxbunny_full.launch.py`, `boxbunny_dev.launch.py` (with simulator), `imu_simulator.launch.py`, `headless.launch.py`
6. Migrate user management from CSV to SQLite (including guest sessions, auth tokens, pattern hashes)
7. WiFi AP setup (hostapd config, auto-start on boot)
8. FastAPI dashboard server with basic auth API (login, signup, pattern setup, session tokens, guest claiming)
9. WebSocket connection between robot GUI and dashboard server (phone login / pattern login → robot GUI responds)
10. New startup screen on robot GUI (Start Training with guest assessment / Log In with pattern + QR / Coach Login)
11. IMU pad navigation in the GUI (left=prev, right=next, centre=enter, head=back) — works with simulator from day one
12. Account picker UI (tappable user cards grid, search-as-you-type filter) and pattern login (Android-style dot grid)
13. Implement theme system with consistent styling
14. Build reusable UI components (stat_card, timer_widget, punch_display, qr_widget, account_picker, pattern_lock, etc.)
15. Sound manager with basic sound effects
16. Refactor all existing pages into the new structure (keep all current functionality working)

**Phase 2 — Training Features, CV+IMU Fusion & Gamification Backend:**
17. CV + IMU fusion pipeline via ROS topics (real-time punch confirmation with type + pad + force level)
18. D435i depth stream integration — user tracking (bbox, depth, displacement) published as `/boxbunny/cv/user_tracking`
19. Derived slip detection (arm IMU miss + no CV block + bbox/depth displacement)
20. Defense event pipeline — arm IMU contact tracking in ALL training modes, not just sparring
21. Height auto-adjustment via YOLO pose bbox top → `/boxbunny/robot/height` ROS topic
22. Degraded mode handling (CV-only, IMU-only, neither — all graceful)
23. Free Training mode
24. Power test with IMU force measurement
25. Enhanced stamina test with CV punch counting
26. Enhanced results pages with charts, analytics, movement data + QR code to view on phone
27. Analytics engine (per-session and historical, including depth/movement analytics and defense stats)
28. Preset system (individual mode — CRUD on phone dashboard, quick-start on robot)
29. Gamification backend — XP/rank calculation, session scoring, personal records tracking, streak management, achievements (data layer only, no robot GUI)

**Phase 3 — AI & Intelligence:**
30. Local LLM server setup and integration (Ollama — fully local, no cloud APIs)
31. AI Coach engine with system prompts
32. Post-session AI analysis
33. AI Coach chat (on phone dashboard via REST API to local LLM)
34. Boxing knowledge base — scraper/builder script + initial corpus
35. AI Coach tip bar during sessions (on robot screen)

**Phase 4 — Coach Mode:**
36. Coach account type, coach authentication flow (pattern + QR), coach login
37. Station mode UI on robot (start/next person/end session)
38. Coach control panel on phone dashboard (preset selection, start/stop, live stats)
39. Coaching session summary — robot screen (aggregate) + phone (full detailed report with leaderboard and AI analysis)

**Phase 5 — Full Mobile Dashboard:**
40. Dashboard home page (gamification: rank, XP, streaks, PRs, milestones)
41. Training history and performance trends pages (including depth/movement analytics)
42. Progress & achievements page
43. Preset manager on phone (create, edit, favorite, set pattern password)
44. Guest session claiming flow (scan → view results → signup → data linked)
45. Export functionality (PDF reports, CSV data)
46. Coach past sessions history on phone

**Phase 6 — Advanced Interaction:**
47. Gesture detection integration (secondary input, toggleable in settings)
48. Gesture overlay and settings

---

## Important Constraints

1. **Target display: 10.1-inch 1024×600** — all layouts must work at this resolution. No scrolling on primary interaction screens (session, countdown, results). History/settings pages can scroll.
2. **Touch-first design** — big buttons, generous spacing, designed for sweaty hands with gloves. No on-screen keyboard required during training flows.
3. **Jetson Orin NX resources** — 16GB RAM and GPU shared between CV inference, LLM, and GUI. Be memory-conscious. Lazy-load heavy models. Profile GPU usage. Time-share GPU between CV and LLM if needed.
4. **No internet dependency** — everything runs locally. No cloud APIs. The Anthropic API dependency must be removed. The Jetson runs its own WiFi AP so phones connect directly to it.
5. **Phone handles text, robot handles training** — all substantial text input (signup, passwords, chat, profile editing, preset naming) happens on the phone via the mobile dashboard. The robot GUI minimises typing: the account picker has an optional search filter for gyms with many users, but the primary flow (tap account → draw pattern) requires zero typing. The robot screen is for big-button, gloves-on interaction.
6. **Use the best tools for each layer** — evaluate and choose what's best for this project. The existing codebase uses PySide6 and Python 3.9, but there is no requirement to stick with either. If a different GUI framework (Qt6/PySide6, Kivy, or even a local web-based UI via Electron/Tauri/CEF), a different language, or a newer Python version would result in a better product, use that instead. The mobile dashboard is a separate web app. Recommend and justify whatever stack is best for a production embedded touchscreen product running on Jetson with the features described above.
7. **Modern Python** — use the latest stable Python version that the Jetson platform supports well. No need to constrain to 3.9.
8. **Existing data** — migration path for existing user data (CSV → SQLite, existing DBs should be preserved)
9. **Robot interface signatures** — preserve the core logic of `set_speed()`, `send_punch()`, `send_round_start()`, `send_round_stop()` when wrapping them in the ROS `robot_node`. The ROS node is the new entry point, but the underlying serial protocol should remain compatible.
10. **Combo curriculum** — preserve the existing 50-combo database and mastery scoring system
11. **Graceful degradation** — every hardware component (camera, robot arm, IMU, LLM) must be optional. The IMU simulator can stand in for real IMU hardware during development. If IMU is unavailable, fall back to CV-only. If CV is unavailable, fall back to IMU-only. If both unavailable, training still runs but no punch data is recorded. The app should launch and be useful even if every hardware component is disconnected. Guest mode must work with zero setup.
12. **No gamification on robot screen** — all gamification (XP, rank, streaks, achievements, session scores, PRs) is displayed exclusively on the phone mobile dashboard, never on the robot's touchscreen.
13. **ROS is the communication backbone** — all inter-component communication (camera → CV → punch processor, IMU → punch events, session manager → robot arm) must go through ROS topics. No direct function calls between hardware interfaces and the GUI. The `gui_bridge` is the only connection point between ROS and the GUI event loop.
14. **IMU navigation vs training mode** — the system must cleanly switch IMU interpretation between navigation mode (pad taps = GUI navigation) and training mode (pad impacts = punch data). The `session/state` ROS topic is the signal for this transition.

---

## CRITICAL — File Safety Rules

**NEVER delete any files or directories.** This is absolute. Past runs of Claude Code have accidentally deleted entire directories outside the project folder (including the Ubuntu desktop). To prevent this:

1. **No `rm`, `rm -rf`, `rmdir`, `shutil.rmtree`, or any delete operation** — not on project files, not on anything. No exceptions.
2. **When refactoring or replacing old code**: move the old files into a `_archive/` directory at the project root with a timestamped subfolder (e.g., `_archive/2026-03-29_refactor/`). Never delete the originals.
3. **When migrating data** (e.g., CSV → SQLite): keep the original CSV files in `_archive/data_migration/`. Never delete them.
4. **When replacing config files**: copy the old version to `_archive/configs/` before overwriting.
5. **Scope**: Only read and write files inside the project directory. Never touch, modify, or even list files outside the project folder. No operations on `~/Desktop`, `~/Documents`, `~/Downloads`, `/home/*/`, `/tmp` outside the project, or any path that is not a subdirectory of the project root.
6. **If in doubt**: ask before performing any filesystem operation that affects existing files.

The `_archive/` folder should be added to `.gitignore` so it doesn't bloat the repo but remains on disk as a safety net.

---

## Code Quality & Production Standards

This is a product going to market. The code must reflect that.

**Code Organisation:**
- Clean module structure as defined in the Architecture section — no monolithic files
- Each file should have a single clear responsibility
- Maximum ~300 lines per file. If a file is getting longer, split it
- Consistent naming conventions throughout: `snake_case` for files/functions/variables, `PascalCase` for classes
- Every module has an `__init__.py` with clean exports
- Type hints on all function signatures
- Docstrings on all public classes and functions (one-liner for simple ones, Google-style for complex ones)

**Error Handling:**
- No bare `except:` — always catch specific exceptions
- Every hardware interface wrapped in try/except with meaningful error messages and graceful fallback
- Logging throughout — use Python's `logging` module with configurable levels (DEBUG/INFO/WARNING/ERROR)
- Log to file (`logs/boxbunny.log`) with rotation (max 10MB, keep 5 backups)
- No `print()` statements in production code — all output goes through the logger

**Testing:**
- pytest test suite in `tests/` mirroring the source structure
- Unit tests for all core logic (analytics engine, gamification calculations, punch processor, preset manager, fusion logic)
- Integration tests for ROS topic flows (using the IMU simulator)
- Test fixtures for common data (sample sessions, punch events, user profiles)
- Tests should be runnable without any hardware connected

**Configuration:**
- All configurable values in a single config system (`.env` file + config module)
- No magic numbers in code — all thresholds, timeouts, window sizes, etc. are named constants or config values
- Environment-specific configs: `config.dev.env`, `config.prod.env`

**Dependencies:**
- `requirements.txt` (or `pyproject.toml`) with pinned versions
- Separate requirements files if needed: `requirements-jetson.txt` (for Jetson-specific packages like TensorRT), `requirements-dev.txt` (for testing and dev tools)
- Install script or Makefile for one-command setup

---

## Auto-Download & Setup

The system should be able to bootstrap itself on a fresh Jetson. Implement a setup/bootstrap script that handles:

**Models:**
- **LLM model**: Download the chosen quantized model (e.g., via Ollama pull or direct download from Hugging Face). The script should detect if the model is already present and skip if so. Store in `models/llm/`.
- **YOLO26n-pose model**: Download if not present. Store in `models/yolo/yolo26n-pose.pt`.
- **CV punch detection model**: This is a custom trained model — script should check for it in `models/cv/` and print a clear error with instructions if missing (since it can't be auto-downloaded from a public source).
- **MediaPipe models** (for gesture detection): Download if not present. Store in `models/mediapipe/`.

**Boxing Knowledge Base:**
- Build a scraper/downloader script (`scripts/build_knowledge_base.py`) that:
  - Scrapes freely available boxing technique guides, training methodologies, and coaching resources from open-source sources (WikiHow boxing articles, public domain boxing manuals, open coaching guides, Reddit r/amateur_boxing FAQs and wiki)
  - Structures the content into organized markdown files in `data/boxing_knowledge/`
  - Categories: `techniques/`, `combos/`, `training_plans/`, `defense/`, `conditioning/`, `common_mistakes/`, `coaching/`, `faq/`
  - Each file should be a clean, well-structured document that the LLM can reference
  - Include a manifest file (`data/boxing_knowledge/manifest.json`) listing all documents with metadata (title, category, source URL, last updated)
  - The script should be idempotent — running it again updates/adds content without duplicating
- If scraping is impractical for certain sources, write comprehensive knowledge documents manually based on well-known boxing training principles. The goal is a corpus of at least 50-100 documents covering all aspects of boxing training that a coach or trainee might ask about.

**System Dependencies:**
- ROS installation check and setup instructions
- `hostapd` and `dnsmasq` for WiFi AP
- `librealsense` for D435i camera
- CUDA/TensorRT for GPU inference on Jetson
- Ollama or llama.cpp installation

**Bootstrap Command:**
```bash
# One-command setup (should work on a fresh Jetson with ROS installed)
./scripts/setup.sh

# What it does:
# 1. Install Python dependencies (pip install -r requirements.txt)
# 2. Build ROS 2 workspace (colcon build)
# 3. Download models (LLM, YOLO pose, MediaPipe)
# 4. Build boxing knowledge base
# 5. Initialize databases (SQLite schema creation)
# 6. Configure WiFi AP (hostapd config, but don't start it)
# 7. Verify all components and print status report
```
