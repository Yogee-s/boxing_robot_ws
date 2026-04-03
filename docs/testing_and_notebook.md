# BoxBunny Testing Infrastructure and Notebook System

This document provides a comprehensive reference for the BoxBunny test suite, integration tests, hardware checks, and the Jupyter notebook-based runner system used for building, testing, and launching the boxing robot platform.

---

## 1. Notebook Overview

### 1.1 The BoxBunny Runner Notebook

**File:** `notebooks/boxbunny_runner.ipynb`

The runner notebook is the primary interface for all build, test, and launch operations. It is designed to be run from JupyterLab on the Jetson Orin NX. All executable cells delegate to scripts in `notebooks/scripts/`, keeping the notebook itself clean and the scripts independently runnable from the command line.

### 1.2 Notebook Structure -- 12 Sections

The notebook is organised into 12 operational sections plus 3 appendix sections for one-time setup:

| Section | Cell ID | Title | Script(s) | Description |
|---------|---------|-------|-----------|-------------|
| **1a** | Build & Setup | `build_setup.sh` | Runs `colcon build --symlink-install` and seeds demo user data into the database. Must be run after any code changes to ROS packages. |
| **1b** | Unit Tests | `run_tests.sh` | Executes the full pytest suite (171 tests) covering fusion logic, gamification, and database operations. |
| **1c** | Integration Tests | `test_integration.py` | Runs 28 integration tests via `%run` magic. Tests config loading, pad constraints, CV+IMU fusion, message fields, motor protocol, reaction time detection, and punch sequences. No hardware required. |
| **1d** | Hardware Check | `hardware_check.py` | Quick status check of all dependencies and hardware: RealSense camera, PyTorch, CUDA, PySide6, FastAPI, MediaPipe, llama-cpp-python, bcrypt, qrcode, CV model file, YOLO model file, LLM model file, and main database. |
| **2a** | Dev Mode (+ Simulator) | `launch_system.sh --dev` | Full system launch including micro-ROS agent, V4 Arm Control GUI, RealSense camera, all ROS nodes, BoxBunny GUI, **and** the Teensy Simulator window. The simulator mirrors real pad strikes and can inject simulated ones for testing. |
| **2b** | Production Mode | `launch_system.sh` | Same as 2a but **without** the Teensy Simulator. For actual training use. |
| **3a** | GUI + Simulator + Arms + ROS | `launch_gui_imu.sh` | Full hardware pipeline: micro-ROS + V4 GUI + core ROS nodes + BoxBunny GUI + Teensy Simulator. Pad strikes flow through the complete pipeline. Requires calibration (A2). |
| **3b** | Simulator + Arms Only | `launch_imu_sim.sh` | Just the Teensy Simulator paired with robot hardware. No BoxBunny GUI or ROS nodes. For testing motor execution and IMU detection in isolation. |
| **3c** | GUI Visual Test | `launch_gui.sh` | Only the BoxBunny PySide6 GUI for visual inspection. No ROS, no Teensy, no hardware. For checking layouts and styling. |
| **4a** | CV Model Live Test | `launch_cv_test.sh` | Launches action prediction with live camera. Shows pose skeleton + action labels in an OpenCV window. User stands 1.5-2m from camera. Press `q` to stop. Requires RealSense D435i and conda `boxing_ai` environment. |
| **4b** | Reaction Time Test | `launch_gui_page.sh reaction_test` | Launches BoxBunny GUI directly to the Reaction Test page. Camera feed with pose skeleton overlay, 3-2-1 countdown, green stimulus flash, "return to neutral" prompt between attempts, and results with tier + rolling history. |
| **4c** | CV + IMU Fusion Test | `launch_cv_imu_test.sh /dev/ttyACM0 115200` | Launches the original `run.py` CV inference window and the Teensy Simulator side by side with the Teensy connected. Left window: CV predictions, FPS, action probabilities, config. Right window: Teensy Simulator with pad strikes, live data, punch buttons, person tracking. Background: `imu_node` + `punch_processor` for the complete fusion pipeline. Requires conda `boxing_ai`, Teensy connected, calibration, and RealSense. |
| **5a** | Phone Dashboard | `dashboard_tunnel.py` | Starts the Flask dashboard server with SSH tunnel and generates QR code for phone access. |
| **5b** | LLM Coach Test | `test_llm.py` | Loads the Qwen 2.5-3B GGUF model and generates a sample coaching tip to verify the LLM pipeline. |
| **5c** | Sound Test | `play_sounds.py` | Plays all 18 sound effects sequentially to verify audio output. |
| **5d** | Demo User Profiles | `view_user_dashboards.py` | Displays visual cards for the 4 demo users (Alex, Maria, Jake, Sarah) with their XP, streaks, and records. |
| **5e** | Benchmark Test | `benchmark_test.py` | Tests percentile ranking calculations against population norms from `data/benchmarks/population_norms.json`. |
| **A1** | Micro-ROS Install | `setup_microros.sh` | One-time installation of the micro-ROS agent for Teensy communication. Must run outside conda. |
| **A2** | Arm Calibration | `launch_arm_calibration.sh` | Launches the V4 Arm Control GUI for motor calibration. Creates `arm_config.yaml`, `strike_library.json`, and `ros_slots.json`. |
| **A3** | Build Vue Frontend | `build_frontend.sh` | Rebuilds the phone dashboard Vue SPA. Only needed after changing `src/boxbunny_dashboard/frontend/` source files. |

### 1.3 Cell Execution Patterns

The notebook uses two distinct execution patterns:

1. **`%%bash --no-raise-error`**: Used for long-running processes (system launches, GUI applications, hardware tests). The `--no-raise-error` flag prevents Jupyter from killing the cell on non-zero exit codes, which is normal when the user closes a GUI window or presses Ctrl+C. These cells run until the launched process terminates or the user clicks the Jupyter "Stop" button.

2. **`%run script.py`**: Used for inline Python scripts (integration tests, hardware check, LLM test, sound test). Output appears directly in the notebook output cell with colored PASS/FAIL indicators.

### 1.4 Quick Start Path

For new users, the recommended workflow is:

1. Run **1a** (Build & Setup) -- builds all ROS packages and seeds the database
2. Run **1b** (Unit Tests) -- verifies the core logic works
3. Choose a launch mode:
   - **2a** (Dev Mode) for full system with simulator
   - **3c** (GUI Visual Test) for UI-only development
   - **4a** (CV Model Live) for CV pipeline testing

For first-time setup on new hardware, also run **A1** (Micro-ROS Install) and **A2** (Arm Calibration).

---

## 2. Test Suite

### 2.1 Test Framework and Configuration

The test suite uses **pytest** as the test runner. Tests are located in `tests/` with the following structure:

```
tests/
    __init__.py
    conftest.py          -- Shared fixtures (database, users, sessions, punches)
    unit/
        __init__.py
        test_punch_fusion.py   -- CV+IMU fusion logic (428 lines)
        test_gamification.py   -- XP, ranks, streaks, scoring (353 lines)
        test_database.py       -- Database CRUD operations (363 lines)
    integration/
        (reserved for future integration tests)
```

**Running tests:**
```bash
python3 -m pytest tests/ -v
```

Or via the notebook: cell **1b** runs `notebooks/scripts/run_tests.sh`.

### 2.2 Shared Test Fixtures (`conftest.py`)

The `conftest.py` file provides 12 pytest fixtures that all test files can use. Every fixture uses mock data -- no hardware, ROS, camera, or IMU is required.

**Path setup:** The conftest adds `src/boxbunny_dashboard`, `src/boxbunny_core`, and `action_prediction` to `sys.path` so all project modules are importable.

#### 2.2.1 Database Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `tmp_data_dir` | function | Creates a temporary directory with copies of all SQL schema files from `data/schema/`. Each test gets a fresh directory. |
| `db_manager` | function | Instantiates a `DatabaseManager` pointing at the temporary directory. Creates fresh empty databases for each test. |
| `sample_user` | function | Creates a user (`testuser` / `testpass123` / "Test User" / individual / beginner) via `db_manager.create_user()` and returns a dict with `id`, `username`, and `display_name`. |

#### 2.2.2 User Profile Fixtures

| Fixture | Description |
|---------|-------------|
| `beginner_profile` | Dict for "Beginner Bob": 120 XP, Novice rank, 4 sessions, prefers 1-2 and 1-1-2 combos |
| `intermediate_profile` | Dict for "Intermediate Ida": 2800 XP, Fighter rank, 45 sessions, prefers 1-2-3 and 3b-3 combos |
| `coach_profile` | Dict for "Coach Charlie": 15000 XP, Champion rank, coach user type |

#### 2.2.3 Session Data Fixtures

| Fixture | Description |
|---------|-------------|
| `sample_session_data` | Training session: 3 rounds, 87 punches, 75% defence, jab-cross-hook combo |
| `sample_sparring_session` | Sparring session: 4 rounds, 210 user punches, 32 robot punches (8 landed), 75% defence |

#### 2.2.4 Punch and CV/IMU Fixtures

| Fixture | Description |
|---------|-------------|
| `sample_punch_events` | 5 confirmed punch events with type, pad, force, CV confidence, and timestamp |
| `sample_cv_detections` | 5 CV detection events spaced 0.5-1.5s apart with timestamps, types, and confidences |
| `sample_imu_impacts` | 5 IMU pad impact events timed ~50ms after corresponding CV detections |
| `sample_defense_events` | Defence evaluation data with arm events, CV block detections, and tracking snapshots |

#### 2.2.5 Preset and State Machine Fixtures

| Fixture | Description |
|---------|-------------|
| `sample_preset` | "Quick Jab Drill": training preset with jab-cross combo, 3 rounds, 120s work, 45s rest |
| `action_labels` | Standard 8-class label list: jab, cross, left_hook, right_hook, left_uppercut, right_uppercut, block, idle |
| `state_machine_config` | Default CausalActionStateMachine config: enter_consecutive=2, exit_consecutive=2, min_hold_steps=2, sustain_confidence=0.78, peak_drop_threshold=0.02 |

### 2.3 Unit Test: Punch Fusion (`test_punch_fusion.py`)

**File:** `tests/unit/test_punch_fusion.py` (428 lines)

Tests the core CV+IMU fusion logic without hardware. Imports directly from `boxbunny_core.punch_fusion` and `boxbunny_core.constants`.

#### Test Classes and What They Cover:

**`TestRingBuffer`** (10 tests)
- Append and length tracking
- `maxlen` eviction (buffer overflow discards oldest)
- `expire()` removes items at or before a given timestamp
- `expire()` on empty buffer returns empty list
- `pop_match()` finds first item within a timestamp window
- `pop_match()` returns `None` when no match exists
- `pop_match()` returns the first matching item when multiple are in range
- `clear()` empties the buffer

**`TestCVIMUFusion`** (3 tests)
- CV and IMU events within 200ms window match correctly
- CV and IMU events more than 200ms apart do NOT match
- Multiple IMU events in window: `pop_match()` returns the first (closest temporal match)

**`TestPadConstraints`** (10 tests)
- Jab is valid on centre pad, invalid on left and right pads
- Left hook is valid on left pad, invalid on right pad
- Right hook is valid on right pad, invalid on left pad
- All offensive punches are valid on head pad
- Left uppercut is valid on left pad
- Right uppercut is valid on right pad

**`TestReclassification`** (10 tests)
- Valid punch type passes through unchanged (e.g., jab on centre -> jab)
- Invalid punch reclassified via secondary class list (e.g., right_hook on left pad with left_hook secondary -> left_hook)
- Invalid punch with no valid secondary -> `unclassified`
- Invalid punch with no secondary list at all -> `unclassified`
- Secondary class below minimum confidence threshold -> `unclassified`
- Secondary class at exactly minimum confidence -> accepted
- Unknown pad name -> punch passes through without constraint
- Cross validated on centre only (invalid on left and right)

**`TestExpiredEvents`** (4 tests)
- CV events expire correctly at the cutoff timestamp
- IMU events expire correctly
- `expire()` with high cutoff removes all items
- `expire()` with low cutoff removes nothing

**`TestDefenseClassification`** (tests for the `classify_defense()` function)
- Arm contact + CV block detection -> `clean_block`
- No arm contact + evasive lateral displacement -> `slip`
- Various combinations mapping to defence types from the `DefenseType` enum

**`TestSessionStats`** (tests for movement tracking)
- `record_tracking()` stores displacement timeline entries
- Max lateral and depth displacement tracked
- Summary fields include `max_lateral_displacement`, `max_depth_displacement`, and `movement_timeline_json`

### 2.4 Unit Test: Gamification (`test_gamification.py`)

**File:** `tests/unit/test_gamification.py` (353 lines)

Tests the XP calculation, rank progression, session scoring, streak management, and achievement unlocking. All tests use pure functions -- no database or hardware.

#### Test Classes:

**`TestRankSystem`** (10 tests)
- Correct rank at each threshold boundary (0=Novice, 500=Contender, 1500=Fighter, 4000=Warrior, 10000=Champion, 25000=Elite)
- Between-rank values return the lower rank (e.g., 999 -> Contender)
- Very high XP -> Elite
- XP to next rank calculation from zero (500 to Contender)
- XP to next rank from Elite (0 remaining)
- XP just below threshold (499 -> 1 XP to Contender)
- Rank indices increase monotonically with XP (no regression)

**`TestSessionXP`** (10 tests)
- Base XP values per mode: training=50, sparring=75, free=25, power=30, stamina=40, reaction=30
- Sparring yields more XP than free training (same score)
- Complete sessions earn more XP than incomplete (1.5x bonus)
- Higher score yields more XP (score multiplier)
- Streak bonus adds flat XP
- Minimum XP floor: even zero-score incomplete sessions earn positive XP (0.5 minimum multiplier)

**`TestSessionScore`** (8 tests)
- Session scoring formula: `volume_ratio*30 + accuracy*30 + consistency*25 + improvement*15`
- Perfect score (all 1.0) = 100
- Zero score (all 0.0) = 0
- Average score (all 0.5, no improvement) between 30-50
- Score clamped to [0, 100] range
- Volume weight contributes exactly 30 points at maximum
- Improvement bonus adds exactly 15 points

**`TestRankProgression`** (via database, uses `db_manager` and `sample_user` fixtures)
- Initial XP is zero for new users
- XP is correctly incremented after sessions
- Rank changes at threshold boundaries

**`TestStreaks`** (via database)
- Streak increments on consecutive training days
- Streak resets after a gap day
- Longest streak tracks the historical maximum

**`TestAchievements`** (via database)
- Achievement unlock on criteria met
- Achievement persistence across sessions
- Progress tracking toward multi-step achievements

### 2.5 Unit Test: Database (`test_database.py`)

**File:** `tests/unit/test_database.py` (363 lines)

Tests the `DatabaseManager` class from `boxbunny_dashboard.db.manager`. Every test uses the `db_manager` fixture which creates fresh temporary databases.

#### Test Classes:

**`TestUserManagement`** (11 tests)
- `create_user()` returns a positive user ID
- Duplicate username returns `None`
- `verify_password()` succeeds with correct password
- `verify_password()` fails with wrong password
- `verify_password()` fails with nonexistent username
- `get_user()` returns correct fields (username, display_name, user_type, level, created_at)
- `get_user()` returns `None` for nonexistent ID
- `get_user_by_username()` works correctly
- `list_users()` returns all users
- `list_users(user_type="coach")` filters correctly
- `list_users()` returns empty list when no users exist
- Coach user creation with type and level fields

**`TestPatternLock`** (6 tests)
- `set_pattern()` then `verify_pattern()` succeeds
- Wrong pattern returns `False`
- No pattern set returns `False`
- Pattern is stored as bcrypt hash (starts with `$2`, not plaintext)
- Nonexistent user returns `False`
- Overwriting pattern: new pattern works, old pattern fails

**`TestGuestSessions`** (5 tests)
- `create_guest_session()` returns a token string (length > 8)
- Multiple guest sessions produce unique tokens
- `claim_guest_session()` succeeds for valid token + user
- Double claim returns `False`
- Claiming nonexistent token returns `False`
- `cleanup_expired_guests()` exercises the cleanup path

**`TestPresets`** (6 tests)
- Create and retrieve preset by user ID
- Update preset fields (name, is_favorite)
- Invalid field update returns `False`
- Increment use count
- Multiple presets per user
- Favourite presets sorted first in query results

**`TestSessions`** (tests for training session CRUD)
- Create training session with all fields
- Session retrieval with summary JSON parsing
- Session history pagination
- Session state transitions

### 2.6 Test Count Summary

The total test count across all unit test files is approximately **171 tests** covering:

| Category | Approx. Count | Files |
|----------|---------------|-------|
| Ring buffer operations | 10 | test_punch_fusion.py |
| CV+IMU fusion matching | 3 | test_punch_fusion.py |
| Pad-location constraints | 10 | test_punch_fusion.py |
| Reclassification logic | 10 | test_punch_fusion.py |
| Event expiry | 4 | test_punch_fusion.py |
| Defence classification | ~8 | test_punch_fusion.py |
| Session stats / movement | ~5 | test_punch_fusion.py |
| Rank system | 10 | test_gamification.py |
| Session XP calculation | 10 | test_gamification.py |
| Session scoring | 8 | test_gamification.py |
| Rank progression (DB) | ~8 | test_gamification.py |
| Streaks (DB) | ~5 | test_gamification.py |
| Achievements (DB) | ~5 | test_gamification.py |
| User management | 11 | test_database.py |
| Pattern lock | 6 | test_database.py |
| Guest sessions | 5 | test_database.py |
| Presets | 6 | test_database.py |
| Training sessions (DB) | ~10 | test_database.py |
| Other DB operations | ~37 | test_database.py |

---

## 3. Integration Tests

### 3.1 Overview

**File:** `notebooks/scripts/test_integration.py` (444 lines)

The integration test script performs **28 targeted tests** that verify the critical integration points between subsystems without requiring any hardware. It is executed via `%run` in notebook cell 1c or directly from the command line:

```bash
python3 notebooks/scripts/test_integration.py
```

The script uses a simple custom test framework (not pytest) with colored PASS/FAIL output and a final summary. It adds `src/boxbunny_core` to the Python path and imports real modules.

### 3.2 Test Categories

#### 3.2.1 Config Loading (2 tests)

| # | Test | What It Verifies |
|---|------|-------------------|
| 1 | Load boxbunny.yaml with new fusion params | `cv_only_min_consecutive_frames == 3`, `cv_only_min_confidence == 0.6`, `imu_only_default_confidence == 0.3` |
| 2 | Load free_training config | `counter_cooldown_ms == 1500`, `pad_counter_strikes` dict contains `"centre"` key with `"1"` as a valid strike |

Both tests use `boxbunny_core.config_loader.load_config()` to load `config/boxbunny.yaml`.

#### 3.2.2 Constants & Topics (5 tests)

| # | Test | What It Verifies |
|---|------|-------------------|
| 3 | New topic constants exist | `Topics.MOTOR_COMMANDS`, `Topics.MOTOR_FEEDBACK`, `Topics.ROBOT_HEIGHT_CMD`, `Topics.ROBOT_STRIKE_DETECTED`, `Topics.ROBOT_STRIKE_COMPLETE`, `Topics.CV_DEBUG_INFO` |
| 4 | Pad constraints: centre | Only jab and cross are valid on the centre pad. Hooks and uppercuts are explicitly excluded. |
| 5 | Pad constraints: left | Only left_hook and left_uppercut are valid (exactly 2 punches). |
| 6 | Pad constraints: right | Only right_hook and right_uppercut are valid (exactly 2 punches). |
| 7 | Pad constraints: head | All offensive punch types are valid (jab, cross, all hooks, all uppercuts). |

#### 3.2.3 CV+IMU Fusion Logic (6 tests)

| # | Test | What It Verifies |
|---|------|-------------------|
| 8 | PendingCV has consecutive_frames field | The `PendingCV` dataclass accepts and stores `consecutive_frames` |
| 9 | PendingIMU has accel_magnitude field | The `PendingIMU` dataclass accepts and stores `accel_magnitude` |
| 10 | infer_punch_from_pad: centre -> jab | Pad-to-punch inference mapping: centre=jab, left=left_hook, right=right_hook, head=jab, unknown=unclassified |
| 11 | Reclassify: jab on left pad -> unclassified | Jab is not valid on the left pad, so it becomes unclassified |
| 12 | Reclassify: jab on centre -> jab | Jab is valid on centre pad, passes through unchanged |
| 13 | Reclassify: r_hook on left with l_hook secondary -> l_hook | Right hook is invalid on left pad, but the secondary class list offers left_hook which IS valid |
| 14 | RingBuffer CV-IMU matching within window | IMU event at timestamp 10.0 is found by `pop_match(9.9, 10.1)` and returns correct pad and accel_magnitude |

#### 3.2.4 Session Stats & Movement (2 tests)

| # | Test | What It Verifies |
|---|------|-------------------|
| 15 | SessionStats records movement timeline | `record_tracking()` with 0.6s delay between calls (exceeding 0.5s sampling interval) creates timeline entries. Max lateral and depth displacement tracked correctly. |
| 16 | SessionStats summary includes movement fields | `to_summary_fields()` returns `max_lateral_displacement`, `max_depth_displacement`, and `movement_timeline_json` (valid JSON list) |

#### 3.2.5 Motor Command Protocol (2 tests)

| # | Test | What It Verifies |
|---|------|-------------------|
| 17 | Motor command format | Verifies the command payload is exactly 9 doubles: 4 positions + 4 speeds + 1 enable flag. All values are `float`. |
| 18 | Motor feedback parse | Verifies the feedback format is exactly 21 doubles: 4 positions + 4 currents + 1 CAN count + 12 IMU values (3 axes x 4 pads). Computes IMU0 acceleration magnitude from the first 3 IMU values and confirms it is consistent with gravity (~9.81 m/s^2). |

#### 3.2.6 InferenceEngine (3 tests)

| # | Test | What It Verifies |
|---|------|-------------------|
| 19 | InferenceEngine class imports | `InferenceEngine`, `InferenceResult`, and `RollingFeatureBuffer` import from `action_prediction.lib.inference_runtime`. Gracefully skips if PyTorch is not available. |
| 20 | InferenceResult dataclass fields | Verifies fields exist: `action`, `confidence`, `consecutive_frames`, `movement_delta`, `keypoints`, `bbox`, `fps`. Default `action` is `"idle"`, default `consecutive_frames` is 0. |
| 21 | RollingFeatureBuffer normalisation modes | Creates a buffer with `clip_p90` normalisation, adds 3 frames of random data, confirms `is_ready` is True and feature shape is `(3, 64)`. |

#### 3.2.7 Reaction Time Motion Detection (2 tests)

| # | Test | What It Verifies |
|---|------|-------------------|
| 22 | Keypoint motion detection above threshold | Simulates 17 COCO keypoints with a 30px shoulder displacement. Confirms max displacement exceeds the 20px detection threshold. |
| 23 | Keypoint motion below threshold (idle) | Simulates a small ~5.4px shift. Confirms it does NOT exceed the threshold. |

#### 3.2.8 ROS Message Fields (4 tests)

| # | Test | What It Verifies |
|---|------|-------------------|
| 24 | ConfirmedPunch has accel_magnitude field | Creates a `ConfirmedPunch` message, sets `accel_magnitude = 35.5`, confirms it reads back correctly |
| 25 | PunchDetection has consecutive_frames field | Sets `consecutive_frames = 7`, confirms readback |
| 26 | PadImpact has accel_magnitude field | Sets `accel_magnitude = 42.0`, confirms readback |
| 27 | SessionPunchSummary has movement fields | Sets `max_lateral_displacement`, `max_depth_displacement`, and `movement_timeline_json`, confirms all read back correctly |

#### 3.2.9 Punch Sequences (1 test)

| # | Test | What It Verifies |
|---|------|-------------------|
| 28 | Punch sequence files exist and are valid JSON | Scans `data/punch_sequences/*.json`. Asserts at least 6 files exist. For each file: parses as JSON, confirms it is a list with at least 2 waypoints, and each waypoint has a `"pos"` key with at least 2 elements. |

### 3.3 Output Format

The script prints a structured report with section headers and colored PASS/FAIL indicators:

```
============================================================
  BoxBunny Integration Tests
============================================================

── Config Loading ──
  [ 1] Load boxbunny.yaml with new fusion params ... PASS
  [ 2] Load free_training config ... PASS

── Constants & Topics ──
  [ 3] New topic constants exist ... PASS
  ...

============================================================
  28/28 passed, 0 failed
============================================================
```

If any test fails, the exit code is 1 (non-zero), which is captured by Jupyter as a cell error.

---

## 4. Hardware Check

### 4.1 Overview

**File:** `notebooks/scripts/hardware_check.py` (47 lines)

A quick diagnostic script that probes all hardware and software dependencies. Executed via `%run` in notebook cell 1d. Designed to be the first thing you run when setting up a new environment or debugging a launch failure.

### 4.2 Checks Performed

The script performs 13 checks, each structured as a name + lambda function:

| Check | What It Probes | OK Condition |
|-------|----------------|--------------|
| RealSense Camera | `pyrealsense2.context().query_devices().size()` | At least 1 device connected |
| PyTorch | `torch.__version__` | Module imports, shows version |
| CUDA Available | `torch.cuda.is_available()` | Returns `True` |
| PySide6 | `PySide6.__version__` | Module imports, shows version |
| FastAPI | `fastapi.__version__` | Module imports, shows version |
| MediaPipe | `mediapipe.__version__` | Module imports, shows version |
| llama-cpp-python | `llama_cpp.__version__` or `'installed'` | Module imports |
| bcrypt | `bcrypt.__version__` | Module imports, shows version |
| qrcode | `import qrcode` | Module imports |
| CV Model | `os.path.exists('action_prediction/model/best_model.pth')` | File exists at expected path |
| YOLO Model | `os.path.exists('action_prediction/model/yolo26n-pose.pt')` | File exists at expected path |
| LLM Model | `os.path.exists('models/llm/qwen2.5-3b-instruct-q4_k_m.gguf')` | File exists at expected path |
| Main DB | `os.path.exists('data/boxbunny_main.db')` | Database file exists |

### 4.3 Output Format

Each check prints a single line with a `+` (OK) or `-` (MISSING) indicator:

```
=== Hardware & Dependency Check ===
  [+] RealSense Camera: OK
  [+] PyTorch: OK (2.1.0)
  [+] CUDA Available: OK
  [+] PySide6: OK (6.6.1)
  [+] FastAPI: OK (0.109.0)
  [+] MediaPipe: OK (0.10.9)
  [+] llama-cpp-python: OK (installed)
  [+] bcrypt: OK (4.1.2)
  [+] qrcode: OK
  [+] CV Model: OK
  [+] YOLO Model: OK
  [+] LLM Model: OK
  [+] Main DB: OK
```

If a check fails, the error type is shown:
```
  [-] RealSense Camera: MISSING (RuntimeError)
```

---

## 5. Manual Testing Guide

### 5.1 Testing the CV Model (Cell 4a)

**What it launches:** The `action_prediction/run.py` script with live RealSense D435i camera feed.

**Prerequisites:**
- RealSense D435i camera connected via USB 3.0
- Conda `boxing_ai` environment activated (with PyTorch, YOLO, MediaPipe)
- Model files present: `action_prediction/model/best_model.pth` and `yolo26n-pose.pt`

**What to do:**
1. Stand 1.5-2 metres from the camera
2. Perform boxing punches (jab, cross, hooks, uppercuts)
3. Observe the OpenCV window showing pose skeleton overlay and action labels

**Expected output:**
- Live camera feed with YOLO-detected pose skeleton drawn on the person
- Top-left: current predicted action (e.g., "JAB"), confidence percentage, and FPS
- Bar chart of action probabilities below the prediction
- Consecutive frame count (how many frames in a row the same action is detected)
- The prediction should change within 2-4 frames of starting a new punch

**Success criteria:**
- Skeleton tracking locks onto the user
- FPS is 15+ on the Jetson Orin NX (GPU inference)
- Punches are correctly classified at least 70% of the time
- Idle is correctly detected when standing still

### 5.2 Testing CV+IMU Fusion (Cell 4c)

**What it launches:** Two windows side by side plus background ROS nodes:
- Left: CV inference window (same as 4a)
- Right: Teensy Simulator GUI (real hardware or simulated pads)
- Background: `imu_node`, `punch_processor` ROS nodes

**Prerequisites:**
- All prerequisites from 4a PLUS:
- Teensy 4.1 connected via USB (`/dev/ttyACM0` at 115200 baud)
- Micro-ROS agent running (cell A1 setup complete)
- Arm calibration complete (cell A2) if testing motor execution

**What to do:**
1. Strike the physical pads while standing in camera view
2. Watch both windows simultaneously

**Expected output:**
- CV window: shows detected punch type from the camera
- Simulator window: shows IMU impact detection (pad, force level)
- Console output: fusion decisions showing CV+IMU matching
- When both CV and IMU agree (same punch within 200ms), the confirmed punch is published

**Success criteria:**
- IMU impacts appear within ~50ms of physical strikes
- CV predictions and IMU impacts are correctly matched by the fusion pipeline
- Pad-location constraints correctly reject impossible combinations (e.g., jab on left pad)
- Reclassification works for borderline cases

### 5.3 Testing the Simulator (Cell 3b)

**What it launches:** Teensy Simulator + V4 Arm Control GUI.

**Prerequisites:**
- Teensy connected
- Calibration complete (cell A2)

**What to do:**
1. Use simulator GUI buttons to trigger virtual pad strikes
2. Watch the robot arms move in response

**Expected output:**
- Simulator shows real-time IMU data from all 4 pads
- Pressing punch buttons in the simulator causes the physical robot arms to execute strikes
- The simulator correctly mirrors real pad strikes when the user physically hits the pads

**Success criteria:**
- Motor commands execute within 100ms of button press
- Strike positions match the calibrated library
- No motor faults or overcurrent errors

### 5.4 Testing the Full System (Cell 2a)

**What it launches:** Everything -- micro-ROS agent, V4 GUI, RealSense, all ROS nodes, BoxBunny GUI, and Teensy Simulator.

**Prerequisites:** All hardware connected and calibrated.

**What to do:**
1. Log in via the BoxBunny GUI (pattern lock or password)
2. Select a training mode (e.g., Techniques > Jab-Cross drill)
3. Configure rounds, work time, rest time
4. Start the session (centre pad or Start button)
5. Perform the drill -- throw the requested combo
6. Watch the GUI update with punch confirmations, combo progress, streaks

**Expected output:**
- Countdown (3-2-1-GO) with sound effects
- Live combo display highlighting the current expected punch
- Punch counter incrementing on each confirmed strike
- Round timer counting down
- Bell sound at round end
- Results screen with accuracy, punches, and best streak

**Success criteria:**
- Punch detection latency < 200ms (from strike to GUI confirmation)
- Combo detection correctly advances through the sequence
- Rest periods trigger correctly between rounds
- Session completes and results are saved to database

### 5.5 Testing the Phone Dashboard (Cell 5a)

**What it launches:** Dashboard server + SSH tunnel + QR code display.

**What to do:**
1. Scan the QR code with a phone
2. Log in with a demo account (e.g., alex / boxing123)
3. Browse the dashboard views: home, history, performance, achievements, settings, chat

**Expected output:**
- Dashboard home shows user stats, XP, rank, streak
- History shows past sessions
- Chat connects to the LLM and generates responses

**Success criteria:**
- QR login works and auto-logs the touchscreen GUI
- Remote control buttons successfully start sessions on the robot
- Height control buttons move the robot height motor
- Session history is consistent between touchscreen and phone

### 5.6 Testing the LLM (Cell 5b)

**What it launches:** `test_llm.py` which loads the Qwen 2.5-3B model.

**Prerequisites:**
- Model file: `models/llm/qwen2.5-3b-instruct-q4_k_m.gguf`
- `llama-cpp-python` installed

**Expected output:**
- Model loading message with time taken
- A sample coaching tip generated from a boxing prompt
- Generation time in seconds

**Success criteria:**
- Model loads without errors
- Response is coherent boxing advice
- Generation completes within 30 seconds on the Jetson

---

## 6. Test Data

### 6.1 Demo Users

Four demo users are pre-seeded into the database by the `build_setup.sh` script:

| Username | Password | Display Name | Level | User Type | Pattern |
|----------|----------|-------------|-------|-----------|---------|
| alex | boxing123 | Alex | intermediate | individual | `[0, 1, 2, 5, 8]` |
| maria | boxing123 | Maria | beginner | individual | Yes |
| jake | boxing123 | Jake | advanced | individual | Yes |
| sarah | coaching123 | Sarah | advanced | coach | Yes |

These users can be used for testing both the touchscreen GUI and phone dashboard. The pattern `[0, 1, 2, 5, 8]` corresponds to drawing an "L" shape (top-left, top-centre, top-right, middle-right, bottom-right) on the 3x3 pattern grid.

Demo user profiles are viewable via notebook cell **5d** (`view_user_dashboards.py`), which displays visual cards showing each user's XP, current rank, streak, session count, and personal records.

### 6.2 Sample Session Data Format

A training session record in the database contains these fields:

```json
{
    "session_id": "test_session_001",
    "mode": "training",
    "difficulty": "beginner",
    "started_at": "2026-03-29T10:00:00",
    "ended_at": "2026-03-29T10:15:00",
    "is_complete": true,
    "rounds_completed": 3,
    "rounds_total": 3,
    "work_time_sec": 180,
    "rest_time_sec": 60,
    "config": {
        "combo": "jab-cross-hook",
        "speed": "medium"
    },
    "summary": {
        "total_punches": 87,
        "punch_distribution": {
            "jab": 35,
            "cross": 30,
            "left_hook": 22
        },
        "defense_rate": 0.75,
        "avg_depth": 1.5
    }
}
```

Key fields:
- `mode`: One of `training`, `sparring`, `free`, `power`, `stamina`, `reaction`
- `difficulty`: One of `beginner`, `intermediate`, `advanced`
- `is_complete`: Whether all rounds were completed
- `work_time_sec` / `rest_time_sec`: Per-round timing
- `summary.total_punches`: Total confirmed punches across all rounds
- `summary.punch_distribution`: Breakdown by punch type
- `summary.defense_rate`: Fraction of robot punches successfully defended (sparring)
- `summary.avg_depth`: Average user depth from camera (metres)

### 6.3 Population Benchmarks

**File:** `data/benchmarks/population_norms.json`

Contains population-level statistics used for peer comparison on the dashboard. The benchmark system computes percentile rankings by comparing the user's metrics against age-group and gender-stratified norms.

Benchmark categories:
- **Reaction time:** Measured in milliseconds. Lower is better. Percentile computed inversely (faster = higher percentile).
- **Punch rate:** Punches per minute sustained over a session. Higher is better.
- **Power:** Peak force measured via IMU accelerometer magnitude (m/s^2). Higher is better.
- **Defense rate:** Percentage of robot punches successfully blocked or evaded. Higher is better.

Percentile tiers used in the dashboard display:
| Percentile | Tier Label | Color |
|------------|------------|-------|
| 90th+ | Elite | Orange (`text-bb-primary`) |
| 75th-89th | Above Average | Blue (`text-blue-400`) |
| 50th-74th | Average | Grey (`text-bb-text-secondary`) |
| 25th-49th | Below Average | Amber (`text-bb-warning`) |
| Below 25th | Developing | Red (`text-bb-danger`) |

The benchmark test (notebook cell 5e, `benchmark_test.py`) validates the percentile computation logic against known population norms.

### 6.4 Punch Sequences

**Directory:** `data/punch_sequences/`

Contains at least 6 JSON files defining motor waypoint sequences for the robot's punch movements. Each file is a JSON array of waypoints:

```json
[
    {"pos": [0.5, 0.0], "spd": [8.0, 8.0], "t": 0.0},
    {"pos": [-0.3, 0.5], "spd": [10.0, 10.0], "t": 0.15},
    {"pos": [0.0, 0.0], "spd": [6.0, 6.0], "t": 0.5}
]
```

Each waypoint has:
- `pos`: Motor positions (at least 2 values for the 2-DOF arm)
- `spd`: Motor speeds (optional)
- `t`: Timestamp in seconds (optional)

The integration test (test 28) validates that all sequence files exist, are valid JSON, contain at least 2 waypoints, and each waypoint has a `pos` field with at least 2 elements.
