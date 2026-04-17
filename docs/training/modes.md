# BoxBunny Training Modes

This document describes every training mode available in the BoxBunny system, including the underlying algorithms, configuration parameters, and detection pipelines that drive each mode.

---

## Table of Contents

1. [Techniques (Combo Drills)](#1-techniques-combo-drills)
2. [Sparring](#2-sparring)
3. [Free Training](#3-free-training)
4. [Performance Tests](#4-performance-tests)
5. [Coach Station](#5-coach-station)
6. [Punch Classification](#6-punch-classification)
7. [Defense Detection](#7-defense-detection)
8. [CV+IMU Sensor Fusion](#8-cvimu-sensor-fusion)

---

## 1. Techniques (Combo Drills)

**Source:** `drill_manager.py`

Techniques mode teaches users standardized boxing combinations through guided, progressive drills. The system validates combo execution in real-time and tracks mastery over time.

### Punch Code Reference

| Code | Punch           |
|------|-----------------|
| 1    | Jab             |
| 2    | Cross           |
| 3    | Left Hook       |
| 4    | Right Hook      |
| 5    | Left Uppercut   |
| 6    | Right Uppercut  |

### Combo Library

The system includes **50 progressive combinations** distributed across three difficulty levels:

- **Beginner** -- Fundamental combinations using jabs, crosses, and basic hooks. Typically 2-3 punches per combo. Examples: 1-2 (jab-cross), 1-1-2 (double jab-cross), 1-2-3 (jab-cross-left hook).
- **Intermediate** -- Longer sequences incorporating all six punch types. Typically 3-5 punches per combo. Introduces uppercuts and mixed combinations.
- **Advanced** -- Complex sequences of 4-8 punches requiring fluid transitions between all punch types, directional changes, and rapid sequencing.

### Combo Mastery System

Each combo is tracked individually in the `combo_progress` table with the following metrics:

- **Attempts:** Total number of times the user has attempted the combo.
- **Best Accuracy:** Highest accuracy score achieved (0.0 to 1.0).
- **Average Accuracy:** Running average across all attempts.
- **Mastered Flag:** Boolean flag set when the user achieves a sufficiently high accuracy consistently.
- **Last Attempted:** Timestamp of the most recent attempt.

### Real-Time Validation

`drill_manager.py` validates combo execution by:

1. Displaying the target combo sequence to the user.
2. Listening for incoming punch classifications (from the CV+IMU fusion pipeline).
3. Comparing each detected punch against the expected punch in the sequence.
4. Scoring accuracy as the ratio of correct punches to total expected punches.
5. Updating the combo_progress record after each attempt.

---

## 2. Sparring

**Source:** `sparring_engine.py`

Sparring mode pits the user against an AI opponent that controls the robot arm. The AI uses Markov chain transition matrices to select attack sequences, creating distinct fighting styles.

### AI Opponent Styles

There are **5 AI opponent styles**, each with a unique transition matrix that governs punch selection probabilities:

#### Boxer
- **Profile:** Technical, adaptive, balanced offense/defense.
- **Behavior:** Mixes all punch types with balanced probabilities. Tends to use combinations rather than single punches. Moderate attack frequency with clean technique transitions.

#### Brawler
- **Profile:** Aggressive, predictable, high-volume punches.
- **Behavior:** Heavily weighted toward power punches (hooks and crosses). Short gaps between attacks. High repetition of the same punch type. Predictable but overwhelming.

#### Counter-Puncher
- **Profile:** Reactive, patient, waits for user to punch then counters.
- **Behavior:** Low base attack frequency. Attack probability spikes immediately after detecting a user punch. Favors quick, precise counters (jabs and crosses). Long idle periods followed by sudden bursts.

#### Pressure
- **Profile:** Overwhelming, relentless, short rest periods.
- **Behavior:** Very high attack frequency. Minimal gaps between punches. Mixes all punch types but with shorter wind-up. Designed to test the user's defensive endurance.

#### Switch
- **Profile:** Mixed strategy, unpredictable style changes.
- **Behavior:** Periodically transitions between the other four styles. Transition timing is randomized. Creates an unpredictable opponent that requires the user to adapt continuously.

### Markov Chain Transition Matrices

Each style defines a 6x6 transition matrix where rows represent the current punch and columns represent the next punch. `sparring_engine.py` uses **weighted random selection** from the current row to determine the next attack. This creates naturalistic punch sequences that feel distinct for each style.

### Difficulty Scaling

Difficulty adjusts two primary parameters:

- **Attack Frequency:** How often the robot throws punches (time between attacks decreases with difficulty).
- **Speed:** The angular velocity of the robot arm during punches.

Robot punch speed tiers:

| Speed Tier | Angular Velocity |
|------------|-----------------|
| Slow       | 8 rad/s         |
| Medium     | 15 rad/s        |
| Fast       | 25 rad/s        |
| Max        | 30 rad/s        |

### Weakness Tracking

The sparring engine maintains a **weakness profile** per user (stored in `sparring_weakness_profile` table). For each of the 6 punch types, the system tracks:

- **Defense Success Rate:** How often the user successfully defends against that punch type.
- **Exposure Count:** How many times the user has faced that punch type.

The AI biases its attack selection toward punch types where the user has a lower defense success rate. This forces users to improve their weakest defensive areas over time.

### Sparring Data Recorded

After each sparring session, the following is persisted to `sparring_sessions`:

- Style and difficulty used
- Rounds completed
- User punch count
- Robot punches thrown and landed
- Overall defense rate
- Punch distribution (JSON breakdown by type)
- Defense breakdown (JSON: blocks, slips, dodges, hits, unknowns)

### Base Tracking (optional)

Sparring sessions can engage live **person-tracking** — the base rotates to keep the user centred while they move around. Toggle it from either:

- **Sparring Setup page** (pre-configure) — button disabled until BLE is connected.
- **Session page** (start/stop mid-session).

Both surfaces call `/boxbunny/ble/tracking_start` and `/boxbunny/ble/tracking_stop` on `ble_bridge_node`. The bridge listens to `/boxbunny/cv/person_direction` + `/boxbunny/cv/user_tracking` and emits `L:<rpm>/R:<rpm>/S` over BLE to the base Arduino. Near-range gating (default 0.8 m ± 0.15 m hysteresis) means background walkers do not drive the motor.

**Safety**: session end (STOP button, timer hitting zero, or navigating away) always calls `tracking_stop` — the motor can never be left running after a session. BLE disconnect mid-session also forces the local flag off.

Full detail: [../hardware/ble_bridge.md](../hardware/ble_bridge.md).

---

## 3. Free Training

**Source:** `free_training_engine.py`

Free Training is an open, reactive session with no structured drills or AI opponent. The robot responds to user pad hits with counter-strikes, creating a dynamic back-and-forth experience.

### Pad-to-Counter Mapping

When the user hits a pad, the robot responds with a contextually appropriate counter-strike:

| Pad Hit  | Robot Counter Options       |
|----------|-----------------------------|
| Centre   | Jab (1) or Cross (2)        |
| Left     | Left Hook (3) or Left Uppercut (5) |
| Right    | Right Hook (4) or Right Uppercut (6) |
| Head     | Jab (1) or Cross (2)        |

### Timing Parameters

- **Counter Cooldown:** 300ms minimum between counter-strikes. This prevents the robot from firing too rapidly and gives the user time to react.
- **Guard Return Timeout:** If the user is idle for 5 seconds, the robot returns to its guard position automatically.

### Session Flow

1. User begins hitting pads freely.
2. Each pad impact triggers a counter-strike from the robot (subject to cooldown).
3. User defends or evades the counter, then continues punching.
4. After 5 seconds of inactivity, the robot resets to guard.
5. Session continues until the user or timer ends it.

### Base Tracking (optional)

Free Training has the same **Start/Stop Tracking** button as sparring, on the free-training session page (next to Tap START). Identical pipeline: `/boxbunny/ble/tracking_start` → `ble_bridge_node` → BLE → base motor rotates to follow the user inside the near-range engagement zone. `tracking_stop` always fires on session end.

---

## 4. Performance Tests

Performance tests are standalone assessments that measure specific physical attributes. Each test produces structured results stored in dedicated tables.

### 4.1 Power Test

**Measures:** Maximum punch force via IMU accelerometer data.

**Procedure:**
1. User throws a series of punches at the pad.
2. The IMU accelerometer on the pad measures impact force for each punch.
3. The system records peak force and average force across all punches.

**Recorded Metrics:**
- `peak_force` -- Highest single-punch force reading.
- `avg_force` -- Mean force across all punches in the test.
- `punch_count` -- Total punches thrown during the test.
- `results_json` -- Per-punch breakdown with individual force readings.

### 4.2 Stamina Test

**Measures:** Sustained punch rate over a fixed duration.

**Procedure:**
1. User punches continuously for the test duration (default: 120 seconds).
2. The system counts total punches and calculates rate over time.

**Recorded Metrics:**
- `duration_sec` -- Test duration (default 120s).
- `total_punches` -- Raw punch count over the entire test.
- `punches_per_minute` -- Average sustained rate.
- `fatigue_index` -- Ratio of late-round punch rate to early-round punch rate. A value of 1.0 means no fatigue; lower values indicate the user slowed down significantly. Calculated as: (punch rate in last 30s) / (punch rate in first 30s).
- `results_json` -- Time-series punch rate data.

### 4.3 Reaction Test

**Measures:** Time from visual stimulus to punch detection.

**Procedure:**
1. The system presents **3 trials** of visual stimulus.
2. Detection uses the camera with YOLO pose detection to identify when the user throws a punch.
3. Reaction time is measured from stimulus presentation to detected punch initiation.

**Foreground-user gating**: triggers are rejected unless the primary tracked user is (a) detected, (b) within `_MAX_USER_DEPTH_M` (default 2.0 m) of the camera, and (c) roughly centred horizontally (middle 40% of the 960 px frame). Background walkers in a busy gym no longer trip the trigger. Tunable constants at the top of [`src/boxbunny_gui/boxbunny_gui/pages/performance/reaction_test_page.py`](../../src/boxbunny_gui/boxbunny_gui/pages/performance/reaction_test_page.py).

**Recorded Metrics:**
- `num_trials` -- Number of trials (default 10, but the standard test uses 3).
- `avg_reaction_ms` -- Mean reaction time across all trials.
- `best_reaction_ms` -- Fastest single trial.
- `worst_reaction_ms` -- Slowest single trial.
- `tier` -- Classification tier based on reaction time.
- `results_json` -- Per-trial reaction times.

**Reaction Tier Classification:**

| Tier        | Description                                    |
|-------------|------------------------------------------------|
| Lightning   | Exceptionally fast reactions                   |
| Fast        | Above-average reaction speed                   |
| Average     | Normal reaction time range                     |
| Developing  | Below-average, room for improvement            |

---

## 5. Coach Station

Coach Station enables group circuit training, where a coach manages multiple participants rotating through the boxing station.

### Setup Flow

1. **Coach logs in** with a coach-type account (e.g., sarah/coaching123).
2. **Sets student count** using - / + buttons on the dashboard (range: 1 to 30 participants).
3. **Selects a preset** configuration or taps "Start Station" to begin with defaults.

### Session Flow

1. Each participant sees a **GO button** on screen.
2. The participant either **hits the centre pad** or **taps GO** to start their turn.
3. A **countdown timer** runs for the configured work duration (`work_sec` from config).
4. When the timer expires, the system **auto-advances to the next participant** after a 3-second transition.
5. After the **last participant** completes their turn, the session **ends automatically**.
6. The coach can **end the session at any time** via the dashboard.

### Data Storage

- **coaching_sessions** (main DB): Records the coach, preset used, start/end times, total participants, and notes.
- **coaching_participants** (main DB): Records each participant's number, optional name, and session data as JSON.

---

## 6. Punch Classification

**Model:** `FusionVoxelPoseTransformerModel`

The punch classification system uses a computer vision model to classify user actions in real-time from the camera feed.

### Action Classes

The model recognizes **8 action classes**:

| Class            | Description                          |
|------------------|--------------------------------------|
| jab              | Straight lead-hand punch             |
| cross            | Straight rear-hand punch             |
| left_hook        | Hook punch with the left hand        |
| right_hook       | Hook punch with the right hand       |
| left_uppercut    | Uppercut with the left hand          |
| right_uppercut   | Uppercut with the right hand         |
| block            | Defensive blocking posture           |
| idle             | No active punch or defense           |

### Inference Performance

- **Frame Rate:** 30 fps
- **Optimization:** TensorRT with FP16 precision for real-time inference on Jetson hardware.

### Post-Processing Pipeline

Raw model outputs go through a multi-stage smoothing pipeline to reduce noise and false positives:

1. **EMA Smoothing (Exponential Moving Average)**
   - Alpha = 0.35
   - Smooths confidence scores over time to reduce frame-to-frame jitter.

2. **Hysteresis Thresholding**
   - Margin = 0.12
   - A new class must exceed the current class confidence by this margin to trigger a transition. Prevents rapid flickering between classes.

3. **State Machine**
   - Minimum hold = 3 frames
   - Sustain confidence = 0.78
   - A classification must be held for at least 3 consecutive frames before it is accepted. The confidence must remain above 0.78 to sustain the classification.

This three-stage pipeline ensures that only deliberate, sustained actions are registered, filtering out transient poses and classification noise.

---

## 7. Defense Detection

Defense detection is evaluated during sparring mode when the robot throws a punch at the user. The system classifies the user's defensive response into one of five categories.

### Defense Categories

| Defense Type | Detection Method                                                              |
|-------------|-------------------------------------------------------------------------------|
| **Block**   | CV model detects the `block` action class with confidence >= 0.3.            |
| **Slip**    | User tracking shows lateral displacement >= 40px OR depth displacement >= 0.15m. |
| **Dodge**   | Lateral displacement >= 20px OR depth displacement >= 0.08m (lower thresholds than slip). |
| **Hit**     | Robot arm IMU detects physical contact with the user.                         |
| **Unknown** | Robot arm missed the user, but no clear defensive movement was detected.      |

### Detection Priority

When the robot throws a punch, the system evaluates defense within a detection window:

1. Check for block (CV model).
2. Check for slip (high displacement).
3. Check for dodge (moderate displacement).
4. Check for hit (IMU contact).
5. Default to unknown if none of the above triggers.

### Defense Rate Calculation

```
defense_rate = (total_robot_attacks - hits) / total_robot_attacks
```

A defense rate of 1.0 means the user was never hit. A defense rate of 0.0 means every robot attack landed. This metric is stored per sparring session and tracked over time.

### Defense Breakdown

The full breakdown (counts per category) is stored as JSON in `defense_breakdown_json` on the `sparring_sessions` table, enabling detailed analysis of defensive tendencies.

---

## 8. CV+IMU Sensor Fusion

**Source:** `punch_processor.py`

The sensor fusion system combines computer vision predictions with IMU pad impact data to produce high-confidence punch classifications. This is the core pipeline that feeds into all training modes.

### Fusion Window

- **Matching Window:** 500ms
- When an IMU pad impact is detected, the system looks for a matching CV prediction within a 500ms window (before or after the impact).

### Pad-Constraint Filtering

Each punch type is only valid for certain pads. If the CV prediction does not match the hit pad, the punch is reclassified:

| Punch Type      | Valid Pads         |
|-----------------|--------------------|
| Jab (1)         | Centre only        |
| Cross (2)       | Centre only        |
| Left Hook (3)   | Left, Head         |
| Right Hook (4)  | Right, Head        |
| Left Uppercut (5)  | Left, Head      |
| Right Uppercut (6) | Right, Head     |

### Reclassification Logic

If the primary CV prediction is invalid for the detected pad:

1. Check the **secondary prediction** (second-highest confidence class).
2. If the secondary prediction is valid for the pad AND its confidence >= 0.25, use it instead.
3. Otherwise, assign a default punch type based on the pad alone.

### IMU Debounce

- **Debounce Period:** 150ms per pad.
- After a pad registers an impact, it ignores subsequent impacts for 150ms to prevent double-counting from a single punch.

### CV-Only Punches

When the CV model detects a punch but no IMU impact is registered (e.g., shadow boxing or missed punches):

- Requires >= **3 consecutive frames** of the same classification.
- Requires >= **0.7 confidence** throughout.
- These are accepted as valid punches but with lower certainty.

### IMU-Only Punches

When a pad impact is detected but no matching CV prediction exists:

- The punch is accepted with a **default confidence of 0.3**.
- The punch type is inferred from the pad location using the default mapping.

### Fusion Priority

1. **CV + IMU match:** Highest confidence. Both systems agree.
2. **CV + IMU with reclassification:** CV secondary prediction matches pad.
3. **CV-only:** Sustained high-confidence detection without pad impact.
4. **IMU-only:** Pad impact without CV confirmation.

This layered approach ensures robust punch detection across all scenarios while minimizing false positives and false negatives.
