# BoxBunny Data Collection and Analytics System

## 1. Data Sources

The BoxBunny training robot collects data from multiple independent sensor streams that are fused, aggregated, and persisted across the system. Every data source is published as a ROS 2 message on a named topic (configured in `config/ros_topics.yaml`). The table below catalogues each source.

| # | Data Source | ROS Topic | Message Type | Rate | Key Fields |
|---|-------------|-----------|-------------|------|------------|
| 1 | **CV Punch Detection** | `/boxbunny/cv/detection` | `PunchDetection` | ~30 Hz (camera frame rate) | `timestamp`, `punch_type` (jab/cross/left_hook/right_hook/left_uppercut/right_uppercut/block/idle), `confidence` (0.0--1.0), `raw_class`, `consecutive_frames` |
| 2 | **IMU Pad Impact** | `/boxbunny/imu/pad/impact` | `PadImpact` | Event-driven | `timestamp`, `pad` (left/centre/right/head), `level` (light/medium/hard), `accel_magnitude` (m/s^2) |
| 3 | **IMU Punch Event** | `/boxbunny/imu/punch_event` | `PunchEvent` | Event-driven (training mode only) | `timestamp`, `pad`, `level`, `force_normalized` (0.33/0.66/1.0), `accel_magnitude` |
| 4 | **Confirmed Punch (Fusion)** | `/boxbunny/punch/confirmed` | `ConfirmedPunch` | Event-driven | `timestamp`, `punch_type`, `pad`, `level`, `force_normalized`, `cv_confidence`, `imu_confirmed` (bool), `cv_confirmed` (bool), `accel_magnitude` |
| 5 | **Defense Event** | `/boxbunny/punch/defense` | `DefenseEvent` | Event-driven | `timestamp`, `arm`, `robot_punch_code` (1--6), `struck` (bool), `defense_type` (block/slip/dodge/unknown) |
| 6 | **User Tracking** | `/boxbunny/cv/user_tracking` | `UserTracking` | ~30 Hz | `timestamp`, `bbox_centre_x`, `bbox_centre_y`, `bbox_top_y`, `bbox_width`, `bbox_height`, `depth` (metres), `lateral_displacement` (pixels), `depth_displacement` (metres), `user_detected` (bool) |
| 7 | **Person Direction** | `/boxbunny/cv/person_direction` | `std_msgs/String` | ~30 Hz | `data`: "left", "right", or "centre" |
| 8 | **Robot Command** | `/boxbunny/robot/command` | `RobotCommand` | Event-driven | `command_type` (punch/set_speed), `punch_code` (1--6), `speed` (slow/medium/fast) |
| 9 | **Session State** | `/boxbunny/session/state` | `SessionState` | State-change events | `state` (idle/countdown/active/rest/complete), `mode` (training/sparring/free/power/stamina/reaction), `username` |
| 10 | **Coach Tip** | `/boxbunny/coach/tip` | `CoachTip` | ~Every 18 s during session | `timestamp`, `tip_text`, `tip_type` (technique/encouragement/correction/suggestion), `trigger`, `priority` |

### Data Source Relationship Diagram

```
   RealSense D435i                 Teensy MCU (4x IMU)
   (Color + Depth)                 (pad impacts, arm strikes)
        |                                  |
        v                                  v
   +-----------+                    +-----------+
   |  cv_node  |                    | imu_node  |
   +-----------+                    +-----------+
     |       |                        |       |
     v       v                        v       v
  detection  user_tracking      punch_event  arm_event
     \         \                    /         /
      \         \                  /         /
       +-----> +-------------------+ <------+
               | punch_processor   |
               | (CV + IMU Fusion) |
               +-------------------+
                  |           |
                  v           v
          confirmed_punch  defense_event
                  |           |
                  v           v
              +-------------------+
              | session_manager   |  <--- person_direction
              | (data collector)  |  <--- robot_command
              +-------------------+
                       |
                       v
              SessionPunchSummary
              (published at session end)
                       |
                       v
              +-------------------+
              | Dashboard API     |
              | (db/manager.py)   |
              +-------------------+
                       |
                       v
              SQLite per-user DB
```


## 2. Collection Strategy

All per-session data collection occurs in `session_manager.py` (`src/boxbunny_core/boxbunny_core/session_manager.py`). The node manages the session lifecycle (idle -> countdown -> active -> rest -> complete) and only accumulates data during the `active` state for punch/defense events. Some tracking (CV predictions, IMU strikes, direction) is collected whenever a session exists regardless of sub-state.

### 2.1 Session Data Model

The `SessionData` dataclass is the in-memory accumulator for all data during a session:

```python
@dataclass
class SessionData:
    session_id: str = ""
    mode: str = "training"               # training/sparring/free/power/stamina/reaction
    difficulty: str = "beginner"
    username: str = "guest"
    config: Dict = field(default_factory=dict)
    rounds: List[RoundData] = field(default_factory=list)
    current_round: int = 0
    total_rounds: int = 3
    work_time_s: int = 180
    rest_time_s: int = 60
    started_at: float = 0.0

    # --- Accumulated counters ---
    total_punches: int = 0
    punch_distribution: Dict[str, int]    # punch_type -> count
    force_distribution: Dict[str, float]  # punch_type -> sum of force_normalized
    force_counts: Dict[str, int]          # punch_type -> count (for averaging)
    pad_distribution: Dict[str, int]      # pad -> count

    # --- Robot attack data ---
    robot_punches_thrown: int = 0
    robot_punches_landed: int = 0
    defense_breakdown: Dict[str, int]     # defense_type -> count

    # --- Movement tracking ---
    depth_samples: List[float]            # from UserTracking.depth
    lateral_samples: List[float]          # abs(UserTracking.lateral_displacement)

    # --- Raw sensor logs ---
    cv_prediction_events: List[Dict]      # grouped CV events (see 2.3)
    imu_strikes: List[Dict]              # raw IMU punch events
    direction_changes: List[Dict]         # person direction timeline
    defense_reactions: List[Dict]         # experimental reaction timing
```

Each round has its own `RoundData` with per-punch records:

```python
@dataclass
class RoundData:
    punches: List[Dict]          # per-punch fusion records
    defense_events: List[Dict]   # per-defense records
    start_time: float = 0.0
    end_time: float = 0.0
```

### 2.2 Per-Punch Fusion Records

When a `ConfirmedPunch` message arrives during an active round, the session manager stores an enriched record:

```python
{
    "type": "jab",                    # punch_type from fusion
    "pad": "centre",                  # which physical pad was hit
    "force": 0.66,                    # force_normalized (0.33/0.66/1.0)
    "cv_conf": 0.87,                  # CV model confidence
    "ts": 1711900000.123,             # UNIX timestamp
    "imu_confirmed": True,            # did an IMU pad impact match?
    "cv_confirmed": True,             # did a CV detection match?
    "accel": 35.2,                    # raw accelerometer magnitude (m/s^2)
}
```

These records are appended to `rounds[-1].punches` and also used to update the running distribution counters (`punch_distribution`, `force_distribution`, `pad_distribution`).

**Fusion confirmation flags:** A punch can be confirmed by either sensor alone or both:
- `imu_confirmed=True, cv_confirmed=True` -- both sensors agree (highest confidence)
- `imu_confirmed=True, cv_confirmed=False` -- pad detected impact but CV did not classify an action (type inferred from pad location using `PadLocation.VALID_PUNCHES`)
- `imu_confirmed=False, cv_confirmed=True` -- CV classified a punch but no pad impact (requires >= 3 consecutive frames at >= 70% confidence)

### 2.3 CV Prediction Event Grouping

Raw CV detections arrive at ~30 Hz, which would flood the session data with thousands of per-frame records. The session manager uses an **event grouping algorithm** to collapse consecutive frames of the same prediction type into a single event.

**Algorithm:**

1. Each `PunchDetection` message is filtered: only non-idle, non-block predictions above 50% confidence are tracked.
2. If the prediction type matches the current ongoing event, the frame is accumulated (frame count incremented, confidence summed, peak tracked).
3. If the prediction type changes (or drops below threshold), the current event is **closed** and a new one begins.

When closed, the event record contains:

```python
{
    "ts": 1711900000.5,           # timestamp of first frame in the event
    "type": "jab",                 # the classified action
    "peak_conf": 0.923,           # highest single-frame confidence
    "avg_conf": 0.871,            # mean confidence across all frames
    "frame_count": 8,             # number of consecutive frames
}
```

Events are capped at 500 per session to prevent unbounded memory growth. The grouping state is maintained via private fields on the `SessionManager`:

```
_cv_current_type     -- the action type currently being tracked
_cv_current_start    -- timestamp of first frame
_cv_current_frames   -- number of frames accumulated
_cv_current_conf_sum -- running sum of confidences (for avg)
_cv_current_peak_conf -- highest confidence seen
```

### 2.4 Raw IMU Strikes

Every `PunchEvent` from the IMU node is stored as a raw record regardless of whether it matches a CV prediction:

```python
{
    "ts": 1711900001.2,
    "pad": "left",                # which pad was hit
    "level": "hard",              # Teensy-classified force level
    "accel": 42.7,                # raw accelerometer magnitude (m/s^2)
}
```

### 2.5 Direction Changes

The person direction topic publishes "left", "right", or "centre" at camera frame rate. The session manager only records transitions -- when the direction changes, a segment is closed:

```python
{
    "ts": 1711900002.0,           # when this direction segment started
    "direction": "left",          # the direction during this segment
    "duration_s": 3.45,           # how long they stayed in this direction
}
```

### 2.6 Defense Reaction Timing (Experimental)

When the robot throws a punch (`RobotCommand` with `command_type="punch"`), the session manager records the timestamp and punch code. When a subsequent `DefenseEvent` arrives, a reaction record is created:

```python
{
    "ts": 1711900003.0,           # when the robot attacked
    "punch_code": "3",            # what the robot threw (left hook)
    "defense_detected": "slip",   # how the user defended (or "none")
    "reaction_time_ms": 280,      # ms between robot attack and defense detection
                                   # null if the user was struck
}
```

This is marked **experimental** because the reaction timing depends on the latency between the robot command and the CV/IMU detection, which introduces measurement error.


## 3. Session Summary

When a session ends (either all rounds complete or the `EndSession` service is called), the `_build_summary()` method computes comprehensive statistics from the accumulated data.

### 3.1 Summary JSON Structure

The complete summary dictionary:

```json
{
    "session_id": "abc123def456",
    "mode": "sparring",
    "difficulty": "medium",

    "total_punches": 127,
    "punch_distribution": {
        "jab": 42,
        "cross": 35,
        "left_hook": 22,
        "right_hook": 18,
        "left_uppercut": 5,
        "right_uppercut": 5
    },
    "force_distribution": {
        "jab": 0.554,
        "cross": 0.712
    },
    "pad_distribution": {
        "centre": 77,
        "left": 22,
        "right": 18,
        "head": 10
    },

    "robot_punches_thrown": 45,
    "robot_punches_landed": 12,
    "defense_rate": 0.733,
    "defense_breakdown": {
        "block": 18,
        "slip": 8,
        "dodge": 7,
        "hit": 12
    },

    "avg_depth": 1.45,
    "depth_range": 0.82,
    "lateral_movement": 234.5,

    "rounds_completed": 3,
    "duration_sec": 648.2,

    "cv_prediction_summary": {
        "jab": { "events": 38, "total_frames": 152, "avg_conf": 0.834 },
        "cross": { "events": 30, "total_frames": 120, "avg_conf": 0.791 }
    },
    "imu_strike_summary": {
        "centre": 77,
        "left": 22,
        "right": 18,
        "head": 10
    },
    "imu_strikes_total": 127,

    "direction_summary": {
        "left": 82.3,
        "right": 95.1,
        "centre": 470.8
    },

    "experimental": {
        "defense_reactions": [ ... ],
        "defense_rate": 0.650,
        "defense_breakdown": {
            "slip": 8,
            "block": 18,
            "dodge": 7,
            "none": 12
        },
        "avg_reaction_time_ms": 312
    }
}
```

### 3.2 Reliable Data vs Experimental Data

The summary is split into two reliability tiers:

**Reliable (Primary) Data:**
- `total_punches`, `punch_distribution`, `force_distribution`, `pad_distribution` -- derived from fusion-confirmed punches. Each punch has been verified by at least one sensor (CV or IMU) with the other providing supporting evidence.
- `defense_rate`, `defense_breakdown` -- based on IMU arm-strike contact detection combined with CV block detection.
- `avg_depth`, `depth_range`, `lateral_movement` -- from RealSense depth sensor (hardware-measured).
- `cv_prediction_summary` -- aggregated from grouped CV events. Useful for understanding the model's behaviour.
- `imu_strike_summary` -- raw pad hit counts from hardware.

**Experimental Data** (in the `experimental` key):
- `defense_reactions` -- individual reaction timing records. Accuracy depends on the latency between: (1) robot command publication, (2) physical arm reaching the user, (3) CV/IMU detecting the defense.
- `avg_reaction_time_ms` -- mean reaction time derived from the above. Treated as an estimate, not a precise measurement.
- The experimental `defense_rate` may differ from the primary one because it uses different methodology (reaction record matching vs. simple struck/not-struck counts).

### 3.3 Computation Details

**Average force per punch type:** For each type, `force_distribution[type] / force_counts[type]` gives the mean normalised force (0.33 = light, 0.66 = medium, 1.0 = hard).

**Defense rate:** `(robot_punches_thrown - robot_punches_landed) / robot_punches_thrown`. A value of 0.733 means the user successfully defended 73.3% of robot attacks.

**Direction summary:** Total seconds spent in each direction (left, right, centre). Useful for identifying if the user favours one side.

**CV prediction summary:** Per-type aggregation of the grouped events -- total events, total frames, and weighted average confidence (weighted by frame count, not by event count).


## 4. Database Persistence

### 4.1 Architecture

The database layer (`src/boxbunny_dashboard/boxbunny_dashboard/db/manager.py`) uses SQLite with a two-tier design:

```
data/
  boxbunny_main.db              <-- shared: users, auth_sessions, presets
  users/
    alex/
      boxbunny.db               <-- per-user: training_sessions, events, tests, XP
    maria/
      boxbunny.db
    jake/
      boxbunny.db
```

All connections use WAL journal mode for concurrent read/write safety, and foreign keys are enforced.

### 4.2 Main Database (boxbunny_main.db)

Stores shared data:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `users` | User accounts | `id`, `username`, `password_hash`, `display_name`, `user_type` (individual/coach), `level`, `age`, `gender`, `height_cm`, `weight_kg`, `reach_cm`, `stance`, `pattern_hash`, `settings_json`, `avatar`, `last_login` |
| `auth_sessions` | Login tokens | `user_id`, `session_token`, `device_type`, `expires_at`, `is_active` |
| `guest_sessions` | Temporary guest access | `guest_session_token`, `expires_at`, `claimed_by_user_id`, `claimed_at` |
| `presets` | Saved training configurations | `user_id`, `name`, `description`, `preset_type`, `config_json`, `is_favorite`, `tags`, `use_count`, `created_at`, `updated_at` |

**Password hashing:** Supports both SHA-256 with salt (`sha256:<salt>:<hash>`) and bcrypt. New passwords use SHA-256 for GUI compatibility. Pattern locks are hashed identically (the pattern sequence `[1,4,7,8,9]` is joined as `"1-4-7-8-9"` then hashed).

### 4.3 Per-User Database (users/<username>/boxbunny.db)

Each user has an isolated database:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `training_sessions` | Session records | `session_id` (PK), `mode`, `difficulty`, `started_at`, `ended_at`, `is_complete`, `rounds_completed`, `rounds_total`, `work_time_sec`, `rest_time_sec`, `config_json`, `summary_json` |
| `session_events` | Timestamped events | `session_id` (FK), `timestamp`, `event_type`, `data_json` |
| `power_tests` | Power test results | `peak_force`, `avg_force`, `punch_count`, `results_json` |
| `stamina_tests` | Stamina test results | `duration_sec`, `total_punches`, `punches_per_minute`, `fatigue_index`, `results_json` |
| `reaction_tests` | Reaction time results | `num_trials`, `avg_reaction_ms`, `best_reaction_ms`, `worst_reaction_ms`, `tier`, `results_json` |
| `user_xp` | XP and rank | `total_xp`, `current_rank`, `rank_history_json` |
| `streaks` | Training streaks | `current_streak`, `longest_streak`, `last_training_date`, `weekly_goal`, `weekly_progress` |
| `achievements` | Unlocked achievements | `achievement_id` (UNIQUE), `unlocked_at` |
| `personal_records` | Personal bests | `record_type` (UNIQUE), `value`, `achieved_at`, `previous_value` |

### 4.4 Session Persistence Flow

```
Session End (service call or all rounds complete)
        |
        v
_build_summary()  -->  summary dict
        |
        v
_publish_session_summary()  -->  SessionPunchSummary msg on ROS
        |                                |
        v                                v
Dashboard WebSocket                 LLM node (post-session analysis)
(live update to phone)
        |
        v
Dashboard API receives summary
        |
        v
db.save_training_session(username, {
    "session_id": ...,
    "mode": ...,
    "summary": summary_dict,   -->  stored as summary_json TEXT
    ...
})
        |
        v
db.save_session_event(username, session_id, ts, "raw_cv_predictions", cv_events)
db.save_session_event(username, session_id, ts, "raw_imu_strikes", imu_strikes)
db.save_session_event(username, session_id, ts, "direction_timeline", direction_changes)
```

The `summary_json` column in `training_sessions` stores the entire summary dictionary as a JSON string. The `session_events` table stores the raw data arrays as separate event rows, enabling the dashboard to fetch them on-demand.

### 4.5 Querying Session Data

The `DatabaseManager` provides typed access methods:

```python
# Get paginated history
sessions = db.get_session_history("alex", limit=50, mode="sparring")

# Get full detail including events
detail = db.get_session_detail("alex", "abc123def456")
# detail["summary_json"]  -> the summary dict
# detail["events"]        -> list of {event_type, data_json, timestamp}

# Get parsed events
events = db.get_session_events("alex", "abc123def456")
# returns [{event_type: "raw_cv_predictions", data: [...]}, ...]
```


## 5. Dashboard Data Display

The phone dashboard (`SessionDetailView.vue`) presents session data in three tiers, with progressive disclosure to avoid overwhelming the user.

### 5.1 Tier 1: Primary Fusion Data (Always Shown)

The main session detail view always displays:

- **Header Card:** Mode badge (Training/Sparring/Free), difficulty badge, completion status, date, duration, letter grade (A/B/C/D/F based on session score).
- **Round Progress:** Progress bar showing rounds completed vs total.
- **Punch Distribution:** Interactive bar chart and doughnut chart showing punch type breakdown (from `punch_distribution`).
- **Round-by-Round Breakdown:** Per-round punch counts, punches-per-minute, and intensity bars.
- **Fatigue Curve:** Line chart of punches-per-minute across rounds (shows endurance drop-off).
- **Defense Breakdown:** Grid showing blocks, slips, dodges with icons and counts (from `defense_breakdown`).
- **Movement Trace:** Canvas-rendered 2D visualisation of lateral (L/R) vs depth (F/B) movement over the session, with animated playback.
- **Summary Table:** Key-value pairs: total punches, duration, defense rate, average depth, lateral movement.
- **vs Your Average:** Comparison of this session's metrics against the user's historical average, with percentage change indicators.
- **AI Coach Analysis:** LLM-generated post-session feedback (1--2 sentences).
- **XP Earned:** Gamification reward for the session.

### 5.2 Tier 2: Raw Sensor Data (Toggle)

Hidden by default behind a "Raw Sensor Data" collapsible section. When expanded, it loads from the `GET /api/sessions/{session_id}/raw` endpoint and shows:

- **CV Prediction Events:** Grid cards for each punch type showing event count and average confidence.
- **IMU Pad Strikes:** Grid showing per-pad hit counts (left, centre, right, head).
- **Position Time:** Grid showing seconds spent in each direction (left, right, centre).

### 5.3 Tier 3: Experimental Data (Toggle with Beta Badge)

Hidden behind a "Defense Analysis" section with a prominent amber "BETA" badge. When expanded:

- **Disclaimer text:** "Based on CV detection -- may not capture all defensive movements"
- **Defense Rate:** Percentage computed from experimental reaction records.
- **Avg Reaction Time:** In milliseconds.
- **Defense Breakdown:** Per-type counts (block, slip, dodge, none).

### 5.4 Export Options

The session detail view provides three export actions:

- **Share:** Opens a modal with a summary card that can be copied to clipboard.
- **CSV:** Downloads a CSV file via `GET /api/export/session/{session_id}/csv`.
- **PDF:** Downloads an HTML report (printable as PDF) via `GET /api/export/session/{session_id}/pdf`.

### 5.5 Data Tier Summary

```
+------------------------------------------------+
| TIER 1: Primary (Always Visible)               |
| - Punch distribution (bar + donut charts)      |
| - Round breakdown, fatigue curve               |
| - Defense breakdown (blocks/slips/dodges)       |
| - Movement trace (animated canvas)             |
| - AI Coach analysis                            |
| - XP earned, vs-average comparison             |
+------------------------------------------------+
                    |
           [Show Raw Data] toggle
                    |
+------------------------------------------------+
| TIER 2: Raw Sensor Data                        |
| - CV prediction event summary                  |
| - IMU pad strike counts                        |
| - Direction time distribution                  |
+------------------------------------------------+
                    |
        [Defense Analysis BETA] toggle
                    |
+------------------------------------------------+
| TIER 3: Experimental                           |
| - Defense rate (experimental calculation)       |
| - Average reaction time (ms)                   |
| - Per-type defense breakdown                   |
| - "May not capture all defensive movements"    |
+------------------------------------------------+
```
