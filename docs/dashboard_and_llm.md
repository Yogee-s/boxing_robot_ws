# BoxBunny Phone Dashboard and AI Coaching System

## 1. Dashboard Architecture

The BoxBunny phone dashboard is a web application designed for mobile-first access from the user's smartphone. It serves as the companion interface to the desktop GUI running on the Jetson Orin, providing session history, analytics, AI coaching chat, remote control, and gamification.

### 1.1 Technology Stack

| Layer | Technology | Location |
|-------|-----------|----------|
| **Backend** | FastAPI (Python, async) | `src/boxbunny_dashboard/boxbunny_dashboard/` |
| **Frontend** | Vue.js 3 SPA (Composition API) | `src/boxbunny_dashboard/frontend/` |
| **Database** | SQLite (WAL mode) | `data/boxbunny_main.db` + `data/users/<username>/boxbunny.db` |
| **Real-time** | WebSocket (FastAPI native) | `websocket.py` |
| **AI Model** | Qwen2.5-3B-Instruct Q4_K_M (GGUF) | `models/llm/` |
| **Hosting** | Jetson Orin local Wi-Fi AP | Port 8080, SSID "BoxBunny" |

### 1.2 System Diagram

```
    User's Phone                          Jetson Orin Nano
    (Browser)                             (BoxBunny system)
  +----------------+                   +---------------------------+
  |  Vue.js 3 SPA  | <--- HTTP/WS ---> |  FastAPI Backend          |
  |  (Tailwind CSS) |    Wi-Fi AP      |  (port 8080)              |
  +----------------+    "BoxBunny"      |    |                      |
                                        |    +-> SQLite DBs         |
                                        |    +-> ROS 2 Bridge       |
                                        |    +-> LLM (local GPU)    |
                                        +---------------------------+
                                               |
                                               v
                                        +-----------------+
                                        | Desktop GUI     |
                                        | (PyQt5, Jetson) |
                                        | reads /tmp/     |
                                        | command files   |
                                        +-----------------+
```

### 1.3 QR Code Login Flow

The Jetson runs a Wi-Fi access point with SSID "BoxBunny" (password in `config/boxbunny.yaml`). The desktop GUI displays a QR code containing the dashboard URL. The login flow:

1. User scans QR code on the desktop GUI screen with their phone.
2. Phone browser opens `http://<jetson-ip>:8080`.
3. Login view shows account picker (list of registered users).
4. User authenticates via password or pattern-lock.
5. On successful login, a shared file (`/tmp/boxbunny_gui_login.json`) is written containing user info.
6. Desktop GUI detects this file and auto-navigates to the user's home page.
7. A Bearer token is returned and stored in the phone's `localStorage` for subsequent API calls.

### 1.4 Real-time WebSocket

A WebSocket connection (`/ws`) provides live session data streaming:
- Session state changes (countdown, active, rest, complete)
- Live punch counts during training
- Coach tips as they are generated
- Session-complete events with summary data

The `ConnectionManager` in `websocket.py` maintains a buffer of state per user and supports role-based broadcasting (e.g., broadcast to all "individual" users or only to "coach" users).


## 2. API Endpoints

All endpoints require a Bearer token in the `Authorization` header (except `/api/auth/users`, `/api/auth/login`, `/api/auth/signup`, and `/api/auth/guest-claim`).

### 2.1 Authentication (`/api/auth/`)

Source: `api/auth.py`

| Method | Path | Purpose | Request Body | Response |
|--------|------|---------|-------------|----------|
| `GET` | `/api/auth/users` | List all registered users (for account picker) | -- | `[{id, username, display_name, level, user_type, has_pattern}]` |
| `POST` | `/api/auth/login` | Password login | `{username, password, device_type}` | `{token, user_id, username, display_name, user_type}` |
| `POST` | `/api/auth/pattern-login` | Pattern lock login | `{username, pattern: [int]}` | `{token, user_id, username, display_name, user_type}` |
| `POST` | `/api/auth/signup` | Create new account | `{username, password, display_name, user_type, level}` | `{token, user_id, ...}` (201) |
| `POST` | `/api/auth/pattern-verify` | Verify pattern lock (existing session) | `{user_id, pattern: [int]}` | `{token, ...}` |
| `POST` | `/api/auth/guest-claim` | Link guest session to new account | `{guest_token, username, password, display_name}` | `{token, user_id, ...}` |
| `GET` | `/api/auth/session` | Validate current token | -- | `{user_id, username, display_name, user_type, level, age, gender, height_cm, weight_kg, reach_cm, stance}` |
| `PUT` | `/api/auth/profile` | Update profile fields | `{display_name?, level?, avatar?, age?, gender?, height_cm?, weight_kg?, reach_cm?, stance?}` | `{status: "ok"}` |
| `POST` | `/api/auth/set-pattern` | Set/update pattern lock | `{user_id, pattern: [int, min 4]}` | `{status: "ok"}` |
| `DELETE` | `/api/auth/logout` | Invalidate session token | -- | 204 No Content |

### 2.2 Sessions (`/api/sessions/`)

Source: `api/sessions.py`

| Method | Path | Purpose | Query Params | Response |
|--------|------|---------|-------------|----------|
| `GET` | `/api/sessions/current` | Get latest live/active session | -- | `{active: bool, session: SessionSummary, live_state?: {...}}` |
| `GET` | `/api/sessions/history` | Paginated session history | `page`, `page_size`, `mode?` | `{sessions: [...], total, page, page_size}` |
| `GET` | `/api/sessions/trends` | Time-series analytics | `range` (7d/30d/90d/all) | `{punch_volume, reaction_time, defense_rate, power, stamina, personal_bests, weekly_summary, period_comparison, training_days}` |
| `GET` | `/api/sessions/{id}` | Full session detail | -- | `{session_id, mode, difficulty, ..., config, summary, events}` |
| `GET` | `/api/sessions/{id}/raw` | Raw sensor data | -- | `{cv_predictions, imu_strikes, direction_timeline, cv_prediction_summary, imu_strike_summary, direction_summary, experimental}` |

**Trends endpoint detail:**

The trends endpoint aggregates data across all sessions within the specified range. It returns:

- `punch_volume`: `[{date, value}]` -- punches per session over time
- `reaction_time`: `[{date, value}]` -- average reaction time per session
- `defense_rate`: `[{date, value}]` -- defense success rate over time
- `power`, `stamina`: similar time-series arrays
- `personal_bests`: `{fastest_reaction_ms, most_punches, best_defense_rate, max_power, best_stamina_ppm}`
- `weekly_summary`: `{sessions, total_punches, avg_score}`
- `period_comparison`: `{vs_last_week: "+25%", vs_last_month: "-10%"}`
- `training_days`: `[0, 2, 4]` -- day-of-week indices (Mon=0..Sun=6) for heat map

### 2.3 AI Coach Chat (`/api/chat/`)

Source: `api/chat.py`

| Method | Path | Purpose | Request Body | Response |
|--------|------|---------|-------------|----------|
| `GET` | `/api/chat/status` | Check LLM availability | -- | `{ready: bool, source: "ros"/"direct"/"none", warning?}` |
| `POST` | `/api/chat/message` | Send message to AI coach | `{message, context?: {...}}` | `{reply, timestamp, actions?: [{label, type, config}]}` |
| `GET` | `/api/chat/history` | Get chat history | `limit?` (default 50) | `[{role, content, timestamp}]` |

**Action cards:** When the LLM suggests a specific drill, it embeds `[DRILL:Name|combo=1-2|rounds=2|work=60s|speed=Medium]` tags in its response. The backend parses these into structured `TrainingAction` objects:

```json
{
    "reply": "Great question! For improving your jab-cross speed, I'd recommend...",
    "timestamp": "2026-04-03T10:30:00Z",
    "actions": [
        {
            "label": "Start: Jab-Cross Speed Drill",
            "type": "training",
            "config": {
                "combo_seq": "1-2",
                "Rounds": "3",
                "Work Time": "60s",
                "Speed": "Fast (1s)",
                "route": "training_session",
                "name": "Jab-Cross Speed Drill"
            }
        }
    ]
}
```

The frontend renders these as tappable buttons that auto-navigate to the training page with pre-filled configuration.

### 2.4 Remote Control (`/api/remote/`)

Source: `api/remote.py`

| Method | Path | Purpose | Request Body | Response |
|--------|------|---------|-------------|----------|
| `POST` | `/api/remote/command` | Send command to desktop GUI | `{action, config}` | `{success, message}` |
| `POST` | `/api/remote/height` | Height motor control | `{action: "up"/"down"/"stop"}` | `{success, message}` |
| `GET` | `/api/remote/presets` | Get user's presets for remote launch | -- | `[{name, tag, desc, route, combo, config, difficulty, accent}]` |

**Remote command mechanism:** Commands are written to a shared JSON file at `/tmp/boxbunny_gui_command.json`. The desktop GUI polls this file and executes the command. Supported actions:
- `start_training` -- starts a training session with the provided config
- `start_preset` -- loads and starts a saved preset
- `open_presets` -- navigates the GUI to the presets page
- `stop_session` -- ends the current session
- `navigate` -- navigates to a specific GUI page
- `height_adjust` -- height motor control

### 2.5 Presets (`/api/presets/`)

Source: `api/presets.py`

| Method | Path | Purpose | Request Body | Response |
|--------|------|---------|-------------|----------|
| `GET` | `/api/presets/` | List user's presets | -- | `[{id, name, description, preset_type, config_json, is_favorite, tags, use_count}]` |
| `POST` | `/api/presets/` | Create preset | `{name, preset_type, config_json, description?, tags?}` | Preset object (201) |
| `PUT` | `/api/presets/{id}` | Update preset | `{name?, config_json?, description?, tags?}` | `{id, updated: true}` |
| `DELETE` | `/api/presets/{id}` | Soft-delete (archive) preset | -- | `{id, archived: true}` |
| `POST` | `/api/presets/{id}/favorite` | Toggle favorite | -- | `{id, is_favorite: bool}` |

### 2.6 Gamification (`/api/gamification/`)

Source: `api/gamification.py`

| Method | Path | Purpose | Response |
|--------|------|---------|----------|
| `GET` | `/api/gamification/profile` | XP, rank, streak data | `{total_xp, current_rank, next_rank, xp_to_next_rank, current_streak, longest_streak, weekly_goal, weekly_progress}` |
| `GET` | `/api/gamification/achievements` | Unlocked achievements | `[{achievement_id, unlocked_at}]` |
| `GET` | `/api/gamification/leaderboard/{coaching_session_id}` | Leaderboard for a coaching session | `[{username, display_name, score, rank}]` |
| `GET` | `/api/gamification/benchmarks` | Percentile rankings vs population norms | `{benchmarks, user_stats, demographics}` |

**Rank progression:** Novice (0 XP) -> Contender (500) -> Fighter (1500) -> Warrior (4000) -> Champion (10000) -> Elite (25000).

### 2.7 Coach Mode (`/api/coach/`)

Source: `api/coach.py`

| Method | Path | Purpose | Request Body | Response |
|--------|------|---------|-------------|----------|
| `POST` | `/api/coach/load-config` | Load a preset for the station | `{preset_id}` | `{loaded, preset_id, config}` |
| `POST` | `/api/coach/start-station` | Start a coaching station session | `{name, preset_id?, config?}` | `{coaching_session_id, name, started}` |
| `POST` | `/api/coach/end-session` | End a coaching session | `{coaching_session_id}` | `{coaching_session_id, ended}` |
| `GET` | `/api/coach/live` | Get live participant stats | -- | `[{username, display_name, rounds_completed, score, connected}]` |

All coach endpoints require `user_type = "coach"` (returns 403 otherwise).

### 2.8 Export (`/api/export/`)

Source: `api/export.py`

| Method | Path | Purpose | Query Params | Response |
|--------|------|---------|-------------|----------|
| `GET` | `/api/export/session/{id}/csv` | CSV download for one session | -- | CSV file (streaming) |
| `GET` | `/api/export/session/{id}/pdf` | HTML report (printable as PDF) | -- | HTML file (streaming) |
| `GET` | `/api/export/range` | CSV of all sessions in a date range | `start_date`, `end_date`, `mode?` | CSV file (streaming) |


## 3. Frontend Views

The Vue.js 3 frontend is a single-page application with the following views:

| View | Route | Purpose |
|------|-------|---------|
| `LoginView` | `/login` | Account picker, password/pattern login, signup |
| `DashboardView` | `/` | Home screen: active session, quick stats, recent sessions, weekly progress |
| `HistoryView` | `/history` | Paginated session history with mode filters |
| `SessionDetailView` | `/session/:id` | Full session detail (see below) |
| `TrainingView` | `/training` | Remote session launcher, preset quick-starts |
| `PresetsView` | `/presets` | Manage saved presets (CRUD, favorites) |
| `ChatView` | `/chat` | AI Coach chat interface |
| `CoachView` | `/coach` | Coach-mode: station management, live participants |
| `PerformanceView` | `/performance` | Trend charts, personal bests, benchmarks |
| `AchievementsView` | `/achievements` | XP, rank, unlocked achievements, streaks |
| `SettingsView` | `/settings` | Profile editing, height control, system status |

### 3.1 SessionDetailView In Detail

The `SessionDetailView.vue` is the most data-rich view. It fetches session data from `GET /api/sessions/{id}` and optionally loads raw data from `GET /api/sessions/{id}/raw`. The view is structured as a scrollable card-based layout:

1. **Header Card** -- Mode badge (colour-coded: training=blue, sparring=red, free=green), difficulty badge, completion status badge, formatted date, session duration, and a large letter grade (A--F) in a coloured circle.

2. **Round Progress** -- Large "X / Y" text and a progress bar.

3. **Punch Distribution** -- Two charts via the `PunchChart` component:
   - Bar chart: punch counts per type (jab, cross, hooks, uppercuts)
   - Doughnut chart: proportional punch mix with colour coding

4. **Round-by-Round Breakdown** -- For each round: punch count, punches-per-minute, and an intensity bar (green >70%, yellow >40%, red <40%).

5. **Fatigue Curve** -- Line chart showing punches-per-minute across rounds. Reveals endurance drop-off.

6. **Defense Breakdown** -- 3-column grid with icons: blocks (shield icon), slips (arrow icon), dodges (move icon). Each shows count.

7. **Movement Trace** -- An HTML `<canvas>` element renders a 2D scatter-plot of the user's lateral (L/R) vs depth (F/B) position over the session. Data comes from the `movement_timeline_json` field (sampled at 2 Hz). Includes:
   - Animated playback (Play/Stop button) that traces the path chronologically
   - Start/end time labels

8. **Summary Table** -- Key-value pairs for all summary metrics.

9. **vs Your Average** -- Compares this session's metrics against the user's historical averages with coloured percentage change badges (green = improvement, red = decline).

10. **AI Coach Analysis** -- The LLM-generated post-session analysis text.

11. **XP Earned** -- Gamification reward display.

12. **Raw Sensor Data** (collapsed) -- CV prediction events, IMU pad strikes, position time. Loaded lazily on expand.

13. **Defense Analysis BETA** (collapsed) -- Experimental defense rate, average reaction time, per-type breakdown. Amber "BETA" badge warns user about data reliability.

14. **Action Row** -- Share, CSV, PDF export buttons.


## 4. AI Coach (LLM)

### 4.1 Model

The AI coaching system uses **Qwen2.5-3B-Instruct** quantized to **Q4_K_M** (GGUF format). This model was selected for its ability to run on the Jetson Orin's GPU with acceptable inference times while maintaining coaching-quality output.

| Property | Value |
|----------|-------|
| Model | Qwen2.5-3B-Instruct |
| Quantization | Q4_K_M (GGUF) |
| Context window | 2048 tokens |
| Max generation | 128 tokens |
| Temperature | 0.7 |
| GPU layers | -1 (all offloaded to GPU) |
| Inference library | llama-cpp-python |

### 4.2 Loading and Initialization

The LLM is hosted by `llm_node.py` (`src/boxbunny_core/boxbunny_core/llm_node.py`), a ROS 2 node that runs on the Jetson. Loading follows this sequence:

1. Node initialises with model path from config (`config/boxbunny.yaml` -> `llm.model_path`).
2. A one-shot timer fires 2 seconds after node creation to pre-load the model (`_preload_model`).
3. Pre-loading calls `_lazy_load_model()` which instantiates the `llama_cpp.Llama` object with full GPU offload.
4. If loading fails, a retry is scheduled in 30 seconds (`_schedule_retry`).
5. The model remains loaded for the lifetime of the node.

The dashboard API (`api/chat.py`) has a dual loading path:
- **Primary:** Calls the ROS service `/boxbunny/llm/generate` (which routes to `llm_node`).
- **Fallback:** If ROS is unavailable, loads the GGUF model directly in the dashboard process. A background thread checks ROS availability first and only loads the direct model if ROS is not responding within 3 seconds.

This ensures only one copy of the model is loaded in memory at any time.

### 4.3 System Prompt

The LLM operates with a detailed system prompt that establishes its persona and domain expertise:

```
You are BoxBunny AI Coach, an expert boxing trainer built into a
boxing training robot. Your knowledge is based on the AIBA Coaches
Manual and professional boxing coaching methodology.

Key traits:
- Deep knowledge of boxing technique: stance, footwork, all 6 basic
  punches (jab=1, cross=2, L hook=3, R hook=4, L uppercut=5,
  R uppercut=6), combinations, defenses (slip, block, bob-and-weave)
- Expert in training methodology: initiation, basic, specialization,
  and high-performance stages
- Knows European, Russian, American, and Cuban boxing styles
- Adjusts advice to the user's skill level (beginner/intermediate/advanced)
- Safety-focused: always prioritize proper form to prevent injury
- Keep tips SHORT (1-2 sentences max for real-time tips)
- Reference specific punch types and stats when available
- When the user asks for a drill, use format:
  [DRILL:Name|combo=1-2|rounds=2|work=60s|speed=Medium]
- When movement data is provided, analyze positioning and footwork:
  * Consistent lateral movement avoids being a stationary target
  * Favouring one side exposes you to attacks from that direction
  * Good depth management (depth_range > 0.3m) shows ring awareness
  * Too far back (avg_depth > 2.0m) limits power punch effectiveness
  * Crowding in (avg_depth < 0.8m) makes you vulnerable to uppercuts
```

### 4.4 Tip Generation During Sessions

During active training sessions, the LLM node generates coaching tips every 18 seconds:

```
Tip Tick (every 3s timer)
    |
    +-- Is session active?  No --> return
    |
    +-- Has 18s elapsed since last tip?  No --> return
    |
    +-- Determine tip type:
    |     - If >= 2 recent combo_missed events --> "correction" (trigger: low_accuracy)
    |     - If punches > 50 --> "encouragement" (trigger: milestone)
    |     - Otherwise --> "technique" (trigger: periodic)
    |
    +-- If LLM available:
    |     Generate with prompt: "Give a brief 1-sentence coaching tip. Mode: X, Punches: Y"
    |     Max tokens: 50
    |
    +-- If LLM unavailable or generation fails:
    |     Fall back to pre-written tips from config/fallback_tips.json
    |
    +-- Publish CoachTip message on /boxbunny/coach/tip
```

### 4.5 Post-Session Analysis

When a `SessionPunchSummary` message arrives (session complete), the LLM generates a brief analysis:

- Prompt includes total punches, defense rate, and punch distribution JSON.
- Max tokens: 80 (for robot screen display).
- Published as a CoachTip with priority=2 (high) and trigger="session_end".

### 4.6 Chat via Dashboard

The dashboard chat endpoint enriches the user's message with context:

1. Fetches the user's 5 most recent training sessions (mode, difficulty, rounds, punch count).
2. Builds a system prompt that includes the user's name, level, and training history.
3. Calls the LLM via the ROS service (or direct fallback).
4. Parses any `[DRILL:...]` tags from the response into structured action cards.
5. Persists both user message and assistant reply in the `session_events` table.

### 4.7 Reliability Fixes

Running a 3B-parameter LLM on edge hardware (Jetson Orin) introduces reliability challenges. The system implements a multi-layer timeout chain:

```
+--------------------------------------------------+
|  Frontend (Vue.js)                                |
|  Timeout: 20 seconds                             |
|  If exceeded: shows "AI Coach is thinking..."    |
|  then fallback message after another 5s          |
+--------------------------------------------------+
           |
           v
+--------------------------------------------------+
|  Backend (FastAPI - api/chat.py)                  |
|  Timeout: 15 seconds (ROS service call)          |
|  If exceeded: tries direct model call             |
|  If both fail: returns offline fallback message   |
+--------------------------------------------------+
           |
           v
+--------------------------------------------------+
|  LLM Node (llm_node.py)                          |
|  Timeout: 12 seconds (threaded inference)         |
|  If exceeded: returns empty string                |
|  Increments consecutive_failures counter          |
+--------------------------------------------------+
```

**Timeout chain:** Inference 12s < Backend 15s < Frontend 20s. This ensures that each layer times out before the layer above it, preventing cascading hangs.

**3-failure auto-reload:** After 3 consecutive failures (timeouts or errors), the LLM node:
1. Sets `_available = False`.
2. Deletes the model object (`self._llm = None`).
3. Attempts an immediate reload via `_lazy_load_model()`.
4. If reload fails, schedules a retry in 30 seconds.

**Graceful degradation:** If the LLM is completely unavailable:
- Real-time tips fall back to pre-written tips from `config/fallback_tips.json` (categorised by type: technique, encouragement, correction, suggestion).
- Chat returns a hardcoded message: "I'm currently running in offline mode. Connect the LLM service for personalized coaching feedback."
- Post-session analysis returns a simple text summary: "Session complete: X punches thrown."

### 4.8 LLM Status Monitoring

The `GET /api/chat/status` endpoint checks LLM health:

```json
// Healthy
{"ready": true, "source": "ros"}

// Direct model (ROS unavailable)
{"ready": true, "source": "direct"}

// Degraded (recent failures)
{"ready": true, "source": "ros", "warning": "LLM may be degraded -- recent inference failures detected"}

// Unavailable
{"ready": false, "source": "none"}
```


## 5. Height Control from Dashboard

### 5.1 Overview

The dashboard provides remote height adjustment for the robot's lead screw motor (MDDS10 driver). This allows the user to adjust the robot's height from their phone during setup, without needing to reach the desktop GUI.

### 5.2 Press-and-Hold UX

The frontend implements a press-and-hold interaction:

1. User presses and holds the UP or DOWN button.
2. On press: `POST /api/remote/height` with `action: "up"` or `action: "down"`.
3. The backend writes the command to `/tmp/boxbunny_gui_command.json` with `height_action: "manual_up"` or `"manual_down"`.
4. On release: `POST /api/remote/height` with `action: "stop"`.

### 5.3 Message Flow

```
Phone (press UP)
    |
    v
POST /api/remote/height {action: "up"}
    |
    v
Write /tmp/boxbunny_gui_command.json:
    {"action": "height_adjust", "config": {"height_action": "manual_up"}}
    |
    v
Desktop GUI polls file
    |
    v
Publishes HeightCommand on /boxbunny/robot/height
    {action: "manual_up"}
    |
    v
robot_node receives HeightCommand
    |
    v
Publishes String "UP:200" on /robot/height_cmd
    |
    v
Teensy firmware drives MDDS10 motor

Phone (release UP)
    |
    v
POST /api/remote/height {action: "stop"}
    |
    v
... same chain ... -> "STOP" -> motor stops
```

### 5.4 HeightCommand Message

```
float32 target_height_px       # target position in camera frame (auto-adjust)
float32 current_height_px      # current detected position
string action                  # "adjust", "calibrate", "manual_up", "manual_down", "stop"
```

### 5.5 Dead-Man Switch Safety

The backend tracks height control state via module-level variables:

```python
_height_active: bool = False       # currently moving?
_height_direction: str = "stop"    # current direction
_height_last_cmd: float = 0.0     # timestamp of last command
```

If no new command is received within 5 seconds, the system should auto-stop. This prevents the motor from running indefinitely if the phone disconnects mid-press. The V4 GUI's `HeightTab` implements a software ramp-down on stop: it reduces PWM in 6 steps over ~300ms to avoid jarring mechanical stops.

### 5.6 Auto-Adjustment

During the countdown phase before a round starts, the session manager uses `UserTracking.bbox_top_y` to auto-adjust height:

1. CV node publishes `UserTracking` with `bbox_top_y` (the top of the user's bounding box in pixels).
2. Session manager computes `target = 0.15 * 540` (15% of 540p frame height = 81 pixels from top).
3. If this is the first detection in the countdown and height hasn't been adjusted yet, publishes a `HeightCommand` with `action="adjust"`.
4. `robot_node` converts the pixel error to a PWM and direction: `error = current_height_px - target_height_px`, direction = UP if error > 0 (head too low, extend lead screw), PWM proportional to error magnitude (capped at 255).

This ensures the robot's punching height is calibrated to each individual user at the start of every session.
