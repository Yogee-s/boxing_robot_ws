# BoxBunny GUI and User Experience Documentation

This document provides a comprehensive reference for the BoxBunny graphical user interface (both the touchscreen GUI running on the Jetson Orin NX and the phone-based Vue.js dashboard), the UX design decisions made specifically for a boxing training context, the communication architecture between the two interfaces, and the calibration workflow required for first-time setup.

---

## 1. Touchscreen GUI Overview

### 1.1 Technology Stack and Window Configuration

The touchscreen GUI is built with **PySide6** (the official Qt 6 Python bindings) and runs as a frameless, fixed-size window at **1024x600 pixels**, matching the resolution of the 7-inch touchscreen mounted on the Jetson Orin NX.

**Entry point:** `src/boxbunny_gui/boxbunny_gui/app.py` -- the `BoxBunnyApp` class.

Key architectural decisions:

- **No singletons or globals:** Every subsystem (ROS bridge, page router, sound manager, IMU navigation handler) is constructed in `BoxBunnyApp.__init__()` and passed by reference to page widgets via constructor injection.
- **QStackedWidget page routing:** All pages are registered as named widgets in a `QStackedWidget`. The `PageRouter` class (`nav/router.py`) manages navigation with a history stack, forward/back navigation, and lifecycle hooks (`on_enter(**kwargs)` and `on_leave()`).
- **Graceful offline mode:** If `rclpy` (ROS 2 Python client) is not available at import time, the `GuiBridge` runs in offline mode. All signals still exist but never fire, allowing the GUI to be tested standalone without any ROS infrastructure.

```
BoxBunnyApp
    |-- QApplication (1024x600, fixed size)
    |-- QMainWindow (frameless backup with title bar)
    |-- QStackedWidget (central widget, holds all pages)
    |-- GuiBridge (ROS 2 <-> Qt signal bridge, background QThread)
    |-- PageRouter (string-based navigation with history)
    |-- SoundManager (preloaded WAV effects via QSoundEffect)
    |-- ImuNavHandler (IMU pad -> navigation signals)
    |-- DevOverlay (developer hardware visualizer, toggled from settings)
    |-- DebugDetectionPanel (F12 full-page CV debug view)
    |-- PresetOverlay (head-pad triggered quick-launch)
```

### 1.2 Dark Theme Design System

All visual constants are centralised in a single file: `src/boxbunny_gui/boxbunny_gui/theme.py`. No inline hex colors appear anywhere else in the codebase.

#### 1.2.1 Color Palette

The palette is a **dark navy base with a warm orange accent**, chosen for high contrast in gym lighting and strong visibility through peripheral vision during training:

| Token | Hex | Role |
|-------|-----|------|
| `Color.BG` | `#0B0F14` | Deep navy-black background |
| `Color.BG_GRADIENT_TOP` | `#0E1319` | Slightly lighter for gradient tops |
| `Color.BG_GRADIENT_BTM` | `#080B10` | Slightly darker for gradient bottoms |
| `Color.SURFACE` | `#131920` | Cards, panels, elevated surfaces |
| `Color.SURFACE_LIGHT` | `#1A2029` | Raised interactive elements |
| `Color.SURFACE_HOVER` | `#222B37` | Hover state for surfaces |
| `Color.SURFACE_GLASS` | `rgba(19,25,32,0.85)` | Glassmorphism-style translucent panels |
| `Color.PRIMARY` | `#FF6B35` | Warm orange -- the primary brand accent |
| `Color.PRIMARY_DARK` | `#E85E2C` | Hover state for primary buttons |
| `Color.PRIMARY_PRESSED` | `#CC5025` | Pressed state for primary buttons |
| `Color.PRIMARY_LIGHT` | `#FF8C5E` | Highlights, glow effects |
| `Color.PRIMARY_MUTED` | `#FF6B3518` | Very subtle orange tint (18 = 9% alpha) |
| `Color.PRIMARY_GLOW` | `#FF6B3530` | Subtle glow background (30 = 19% alpha) |
| `Color.WARNING` | `#FFAB40` | Warm amber for warnings |
| `Color.DANGER` | `#FF5C5C` | Vibrant coral-red for errors, destructive actions |
| `Color.SUCCESS` | `#56D364` | Fresh green for confirmations |
| `Color.INFO` | `#58A6FF` | Soft blue for informational elements |
| `Color.PURPLE` | `#BC8CFF` | Lavender accent for performance metrics |
| `Color.TEXT` | `#E6EDF3` | Bright off-white primary text |
| `Color.TEXT_SECONDARY` | `#8B949E` | Muted grey for secondary labels |
| `Color.TEXT_DISABLED` | `#484F58` | Very dim text for disabled states |
| `Color.TEXT_ACCENT` | `#FFB088` | Warm accent text for titles and highlights |
| `Color.BORDER` | `#1C222A` | Default border |
| `Color.BORDER_LIGHT` | `#2A3340` | Lighter border for elevated elements |
| `Color.BORDER_ACCENT` | `#FF6B3540` | Subtle orange border with transparency |

**Punch Type Colors** (used consistently across the combo display widget, dev overlay, charts, and debug panel):

| Punch | Color Token | Hex |
|-------|-------------|-----|
| Jab | `Color.JAB` | `#58A6FF` (blue) |
| Cross | `Color.CROSS` | `#FF5C5C` (coral) |
| Left Hook | `Color.L_HOOK` | `#56D364` (green) |
| Right Hook | `Color.R_HOOK` | `#FFAB40` (amber) |
| Left Uppercut | `Color.L_UPPERCUT` | `#BC8CFF` (purple) |
| Right Uppercut | `Color.R_UPPERCUT` | `#F8E45C` (yellow) |
| Block | `Color.BLOCK` | `#8B949E` (grey) |
| Idle | `Color.IDLE` | `#484F58` (dark grey) |

#### 1.2.2 Typography

The GUI uses the **Inter** variable font family, loaded from `assets/fonts/InterVariable.ttf` at application startup. The `font()` helper in `theme.py` creates `QFont` instances:

```python
def font(size: int = 16, bold: bool = False) -> QFont:
    f = QFont("Inter", max(1, size))
    if bold:
        f.setBold(True)
    return f
```

The global stylesheet sets the fallback chain: `"Inter", "Segoe UI", "Helvetica Neue", sans-serif` with a base size of 15px and font-weight 500.

Standardised text sizes (all in `Size` class):

| Token | Pixels | Usage |
|-------|--------|-------|
| `TEXT_TIMER_XL` | 96 | Giant countdown timers |
| `TEXT_TIMER` | 80 | Large timer display |
| `TEXT_TIMER_SM` | 60 | Smaller timer variant |
| `TEXT_HEADER` | 28 | Page headers |
| `TEXT_SUBHEADER` | 22 | Section headers |
| `TEXT_BODY` | 16 | Body text |
| `TEXT_LABEL` | 14 | Form labels, secondary text |
| `TEXT_CAPTION` | 12 | Captions, badges |
| `TEXT_OVERLINE` | 10 | Overline text, tiny labels |

#### 1.2.3 Dimensions and Layout Constants

| Token | Value | Usage |
|-------|-------|-------|
| `MIN_TOUCH` | 60px | Minimum touch target size (for gloved hands) |
| `SCREEN_W` | 1024px | Screen width |
| `SCREEN_H` | 600px | Screen height |
| `SIDEBAR_W` | 200px | Sidebar width (not currently used -- all full-screen pages) |
| `TOP_BAR_H` | 50px | Top navigation bar height |
| `BUTTON_H` | 60px | Standard button height |
| `BUTTON_H_SM` | 44px | Small button height |
| `BUTTON_H_LG` | 64px | Large button height |
| `BUTTON_W_SM` | 120px | Small button width |
| `BUTTON_W_MD` | 300px | Medium button width |
| `BUTTON_W_LG` | 500px | Large button width (hero CTAs) |
| `LAYOUT_MARGINS` | (60, 40, 60, 40) | Standard page margins (L, T, R, B) |
| `RADIUS` | 12px | Default corner radius |
| `RADIUS_SM` | 8px | Small radius |
| `RADIUS_LG` | 16px | Large radius for cards |
| `RADIUS_XL` | 20px | Extra-large radius |
| `SPACING` | 20px | Standard spacing |
| `SPACING_SM` | 10px | Small spacing |
| `SPACING_XS` | 6px | Extra-small spacing |
| `SPACING_LG` | 24px | Large spacing |
| `SPACING_XL` | 32px | Extra-large spacing |
| `SHADOW_BLUR` | 20px | Default shadow blur |
| `SHADOW_BLUR_LG` | 32px | Large shadow blur |
| `ACCENT_BAR_W` | 4px | Left accent bar width on mode cards |
| `RING_THICKNESS` | 6px | Progress ring stroke width |

#### 1.2.4 Pre-Built Button Styles

The theme module provides factory functions and pre-built stylesheets for common button variants. Each uses the `button_style()` generator which produces a complete QPushButton stylesheet with normal, hover, pressed, and disabled states:

- `PRIMARY_BTN` -- Solid orange background, white text
- `DANGER_BTN` -- Coral-red background
- `WARNING_BTN` -- Amber background
- `SUCCESS_BTN` -- Green background
- `SURFACE_BTN` -- Dark surface background with border
- `GHOST_BTN` -- Transparent background, appears on hover
- `INFO_BTN` -- Surface with blue text and border

Additional style generators:
- `hero_btn_style()` -- Large hero CTA with rounded corners and bold weight
- `secondary_btn_style()` -- Bordered button for Log In/Sign Up
- `mode_card_style(accent)` / `mode_card_style_v2(accent)` -- Premium cards with colored left accent bars and glow hover effects
- `config_tile_style()` / `config_tile_style_v2(accent)` -- Tappable config tiles with top accent bars
- `tab_btn_style(active)` -- Filter/tab pill buttons with active states
- `pill_toggle_style(active)` -- Segmented control pill toggles
- `glass_card_style()` -- Glassmorphism-inspired translucent cards
- `elevated_card_style(accent)` -- Cards with hover elevation
- `accent_frame_style(accent)` -- Frames with left accent bar (stat cards)

#### 1.2.5 Global Stylesheet

The `GLOBAL_STYLESHEET` constant (applied via `QApplication.setStyleSheet()`) provides consistent styling for all Qt widget types including `QWidget`, `QLabel`, `QFrame`, `QScrollArea`, `QScrollBar`, `QCheckBox`, `QSlider`, `QLineEdit`, and `QProgressBar`. Scrollbar styling is minimal (5px wide with rounded handles).

### 1.3 Page Map -- All GUI Pages

The GUI contains **26 registered page routes** organised into 8 folders. Each page is a `QWidget` subclass with optional `on_enter(**kwargs)` and `on_leave()` lifecycle methods.

#### 1.3.1 Auth Pages (`pages/auth/`)

| Route Name | Class | File | Description |
|------------|-------|------|-------------|
| `auth` | `StartupPage` | `startup_page.py` | Landing screen. Gradient "BoxBunny" branding with animated glow CTA button. Three entry points: "Quick Start" (guest assessment), "Log In" (account picker), "Sign Up". Bottom-right "Phone Login" button opens QR popup for phone dashboard pairing. |
| `account_picker` | `AccountPickerPage` | `account_picker_page.py` | Grid of user account cards with gradient circle avatars, display names, level badges, and search/filter. Loads users from the database (falls back to demo users). Clicking a card navigates to pattern lock. |
| `pattern_lock` | `PatternLockPage` | `pattern_lock_page.py` | 3x3 dot grid pattern lock for authentication. Supports drag-to-draw gesture. Toggle button switches to password mode with animated height collapse/expand transitions. Verifies against bcrypt-hashed pattern in database. |
| `signup` | `SignupPage` | `signup_page.py` | Account creation with username, display name, and pattern/password toggle. Pattern grid is default; password is optional fallback. Creates user in database and navigates to guest assessment on success. |
| `guest_assessment` | `GuestAssessmentPage` | `guest_assessment_page.py` | Two-step proficiency check. Step 1: 2-column grid of 6 questions (boxing experience, punch knowledge, combo ability, sparring history, fitness, equipment experience) with 3-option buttons. Step 2: shows suggested level (Beginner/Intermediate/Advanced) with override buttons and rich text descriptions. |

#### 1.3.2 Home Pages (`pages/home/`)

| Route Name | Class | File | Description |
|------------|-------|------|-------------|
| `home` | `HomeIndividualPage` | `home_individual.py` | Main dashboard for logged-in users. Top bar shows avatar (from DB, with gradient fallback), welcome message, settings and close buttons. 2x2 grid of premium mode cards: Techniques (combo drills), Sparring (vs robot AI), Free Training (open session), Performance (power/speed tests). Each card has colored left accent bar and rich text description. Optional history button at bottom. |
| `home_guest` | `HomeGuestPage` | `home_guest.py` | Simplified home for guest/new users. Same mode card layout but adapted for users who completed the quick assessment without an account. |
| `home_coach` | `HomeCoachPage` | `home_coach.py` | Coach-specific home page with station management capabilities, user monitoring, and session oversight features. |

#### 1.3.3 Training Pages (`pages/training/`)

| Route Name | Class | File | Description |
|------------|-------|------|-------------|
| `training_select` | `ComboSelectPage` | `combo_select_page.py` | Scrollable list of available combo drills organised by difficulty tier. Each combo shows the punch sequence (e.g., "1-2-3" = Jab-Cross-Left Hook). Selecting a combo navigates to training config. |
| `training_config` | `TrainingConfigPage` | `training_config_page.py` | Pre-session configuration: rounds, work/rest time, speed (tappable tiles that cycle through values). Shows selected combo and difficulty. Centre pad IMU strike starts the session (`imu_start()` method). |
| `training_session` | `TrainingSessionPage` | `training_session_page.py` | Active training with live timer, combo display (highlighted sequence showing current expected punch), punch counter, streak tracker, and coach tip bar. Connects to ROS bridge for real-time punch confirmation and drill progress updates. |
| `training_rest` | `TrainingRestPage` | `training_rest_page.py` | Between-round rest screen with countdown timer, round progress indicator, and stats from the completed round. |
| `training_results` | `TrainingResultsPage` | `training_results_page.py` | Post-session results: total punches, accuracy percentage, best streak, round-by-round breakdown, LLM coach analysis, and option to retry or return home. |
| `self_select` | `SelfSelectPage` | `self_select_page.py` | Self-directed training mode selector where users can choose their own drill parameters without following a predefined combo. |

#### 1.3.4 Sparring Pages (`pages/sparring/`)

| Route Name | Class | File | Description |
|------------|-------|------|-------------|
| `sparring_select` | `SparringConfigPage` | `sparring_config_page.py` | Sparring session configuration: robot style (boxer/brawler/counter-puncher), difficulty, rounds, work/rest time, and robot punch speed. |
| `sparring_session` | `SparringSessionPage` | `sparring_session_page.py` | Active sparring with live defence tracking. The robot throws punches via motor commands; the user must block or dodge. Shows real-time defence rate, robot punch patterns, and user punch counters. |
| `sparring_results` | `SparringResultsPage` | `sparring_results_page.py` | Post-sparring results: defence rate, punches thrown/landed (both user and robot), round statistics, and AI coaching feedback. |

#### 1.3.5 Performance Pages (`pages/performance/`)

| Route Name | Class | File | Description |
|------------|-------|------|-------------|
| `performance` | `PerformanceMenuPage` | `performance_menu_page.py` | Menu for the three performance test modes: Power Test, Stamina Test, and Reaction Test. Each shown as a premium card with accent color. |
| `power_test` | `PowerTestPage` | `power_test_page.py` | Power measurement test. Measures peak force on pad strikes using IMU accelerometer magnitude. Shows real-time force meter and records personal bests. |
| `stamina_test` | `StaminaTestPage` | `stamina_test_page.py` | Endurance test measuring punch output over time. Tracks punches per minute, total volume, and consistency decay. |
| `reaction_test` | `ReactionTestPage` | `reaction_test_page.py` | Visual stimulus reaction time test using the D435i camera. 3-2-1 countdown, green stimulus flash, measures time to first movement via skeleton keypoint displacement. Shows tier rating (Elite/Fast/Average/Slow) and rolling history. |

#### 1.3.6 History, Presets, Coach, Settings Pages

| Route Name | Class | File | Description |
|------------|-------|------|-------------|
| `history` | `HistoryPage` | `pages/history/history_page.py` | Scrollable list of past training sessions loaded from database. Shows date, mode, duration, punch count, and score for each session. |
| `presets` | `PresetsPage` | `pages/presets/presets_page.py` | User-created training presets with favourite toggles and quick-launch capability. Synced with the phone dashboard. |
| `coach` | `StationPage` | `pages/coach/station_page.py` | Coach station management -- for gym coaches who manage multiple BoxBunny stations and monitor multiple users' sessions. |
| `settings` | `SettingsPage` | `pages/settings/settings_page.py` | System settings: developer mode toggle (shows the hardware visualizer overlay), IMU navigation enable/disable, volume control, and account management. |

---

## 2. UX Design Decisions for the Boxing Context

### 2.1 Pattern Lock Authentication

**Problem:** Users wear 12-16oz boxing gloves during training. Typing a username and password on a touchscreen is impractical.

**Solution:** Pattern lock as the primary authentication method.

The pattern lock (`pages/auth/pattern_lock_page.py`) implements a custom-painted 3x3 dot grid using `QPainter`. Users draw a pattern by dragging through dots (minimum 3 connected). The implementation:

1. **_PatternGrid widget:** Custom `QWidget` with `mousePressEvent`, `mouseMoveEvent`, and `mouseReleaseEvent` handlers. Hit detection uses a generous radius (`_DOT_RADIUS + 10 = 32 pixels`) to accommodate gloved finger presses.
2. **Visual feedback:** Active dots are drawn at full size (22px radius) in the primary orange color. Connecting lines are drawn semi-transparently (alpha 180) between selected dots. A "trailing line" follows the finger from the last selected dot to the current touch position during drag.
3. **Animated mode switching:** When toggling between pattern and password modes, `QPropertyAnimation` on `maximumHeight` provides smooth collapse/expand transitions (250ms, InOutCubic easing).
4. **Security:** Patterns are stored as bcrypt hashes in the SQLite database. The `db_helper.verify_pattern()` function hashes the drawn pattern and compares against the stored hash. A hardcoded fallback (`[0, 1, 2, 5, 8]` / `boxing123`) exists for development when no database is available.

The pattern lock works identically on both the touchscreen GUI and the phone dashboard (Vue PatternLock component), enabling the same authentication experience across devices.

### 2.2 IMU Pad Navigation

**Problem:** During training, users should be able to navigate menus without removing gloves to touch the screen.

**Solution:** Four IMU-equipped pads on the robot are mapped to navigation commands:

| Pad | Command | Action |
|-----|---------|--------|
| Left pad | `prev` | Navigate to previous item / scroll left |
| Right pad | `next` | Navigate to next item / scroll right |
| Centre pad | `enter` | Confirm / select / start session |
| Head pad | `back` | Go back / toggle preset overlay |

**Implementation chain:**

1. **Teensy microcontroller** reads MPU6050 accelerometers on each pad and sends `NavCommand` messages via micro-ROS.
2. **`imu_node.py`** classifies impact events and publishes `NavCommand` to the `/boxbunny/nav/command` topic.
3. **`GuiBridge._RosWorker`** receives the message in a background `QThread` and emits a Qt signal.
4. **`BoxBunnyApp._on_nav_command()`** routes the command:
   - If the `PresetOverlay` is visible, all commands go to it (left/right cycle presets, centre confirms, head dismisses).
   - If on the home page, head pad toggles the preset overlay.
   - Centre pad triggers `imu_start()` on config/test pages to start a session.
5. **`ImuNavHandler`** (`nav/imu_nav_handler.py`) translates commands to Qt signals (`navigate_prev`, `navigate_next`, `select`, `go_back`). Navigation is automatically **disabled** during countdown and active session states to prevent accidental menu changes from punch impacts.

**Keyboard fallback:** For desktop development, the `KeyboardNavFilter` event filter maps arrow keys (Left=prev, Right=next), Enter (select), and Escape (back) to the same signals.

### 2.3 Large Touch Targets and Simple Layouts

The `Size.MIN_TOUCH = 60px` constant enforces a minimum 60-pixel touch target size across all interactive elements. This is significantly larger than the standard 44px mobile guideline because:

1. **Boxing gloves reduce precision:** A gloved fist has a contact area roughly 3-4x larger than a bare fingertip.
2. **Sweat and fatigue:** During rest periods, users need to interact quickly without precise aiming.
3. **Peripheral interaction:** Users may glance at the screen briefly between rounds rather than giving it full attention.

Design patterns enforcing this:
- All buttons use `BUTTON_H = 60px` minimum height
- Mode cards on the home page are large (full-width with 16px padding)
- The "Quick Start" hero button on the startup page is 500x76 pixels
- Account picker cards are 210x155 pixels each
- Pattern lock dots have a 22px visual radius but a 32px hit test radius

### 2.4 Sound Feedback System

**File:** `src/boxbunny_gui/boxbunny_gui/sound.py`

The `SoundManager` preloads all WAV files into `QSoundEffect` objects at startup for zero-latency playback. A priority system prevents low-priority UI sounds from masking high-priority training cues.

**Priority tiers** (higher number = higher priority):

| Priority | Tier | Example Sounds |
|----------|------|----------------|
| 5 | Stimulus | `stimulus.wav` -- reaction test visual cue |
| 4 | Bell | `bell_start.wav`, `bell_end.wav` -- round start/end |
| 3 | Countdown | `countdown_tick.wav`, `countdown_go.wav` -- 3-2-1-GO |
| 2 | Feedback | `hit_confirm.wav`, `combo_complete.wav`, `miss.wav` |
| 1 | UI | `btn_press.wav`, `nav_tick.wav`, `error.wav` |

**All 18 sound files** in `src/boxbunny_gui/assets/sounds/`:

| File | Purpose |
|------|---------|
| `stimulus.wav` | Reaction test green flash audio cue |
| `bell_start.wav` | Round start bell |
| `bell_end.wav` | Round end bell |
| `countdown_tick.wav` | Each countdown number (3, 2, 1) |
| `countdown_go.wav` | "GO!" cue at countdown end |
| `countdown_beep.wav` | Alternative countdown beep |
| `hit_confirm.wav` | Successful punch detection confirmation |
| `combo_complete.wav` | Full combo completed correctly |
| `miss.wav` | Missed or incorrect punch |
| `btn_press.wav` | Button press feedback |
| `button_click.wav` | Alternative button click |
| `nav_tick.wav` | IMU navigation tick |
| `error.wav` | Error/invalid action |
| `impact.wav` | General impact sound |
| `coach_notification.wav` | AI coach tip notification |
| `reaction_stimulus.wav` | Alternative reaction stimulus |
| `rest_start.wav` | Rest period start |
| `session_complete.wav` | Full session completed |

The master volume is configurable (default 0.8) and individual sounds can be toggled on/off via `set_enabled()`.

### 2.5 Visual Feedback -- Developer Overlay and Debug Panel

#### 2.5.1 Developer Overlay (`widgets/dev_overlay.py`)

A 320x220px semi-transparent overlay widget positioned at the bottom-right of the screen. When enabled via Settings > Developer Mode, it shows a real-time visual model of the physical hardware:

```
            [HEAD PAD]
 [L ARM]  [LEFT] [CENTRE] [RIGHT]  [R ARM]
```

Each pad and arm is drawn as a rounded rectangle using `QPainter`. On impact:
- **Pads** flash with a force-level color for 400ms: green (light), orange (medium), red (hard)
- **Arms** flash red (struck) or green (blocked) for 400ms

Below the hardware model, a colored bar shows the current CV prediction (punch type in uppercase) with confidence percentage. A force legend at the bottom maps colors to force levels.

The overlay refreshes at 50ms intervals via `QTimer` and automatically clears expired flash animations.

#### 2.5.2 Debug Detection Panel (`widgets/debug_panel.py`)

A full-page panel toggled by pressing **F12**. Shows CV detection metadata as styled text (no camera feed to avoid GPU contention with the VoxelFlow inference pipeline). Displays:
- Current predicted punch type with confidence
- Frame persistence count (consecutive frames with same prediction)
- IMU match status
- Inference FPS
- Scrolling punch log (last 30 confirmed punches)

Each punch type is color-coded using the canonical punch colors from `theme.py`.

### 2.6 Gesture Control Option (`gesture_node.py`)

**File:** `src/boxbunny_core/boxbunny_core/gesture_node.py`

An optional ROS node that uses **MediaPipe Hands** to detect hand gestures from the D435i camera feed and convert them to `NavCommand` messages. This provides hands-free navigation without requiring physical pad strikes.

Supported gestures:
- **Open palm** (all 5 fingers extended): mapped to `enter` (confirm/select)
- **Thumbs up** (thumb extended, fingers closed): mapped to `enter`
- **Peace sign** (index + middle extended, others closed): mapped to `back`

The gesture classifier (`classify_gesture()`) checks finger extension by comparing fingertip Y-coordinates to PIP joint Y-coordinates (lower Y = higher on screen = extended). Thumb extension uses X-coordinate distance from wrist.

The node is **disabled by default** to avoid interfering with the boxing CV inference pipeline. It subscribes to the RealSense color image topic and publishes `NavCommand` messages with a debounce window to prevent rapid-fire commands from sustained gestures.

---

## 3. Phone Dashboard UX

### 3.1 Purpose and Motivation

The phone dashboard serves three primary use cases:

1. **Remote control:** Start training sessions, adjust robot height, and select presets from a phone rather than walking to the touchscreen.
2. **Post-session analysis:** Review detailed session history, punch distribution charts, performance trends, and peer benchmarks from the couch after training.
3. **Social and gamification:** Track XP, rank progression, achievements, streaks, and weekly goals -- features that benefit from casual browsing outside of active training.

### 3.2 Technology Stack

- **Frontend:** Vue 3 with Composition API (`<script setup>`), Tailwind CSS for styling, Pinia for state management
- **Backend:** FastAPI Python server (`tools/dashboard_server.py`)
- **Database:** SQLite via `DatabaseManager` (`src/boxbunny_dashboard/db/manager.py`)
- **Real-time:** WebSocket connection for live session updates
- **Hosting:** Localhost:8080 with optional SSH tunnel via `localhost.run` for public HTTPS URL

### 3.3 QR Code Login Flow

The phone connection flow works as follows:

1. User taps "Phone Login" on the touchscreen `StartupPage`.
2. A `_QrPopup` dialog opens (500x520px, frameless).
3. Background thread:
   a. Starts the FastAPI dashboard server if not already running (`tools/dashboard_server.py`).
   b. Opens an SSH tunnel to `localhost.run` (up to 3 retry attempts with 20-second timeout each).
   c. When the public HTTPS URL is ready, generates a QR code using the `qrcode` Python library and displays it.
4. User scans the QR code with their phone's camera.
5. Phone opens the URL, showing the Vue dashboard login page.
6. User logs in (password or pattern lock) on the phone.
7. The backend writes a JSON file to `/tmp/boxbunny_gui_login.json` containing the authenticated username.
8. The popup polls this file every 1000ms. On detection, it reads the username, deletes the file, closes the dialog, and auto-navigates the touchscreen GUI to the user's home page.

This flow allows a single QR scan to authenticate both the touchscreen (auto-login) and the phone (manual login), linking both devices to the same user session.

### 3.4 Authentication

The phone dashboard supports two authentication methods, matching the touchscreen:

1. **Password login:** Username + password form. The backend verifies the password against a bcrypt hash stored in SQLite.
2. **Pattern lock login:** Username selection from a dropdown of registered accounts + a 200x200px canvas pattern grid (`components/PatternLock.vue`). The drawn pattern (array of dot indices) is sent to the API, which verifies against the stored bcrypt hash.

New users can sign up directly from the phone with a display name, username, and pattern or password. After signup, the user is directed to a 6-question proficiency assessment (identical questions to the touchscreen `GuestAssessmentPage`), then a result screen where they can accept or override the suggested skill level.

### 3.5 Vue Dashboard Views

All views are in `src/boxbunny_dashboard/frontend/src/views/`:

#### 3.5.1 LoginView (`LoginView.vue`)

Three-step flow within a single view:

1. **Auth step:** Tabbed login/signup form. Login supports password and pattern lock modes. Account dropdown auto-loads from `/api/users` and shows display name, level, and pattern availability. Signup includes display name, username, and pattern/password toggle.
2. **Survey step:** Proficiency questionnaire (6 questions) with animated card transitions and progress bar. Left/right slide animation between questions.
3. **Result step:** Shows suggested level with description and 3-button override grid, then navigates to dashboard.

#### 3.5.2 DashboardView (`DashboardView.vue`)

The main home screen. Rich, data-dense layout with staggered entrance animations:

- **Header:** Welcome message, streak flame display, and rank badge
- **User profile card:** Avatar (SVG from 8 options or initial letter), display name, physical stats (height/weight/reach), edit link to settings
- **XP progress bar:** Current rank, XP total, progress to next rank with labeled thresholds
- **Weekly goal + Streak:** Side-by-side cards. Weekly goal shows progress bar (sessions/goal), streak shows current and longest with flame animation
- **Training heat map:** 7-day grid (Mon-Sun) with filled dots for training days, session and punch totals
- **Quick stats:** 2x2 grid of StatCard components showing total sessions, total punches, best defence rate, and best reaction time. Each with optional trend percentage arrows.
- **Peer comparison:** Horizontal percentile bars for reaction time, punch rate, power, and defense -- compared against demographic peers from `data/benchmarks/population_norms.json`. Tiers: Elite (90th+), Above Average (75th+), Average (50th+), Below Average (25th+), Developing.
- **Recent session:** Latest session card with mode badge, round count, and score
- **AI Coach tip:** Context-aware tip based on streak, goal progress, or session count. Link to full chat view.
- **Quick actions:** 3-column grid linking to Achievements, Presets, and AI Coach (or Coach Mode for coach users)
- **Remote start:** Horizontal scrollable preset cards that send `start_preset` commands to the robot via the API

Data is loaded in parallel on mount via `Promise.all()` for session history, gamification data, and current session state. Additional data (profile, benchmarks, trends, presets) loads non-blocking in the background.

#### 3.5.3 HistoryView (`HistoryView.vue`)

Paginated session history with filter tabs (All, Training, Sparring, Performance). Each session is rendered as a `SessionCard` component showing mode, date, rounds, and key metrics. Skeleton loading states and empty state messages. Infinite scroll via "Load More" button.

#### 3.5.4 SessionDetailView (`SessionDetailView.vue`)

Detailed view of a single training session. Shows:
- Mode badge, difficulty, completion status
- Full date/time and duration
- User context (level, height, weight) at time of session
- Round-by-round breakdown with punch distribution
- Punch chart (using PunchChart component)
- AI analysis if available

#### 3.5.5 ChatView (`ChatView.vue`)

AI coach chat interface with:
- LLM connection status indicator (Ready/Connecting/Loading model)
- Quick action chips: "Analyze my last session", "Suggest a training plan", "What should I work on?", "Tell me about proper jab technique"
- Scrollable message history (user messages right-aligned, AI responses left-aligned)
- Context banner explaining what data the AI coach can access (skill level, sessions, goals, performance)
- Message persistence via Pinia chat store

The LLM runs locally as a Qwen 2.5-3B model via `llama-cpp-python`, accessed through the `GenerateLlm` ROS service or the dashboard API.

#### 3.5.6 SettingsView (`SettingsView.vue`)

Comprehensive settings with multiple sections:

- **Profile:** Avatar picker (8 SVG options: boxer, tiger, eagle, wolf, flame, lightning, shield, crown) plus "Use My Initials" option. Display name editor.
- **Weekly training goal:** +/- buttons to set sessions per week (1-7)
- **Robot height control:** Hold-to-move UP/DOWN buttons that send continuous height commands at 10Hz while pressed. Uses `@touchstart`/`@touchend` for mobile and `@mousedown`/`@mouseup` for desktop. Sends `manual_up`, `manual_down`, or `stop` commands to `/api/remote/height`.
- **Security:** Tabbed password/pattern section. Password change form (current + new). Pattern lock canvas (220px) for setting/updating authentication pattern.
- **Data export:** Date range picker with CSV export button. Downloads session data for the selected period.
- **Navigation links:** Achievements, Training Presets, Coach Dashboard (coach users only)
- **About:** Version and device info
- **Logout button**

#### 3.5.7 AchievementsView (`AchievementsView.vue`)

Gamification display:
- Current rank with XP progress bar to next rank
- Training streak display (current + longest)
- Achievement badge grid (3 columns) using `AchievementBadge` components. Each badge shows locked/unlocked state and unlock date.
- Progress summary: unlocked count / total with progress bar

#### 3.5.8 CoachView (`CoachView.vue`)

Station management for coach users:
- WebSocket connection status and participant count
- Station control: create named sessions with optional preset loading
- Real-time participant monitoring during active sessions
- Session history for the station

#### 3.5.9 TrainingView (`TrainingView.vue`)

Phone-initiated training control:
- Quick start preset grid (2-column) with accent colors and descriptions
- Remote control section for starting custom configurations on the robot
- Live status display when a session is running

#### 3.5.10 PerformanceView (`PerformanceView.vue`)

Performance trends and analytics:
- Time range selector (7d, 30d, 90d, all)
- Summary cards (sessions, punches, personal bests)
- Trend charts for punch volume and consistency over time
- Period-over-period comparisons

#### 3.5.11 PresetsView (`PresetsView.vue`)

Preset management:
- List of user-created presets with favourite toggle and use count
- "New" button to create presets with name, type, configuration, and description
- Quick-launch on robot via remote command

### 3.6 Gamification System

The gamification system is implemented across both the dashboard backend and frontend:

#### 3.6.1 XP and Ranks (6 Tiers)

| Rank | XP Threshold | Description |
|------|-------------|-------------|
| Novice | 0 | Starting rank |
| Contender | 500 | Consistent beginner |
| Fighter | 1,500 | Regular trainer |
| Warrior | 4,000 | Dedicated practitioner |
| Champion | 10,000 | Advanced athlete |
| Elite | 25,000 | Master level |

XP is earned per session based on: `base_xp * score_multiplier * completion_bonus + streak_bonus`

Base XP varies by mode: Training (50), Sparring (75), Free (25), Power (30), Stamina (40), Reaction (30). Score multiplier ranges from 0.5 to 1.0 based on session score (0-100). Complete sessions earn a 1.5x completion bonus. Streak bonuses add flat XP.

#### 3.6.2 Achievements (12 Badges)

Achievement badges are displayed in the `AchievementsView` using `AchievementBadge` components. Each badge has an ID, name, description, unlock criteria, and locked/unlocked visual state. Badges are unlocked automatically when criteria are met during session completion.

#### 3.6.3 Streaks and Weekly Goals

- **Streaks:** Consecutive days with at least one training session. Tracked as `current_streak` and `longest_streak`.
- **Weekly goals:** User-configurable sessions per week target (1-7, default 3). Progress shown on dashboard with progress bar.

#### 3.6.4 Peer Comparison

The benchmark system loads population norms from `data/benchmarks/population_norms.json` and computes the user's percentile rank for reaction time, punch rate, power, and defense. Demographics (age range, gender) are used for peer group filtering.

### 3.7 Real-Time Session Updates via WebSocket

The dashboard connects to the backend WebSocket at `/ws/{username}/{user_type}` upon login. The `useWebSocketStore` (Pinia) manages the connection lifecycle and message routing. During active training sessions, the backend pushes real-time updates including:
- Punch confirmations
- Drill progress (combos completed/remaining, accuracy, streak)
- Session state changes (countdown, active, rest, complete)
- Coach tips

### 3.8 Vue Frontend Components

Reusable components in `src/boxbunny_dashboard/frontend/src/components/`:

| Component | Purpose |
|-----------|---------|
| `NavBar.vue` | Bottom navigation bar with Home, Training, Performance, History, and Settings tabs |
| `PatternLock.vue` | Reusable 3x3 dot pattern lock canvas for login and settings |
| `RankBadge.vue` | Rank display with icon and XP, available in sm/lg sizes |
| `StreakDisplay.vue` | Flame animation with current and longest streak counts |
| `SessionCard.vue` | Compact session summary card used in history lists and dashboard |
| `StatCard.vue` | Metric card with label, value, icon, color accent, and optional trend arrow |
| `PunchChart.vue` | Punch distribution visualization chart |
| `AchievementBadge.vue` | Individual achievement badge with locked/unlocked states |

---

## 4. Communication Between GUI and Dashboard

### 4.1 The Command File Approach

The touchscreen GUI and phone dashboard communicate through a simple file-based protocol using JSON files in `/tmp/`:

| File | Direction | Purpose |
|------|-----------|---------|
| `/tmp/boxbunny_gui_command.json` | Phone -> GUI | Remote commands from dashboard |
| `/tmp/boxbunny_gui_login.json` | Phone -> GUI | Phone login notification for auto-login |
| `/tmp/boxbunny_dashboard_url.txt` | GUI -> Phone | SSH tunnel public URL for QR code |

### 4.2 Remote Command Protocol

When the phone dashboard sends a command:

1. **Phone:** User taps a button (e.g., "Start Jab-Cross Drill").
2. **API:** The FastAPI backend writes a JSON object to `/tmp/boxbunny_gui_command.json`.
3. **GUI:** A 500ms `QTimer` in `BoxBunnyApp._poll_remote_commands()` checks for the file.
4. **GUI:** On detection, reads the JSON, deletes the file, and executes the command.

The JSON command format:
```json
{
    "action": "start_preset",
    "config": {"combo": {"id": "beginner_007", "name": "Jab-Cross", "seq": "1-2"}, ...},
    "username": "alex"
}
```

Supported actions:

| Action | Behaviour |
|--------|-----------|
| `start_preset` | Loads the preset into the `PresetOverlay` and triggers selection |
| `open_presets` | Opens the preset overlay for manual selection |
| `start_training` | Navigates directly to `training_session` with provided config |
| `setup_drill` | Navigates to `training_config` page (user presses centre pad to start) |
| `navigate` | Generic navigation to any named route |

The phone login file uses a similar protocol:
```json
{
    "username": "alex",
    "user_id": 1,
    "user_type": "individual"
}
```

### 4.3 Why This Approach

The file-based approach was chosen over bidirectional WebSocket communication between GUI and dashboard for several reasons:

1. **Simplicity:** No socket server in the PySide6 GUI. The GUI only needs to poll a file path, which is trivial and error-free.
2. **Reliability:** File writes are atomic on Linux. There is no connection state to manage, no reconnection logic, and no heartbeat protocol.
3. **Decoupled lifecycles:** The GUI and dashboard server can start and stop independently. The file acts as a dead-letter queue -- if the GUI is not running, commands simply accumulate (or are overwritten) harmlessly.
4. **Debugging:** Command files are human-readable JSON that can be manually created with a text editor for testing.
5. **No port conflicts:** The dashboard server owns port 8080. The GUI does not need to bind any port.

The 500ms polling interval introduces at most a half-second latency for remote commands, which is imperceptible in practice (the user is looking at their phone, not the robot screen, when sending a command).

### 4.4 Height Control Commands

Height adjustment is a special case requiring continuous commands. When the user holds the UP or DOWN button on the phone:

1. Phone sends `POST /api/remote/height` with `{"action": "up"}` immediately on touch-start, then at 10Hz while held.
2. The API writes a height command file or publishes directly to the ROS height topic.
3. On touch-end, the phone sends `{"action": "stop"}`.

The height motor (DC motor via MDDS10 driver) responds to `manual_up`, `manual_down`, and `stop` actions published on the `/boxbunny/robot/height` topic via the `HeightCommand` message.

---

## 5. Calibration and First-Time Setup

### 5.1 Overview

Before the robot can execute punches, three calibration data files must be created. This is done through the **V4 Arm Control GUI** (`Boxing_Arm_Control/ros2_ws/unified_v4/unified_GUI_V4.py`), a separate PySide6 application with its own tab-based interface.

### 5.2 Required Data Files

All located in `Boxing_Arm_Control/ros2_ws/unified_v4/data/`:

| File | Purpose | Created By |
|------|---------|------------|
| `arm_config.yaml` | Motor offsets (zero positions), direction signs (+/-), pitch angle limits | Calibration & Twin tab |
| `strike_library.json` | Strike wind-up and apex positions for all 6 punch types, Bezier path waypoints with curvature and arc angle | Strike Library tab |
| `ros_slots.json` | ROS slot assignments mapping slot numbers (1-6) to strikes, plus tuning parameters (speed, delay) | ROS Control tab |

If all three files exist on startup, the V4 GUI auto-activates ROS Control mode. If any are missing, the header bar shows status indicators (green/red for Cal, Lib, ROS) showing which need configuration.

### 5.3 V4 GUI Tabs

The V4 GUI (`unified_GUI_V4.py`) has 9 tabs:

| Tab | Description |
|-----|-------------|
| **Manual Control** | Direct motor position and speed sliders for all 4 motors |
| **Calibration & Twin** | Homing wizard, pitch angle scan, direction calibration (4-test motor sign diagnostic), and 3D digital twin visualisation via forward kinematics |
| **Strike Library** | Click-on-arc interface to place wind-up and apex positions. Bezier path editor with curvature and arc angle controls. Supports all 6 standard strikes: Jab, Cross, Left Hook, Right Hook, Left Uppercut, Right Uppercut |
| **Height Adjustment** | Height motor control (lead-screw DC motor via MDDS10 driver) |
| **IMU Diagnostics** | 4-pad accelerometer data visualisation, gravity calibration, strike detection threshold tuning |
| **Dynamic Sparring** | FSM-based autonomous sparring mode (the robot reacts to pad strikes with counter-punches) |
| **Speed Test** | Timed strike execution benchmarking, peak RPM measurement, per-motor current/power budget analysis, CSV export |
| **Analytics** | Real-time motor position and current plotting, session recording |
| **ROS Control** | Front-end ROS interface for slot assignment, strike execution commands, and integration with the BoxBunny system |

### 5.4 Calibration Workflow

The complete calibration procedure (performed once after assembly or motor replacement):

1. **Zero motors:** Calibration & Twin tab, click "Zero All Here" with the arm in the known home position. This records motor encoder positions as the zero reference.
2. **Set roll zero:** Click "Set Roll Zero" to establish the roll axis reference.
3. **Pitch scan:** Click "Run Pitch Scan" to automatically sweep the pitch motor through its range and record the safe limits.
4. **Direction calibration:** Click "Calibrate Directions" to run a 4-test motor sign wizard. Each test briefly moves one motor and asks the operator to confirm the direction was correct. This determines the sign (+/-) for each motor's control signal.
5. **Create strike library:** Switch to the Strike Library tab. For each of the 6 standard strikes:
   a. Select the strike name from the dropdown.
   b. Click on the arc diagram to place the wind-up position (where the arm retracts to before punching).
   c. Click again to place the apex position (the furthest extension of the punch).
   d. Adjust Bezier curvature and arc angle for the motion path.
   e. Save.
6. **Assign ROS slots:** Switch to the ROS Control tab. Map slots 1-6 to the 6 strikes. Set speed and timing parameters. Click "Save Slots".
7. **Verify:** On next launch, all three data files auto-load and the system is ready.

### 5.5 Clearing Calibration

To reset all calibration data (e.g., after mechanical changes):

```bash
python3 clear_calibration.py          # Interactive -- asks for confirmation
python3 clear_calibration.py --force  # Skips confirmation
```

This moves all data files to a timestamped backup folder (`data/_backup_YYYYMMDD_HHMMSS/`) for recovery.

### 5.6 When Recalibration Is Needed

- Motor replacement or rewiring
- Mechanical linkage changes (arm length, joint geometry)
- After a crash or collision that may have shifted motor zero positions
- If strikes are consistently landing off-target

Partial recalibration is possible: for example, updating only the strike library without re-zeroing the motors.

### 5.7 Micro-ROS Agent Setup (One-Time)

The Teensy 4.1 microcontroller communicates with ROS 2 via micro-ROS. The agent must be installed once:

```bash
# Outside conda environment
source /opt/ros/humble/setup.bash
source ~/microros_ws/install/local_setup.bash

# Start agent (needed before every session)
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200
```

Installation is handled by notebook cell A1 (`notebooks/scripts/setup_microros.sh`). The agent runs as a serial bridge between the Teensy's USB connection and the ROS 2 DDS network.
