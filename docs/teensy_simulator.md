# BoxBunny Teensy Simulator

## 1. Purpose

The Teensy Simulator is a standalone development tool that publishes the same ROS 2 messages as the real Teensy 4.1 hardware. It enables full-stack development and testing of the boxing training system without requiring physical IMU pads, DYNAMIXEL motors, or the Teensy microcontroller.

### What It Simulates

| Hardware Component | Simulator Equivalent |
|---|---|
| 4x IMU pad sensors | Clickable pad buttons (Left, Centre, Right, Head) with configurable force |
| Robot arm strikes | 6 punch buttons (Jab, Cross, L/R Hook, L/R Uppercut) + combo presets |
| DYNAMIXEL motor execution | Speed-based delay simulation with strike_feedback and strike_complete publishing |
| Height motor | Status display of HeightCommand messages from session_manager |
| Yaw motor tracking | Live display of person_direction from cv_node |
| Motor positions/currents | Live display when real hardware is connected alongside simulator |
| IMU raw accelerometer data | Live display from motor_feedback topic |

### When To Use It

- **Drill development**: Testing combo sequences, session lifecycle, and drill validation without hardware
- **GUI development**: Simulating punches and pad strikes to test the touchscreen GUI's response
- **Fusion testing**: Combined with `run_with_ros.py` (CV inference), the simulator provides IMU events for testing the CV+IMU fusion pipeline
- **Integration testing**: Verifying that all ROS nodes communicate correctly before hardware deployment

### Launch

```bash
# Option 1: Direct Python execution
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 tools/teensy_simulator.py

# Option 2: As a ROS 2 node (if registered as entry point)
ros2 run boxbunny_core teensy_simulator
```

---

## 2. GUI Layout

The simulator is built with Tkinter and uses the BoxBunny dark theme (matching the PySide6 touchscreen GUI). The window is approximately 600x920 pixels and contains the following sections from top to bottom:

### Title Bar

```
┌──────────────────────────────────────────────────────┐
│  Box Bunny  Teensy Simulator              HW: LIVE   │
└──────────────────────────────────────────────────────┘
```

- "HW: LIVE" (green) appears when real Teensy motor_feedback is detected
- "HW: --" (grey) when no hardware is connected

### Force Section

```
┌──────────────────────────────────────────────────────┐
│  FORCE   [Light] [Medium] [Hard]           30 m/s²   │
│  ═══════════════════════|════════════════════  0-60   │
└──────────────────────────────────────────────────────┘
```

- Three preset buttons set the acceleration slider to 10 (light), 30 (medium), or 50 (hard) m/s^2
- A continuous slider (0-60 m/s^2) allows fine-grained control
- The acceleration value is published in the `accel_magnitude` field of `PadImpact` messages
- Force level classification: < 20 = light (green), 20-39 = medium (amber), >= 40 = hard (red)
- Force normalised value: `min(1.0, accel / 60.0)`

### Strike Speed Section

```
┌──────────────────────────────────────────────────────┐
│  STRIKE SPEED   [Slow] [Med] [Fast]       20 rad/s   │
│  ═══════════════════════|════════════════════  1-30   │
└──────────────────────────────────────────────────────┘
```

- Sets the motor speed for simulated strike execution and V4 GUI commands
- Three presets: Slow (5 rad/s), Medium (15 rad/s), Fast (25 rad/s)
- Continuous slider from 1 to 30 rad/s (maximum capped at 30 for gear safety)
- Speed is used to calculate simulated execution delay: `delay_s = 10.0 / speed`

### User Strike Detection (Pads)

```
┌──────────────────────────────────────────────────────┐
│  USER STRIKE DETECTION                                │
│                                                      │
│            ┌──────┐                                  │
│            │ HEAD │                                  │
│            └──────┘                                  │
│  ┌──┐  ┌──────┐ ┌────────┐ ┌───────┐  ┌──┐         │
│  │ L│  │ LEFT │ │ CENTRE │ │ RIGHT │  │ R│         │
│  │  │  └──────┘ └────────┘ └───────┘  │  │         │
│  └──┘                                  └──┘         │
└──────────────────────────────────────────────────────┘
```

- Four pad buttons arranged from the USER's perspective (left pad on screen-left, right on screen-right)
- Two arm buttons (L, R) on the far sides
- **Clicking a pad**: publishes `PadImpact` + `ConfirmedPunch` + the corresponding `ArmStrike` at the current force level
- **Clicking an arm**: publishes an `ArmStrike` message (left-click = contact=True, shift+click = contact=False)
- Buttons flash with force-level colour on click (green/amber/red)
- When real hardware fires a strike (via `/robot/strike_detected`), the corresponding pad button flashes to mirror the event

### Robot Arm Punches

```
┌──────────────────────────────────────────────────────┐
│  ROBOT ARM PUNCHES                                    │
│                                                      │
│  [Jab] [Cross] [L Hook] [R Hook] [L Upper] [R Upper]│
│                                                      │
│  [1-2] [1-1-2] [1-2-3] [1-2-3-4] [1-2-5-6]         │
└──────────────────────────────────────────────────────┘
```

- **Top row**: Individual punch buttons. Clicking sends a `/robot/strike_command` to the V4 GUI and publishes the associated `ConfirmedPunch`, `PadImpact`, and `ArmStrike` messages.
- **Bottom row**: Combo preset buttons. Each executes a sequence of punches, waiting for strike completion before sending the next command. Available combos:
  - 1-2: Jab-Cross
  - 1-1-2: Jab-Jab-Cross
  - 1-2-3: Jab-Cross-Left Hook
  - 1-2-3-4: Jab-Cross-Left Hook-Right Hook
  - 1-2-5-6: Jab-Cross-Left Uppercut-Right Uppercut
- Each punch is colour-coded: Jab (blue), Cross (red), Hook (green), Uppercut (purple)
- When the BoxBunny GUI or sparring engine sends a `RobotCommand`, the corresponding punch button flashes to show the incoming command

### Robot Arm Execution

```
┌──────────────────────────────────────────────────────┐
│  ROBOT ARM EXECUTION                                  │
│                                                      │
│  [x] Auto Execute    [ ] Forward to HW               │
│                                                      │
│  Pending: L Hook → left arm                          │
│                                                      │
│  [EXECUTE]    Delay: speed-based                     │
│                                                      │
│  L ARM: idle            R ARM: EXECUTING              │
└──────────────────────────────────────────────────────┘
```

This section controls how incoming robot commands (from the sparring engine or drill manager) are handled. Three execution modes:

| Mode | Checkbox State | Behaviour |
|---|---|---|
| **Auto Execute** | Auto = checked, HW = unchecked | Incoming commands are automatically simulated with a speed-based delay. Strike feedback and completion messages are published after the delay. |
| **Manual Execute** | Auto = unchecked, HW = unchecked | Incoming commands are queued as "Pending". The user must click the EXECUTE button to trigger simulated execution. |
| **Forward to HW** | HW = checked (any Auto state) | Incoming commands are forwarded to the V4 Arm Control GUI via `/robot/strike_command`. The real motors execute the strike. Simulated execution still runs in parallel if Auto is checked. |

- **Pending label**: Shows the queued command (e.g., "Pending: L Hook -> left arm") with amber colour
- **Arm status labels**: Show per-arm execution state (idle / pending / EXECUTING)

### Height and Tracking Status

```
┌──────────────────────────────────────────────────────┐
│  HEIGHT: AUTO            TRACKING: RIGHT              │
└──────────────────────────────────────────────────────┘
```

- **HEIGHT**: Displays the current HeightCommand action received from session_manager. Values: STOP (grey), AUTO (amber), UP (green), DOWN (red)
- **TRACKING**: Displays the person direction from cv_node's `/boxbunny/cv/person_direction` topic. Values: LEFT (blue), CENTRE (green), RIGHT (purple), OFFLINE (grey if no updates for 2 seconds)

### Custom Sequence Builder

```
┌──────────────────────────────────────────────────────┐
│  CUSTOM SEQUENCE                                      │
│                                                      │
│  [+HEAD] [+LEFT] [+CENTRE] [+RIGHT]    [PLAY] [CLR] │
│                                                      │
│  LEFT → CENTRE → RIGHT → CENTRE                     │
└──────────────────────────────────────────────────────┘
```

- Build arbitrary pad strike sequences by clicking +PAD buttons
- PLAY executes the sequence with a configurable interval between strikes
- CLR clears the sequence
- The sequence label shows the current sequence with arrow separators

### Teensy Live Data

```
┌──────────────────────────────────────────────────────┐
│  TEENSY LIVE DATA                                     │
│                                                      │
│  Motors    L1       L2       R1       R2              │
│           +1.23    -0.45    +0.89    -0.12           │
│           +0.34A   -0.12A   +0.56A   -0.23A         │
│                                                      │
│  IMU    Centre    Left     Right    Head              │
│  strike threshold: 20 m/s² (gravity-subtracted)      │
│          12.3     9.8      15.2     8.4              │
│                   pk: 35.2                            │
└──────────────────────────────────────────────────────┘
```

This section displays live data from the real Teensy hardware (when connected):

- **Motor positions** (row 1): DYNAMIXEL positions in radians for all 4 motors (L1, L2, R1, R2)
- **Motor currents** (row 2): Current draw in amps. Colour-coded: > 2.0A = red, > 0.5A = amber, else grey
- **IMU raw magnitudes**: Raw accelerometer magnitude (sqrt(ax^2 + ay^2 + az^2)) for each pad. This includes gravity (~9.8 m/s^2). Colour-coded: > 20 = amber, > 12 = white, else grey
- **Peak labels**: When a real strike is detected by the V4 GUI (via `/robot/strike_detected`), the gravity-subtracted peak acceleration is shown below the IMU value for 1.5 seconds

Data source: `motor_feedback` topic (Float64MultiArray with 21 doubles):
```
[pos0, pos1, pos2, pos3,          # Motor positions (radians)
 cur0, cur1, cur2, cur3,          # Motor currents (amps)
 can_rx,                           # CAN bus receive flag
 imu0_x, imu0_y, imu0_z,         # IMU 0 (Centre) raw accel
 imu1_x, imu1_y, imu1_z,         # IMU 1 (Right) raw accel
 imu2_x, imu2_y, imu2_z,         # IMU 2 (Left) raw accel
 imu3_x, imu3_y, imu3_z]         # IMU 3 (Head) raw accel
```

### Event Log

```
┌──────────────────────────────────────────────────────┐
│  EVENT LOG                                            │
│                                                      │
│  PAD  centre   accel=30.0  level=medium              │
│  ARM> L Hook   arm=left  accel=30.0                  │
│  CMD> Cross    (from GUI drill)                      │
│  SIM> Left Hook  completed  (0.7s)                   │
│  HW>  right    peak=35.2m/s²  level=medium           │
└──────────────────────────────────────────────────────┘
```

A scrollable text area showing the last 80 events. Event types:

| Prefix | Source | Description |
|---|---|---|
| `PAD` | User clicked a pad button | Simulated pad impact |
| `ARM` | User clicked an arm button | Simulated arm strike |
| `ARM>` | User clicked a punch button | Robot arm punch simulation |
| `CMD>` | RobotCommand from BoxBunny system | Incoming drill/sparring command |
| `SIM>` | Auto/manual execution completed | Simulated strike completion with duration |
| `HW>` | Real Teensy hardware | Strike detected from V4 GUI with peak acceleration |

---

## 3. Execute Mode

### Auto vs Manual Execution

When a `RobotCommand` arrives from the BoxBunny system (e.g., from sparring_engine or drill_manager), the simulator determines how to handle it based on the checkbox state:

```
RobotCommand arrives
        │
        ├──► Flash punch button (always)
        │
        ├──► If "Forward to HW" checked:
        │    └──► send_strike_command(slot, duration=5.0, speed=...)
        │         to /robot/strike_command for V4 GUI
        │
        └──► If "Auto Execute" checked:
             └──► _start_simulated_execution(cmd)
                  │
                  │  speed = speed_slider value (rad/s)
                  │  delay_s = 10.0 / max(speed, 1.0)
                  │
                  │  Example delays:
                  │  ┌──────────┬──────────┐
                  │  │ Speed    │ Delay    │
                  │  ├──────────┼──────────┤
                  │  │  5 rad/s │ 2.0s     │
                  │  │ 10 rad/s │ 1.0s     │
                  │  │ 15 rad/s │ 0.67s    │
                  │  │ 20 rad/s │ 0.50s    │
                  │  │ 25 rad/s │ 0.40s    │
                  │  │ 30 rad/s │ 0.33s    │
                  │  └──────────┴──────────┘
                  │
                  │  After delay_ms:
                  └──► _finish_execution(cmd, delay_s)
                       │
                       ├──► simulate_strike_complete()
                       │    Publishes to /robot/strike_feedback:
                       │    {"slot": 3, "strike": "Left Hook",
                       │     "status": "completed",
                       │     "duration_allowed": 5.0,
                       │     "duration_actual": 0.67}
                       │
                       └──► Publishes to /boxbunny/robot/strike_complete:
                            {"punch_code": "3", "status": "completed",
                             "duration_ms": 670, "strike": "Left Hook"}
```

### Manual Execution Workflow

1. A RobotCommand arrives (e.g., from sparring engine)
2. The punch button flashes and the pending label updates: "Pending: L Hook -> left arm"
3. The arm status shows: "L ARM: pending" (amber)
4. The user clicks the EXECUTE button
5. Simulated execution begins: "L ARM: EXECUTING" (red)
6. After the speed-based delay, execution completes
7. Strike feedback is published
8. Status resets: "L ARM: idle" (grey), "Pending: --"

This is useful for step-by-step debugging of combo sequences and drill validation.

---

## 4. Integration with V4 Arm Control GUI

The Teensy Simulator acts as a bridge between the BoxBunny system and the V4 Arm Control GUI (the separate motor control application that handles DYNAMIXEL FSM execution).

### Communication Protocol

```
BoxBunny System                 Teensy Simulator              V4 Arm Control GUI
      │                               │                              │
      │ RobotCommand                  │                              │
      │ /boxbunny/robot/command       │                              │
      ├──────────────────────────────►│                              │
      │                               │ /robot/strike_command        │
      │                               │ {"slot": 3, "duration": 5.0, │
      │                               │  "speed": 15.0}             │
      │                               ├─────────────────────────────►│
      │                               │                              │
      │                               │ /robot/strike_feedback       │
      │                               │ {"slot": 3, "strike":        │
      │                               │  "Left Hook", "status":     │
      │                               │◄─────────────────────────────┤
      │                               │  "completed"}               │
      │                               │                              │
      │ /boxbunny/robot/strike_complete│                              │
      │◄──────────────────────────────┤                              │
      │ {"punch_code": "3",           │                              │
      │  "status": "completed"}       │                              │
      │                               │                              │
      │                               │ /robot/strike_detected       │
      │                               │ {"pad_index": 2,             │
      │                               │◄─────────────────────────────┤
      │                               │  "peak_accel": 35.2}        │
      │                               │                              │
      │                               │ (flashes pad on GUI,         │
      │                               │  logs to event log)          │
```

### Punch Slot Assignment

The V4 Arm Control GUI uses a slot-based system where each slot (1-6) has an assigned punch type with specific motor positions. The simulator maps BoxBunny punch codes directly to slots:

| Punch Code | Slot | Punch Name |
|---|---|---|
| "1" | 1 | Jab |
| "2" | 2 | Cross |
| "3" | 3 | Left Hook |
| "4" | 4 | Right Hook |
| "5" | 5 | Left Uppercut |
| "6" | 6 | Right Uppercut |

The `send_punch_slots()` method can assign custom configurations to the V4 GUI:

```python
def send_punch_slots(self, slots: dict) -> None:
    """Assign punch types to V4 GUI slots.
    
    Args:
        slots: {1: {"arm": "left", "strike": "Jab"}, ...}
    """
    msg = StdString()
    msg.data = json.dumps({str(k): v for k, v in slots.items()})
    self._pub_punch_slots.publish(msg)
```

### System Enable/Disable

The simulator can arm or disarm the V4 GUI's motor control:

```python
def send_system_enable(self, enable: bool) -> None:
    """Arm or disarm the V4 GUI motor control."""
    msg = StdString()
    msg.data = "enable" if enable else "disable"
    self._pub_system_enable.publish(msg)
```

This is published to `/robot/system_enable` and controls whether the V4 GUI will accept and execute strike commands.

### Forward to Hardware Mode

When the "Forward to HW" checkbox is enabled:

1. Every incoming `RobotCommand` is translated to a `/robot/strike_command` JSON message
2. The V4 Arm Control GUI receives this and executes the strike using its FSM (alignment -> windup -> apex -> snap-back)
3. The V4 GUI publishes completion feedback on `/robot/strike_feedback`
4. The simulator receives this feedback, flashes the appropriate arm status, and logs the event
5. The V4 GUI's IMU diagnostics tab detects pad strikes and publishes `/robot/strike_detected`
6. The simulator receives this and flashes the corresponding pad button

This allows the simulator to serve as a visual monitor of real hardware activity, showing which commands were sent and when strikes were detected.

---

## 5. Combo System

### Sequential Execution with Completion Waiting

Combo presets (e.g., 1-2-3: Jab-Cross-Left Hook) execute punches sequentially, waiting for each strike to complete before sending the next command. This mirrors real robot behaviour where the arm must return to guard position before throwing the next punch.

```python
def _play_combo(self, sequence: list) -> None:
    """Play a combo sequence, waiting for each strike to complete."""
    self._combo_queue = list(sequence)  # e.g., ["jab", "cross", "l_hook"]
    self._combo_playing = True
    self._play_next_combo_step()

def _play_next_combo_step(self) -> None:
    """Send the next punch in the combo queue."""
    if not self._combo_queue:
        self._combo_playing = False
        return
    ptype = self._combo_queue.pop(0)
    self._on_punch(ptype, color)
    # Timeout: if no feedback within 10s, move on anyway
    self._combo_timeout_id = self._root.after(
        10000, self._play_next_combo_step)
```

The flow for a 1-2-3 combo:

```
[1] Jab sent → /robot/strike_command → wait for feedback
                                            │
                              strike_feedback received
                                            │
[2] Cross sent → /robot/strike_command → wait for feedback
                                            │
                              strike_feedback received
                                            │
[3] L Hook sent → /robot/strike_command → wait for feedback
                                            │
                              strike_feedback received
                                            │
                         Combo complete
```

If no strike_feedback arrives within 10 seconds (e.g., V4 GUI is not running), the combo advances automatically via the timeout.

### Strike Feedback Integration

When the V4 GUI completes a strike, it publishes feedback that the simulator processes:

```python
def _on_strike_feedback_gui(self, data: dict) -> None:
    """Handle V4 GUI strike completion."""
    # data: {"slot": 3, "strike": "Left Hook", "status": "completed",
    #        "duration_actual": 0.85}
    
    # If combo is playing, advance to next step
    if self._combo_playing and self._combo_timeout_id:
        self._root.after_cancel(self._combo_timeout_id)
        self._play_next_combo_step()
```

This creates a tight feedback loop: the simulator sends a command, the V4 GUI executes it on real motors, and when the motors complete the sequence, the next command is sent. This ensures combos play at the actual hardware speed, not an arbitrary timer.

---

## 6. ROS Node: TeensySimulatorNode

The simulator's ROS node (`TeensySimulatorNode`) publishes on the same topics as the real Teensy hardware, making it transparent to the rest of the system.

### Published Topics

| Topic | Message Type | Description |
|---|---|---|
| `/boxbunny/imu/pad/impact` | PadImpact | Simulated pad strikes with configurable force |
| `/boxbunny/imu/arm/strike` | ArmStrike | Simulated arm contact (struck/missed) |
| `/boxbunny/imu/status` | IMUStatus | Heartbeat at 1Hz. `is_simulator=True` unless real HW detected |
| `/boxbunny/punch/confirmed` | ConfirmedPunch | Direct confirmed punches (bypass fusion for testing) |
| `/robot/strike_command` | String (JSON) | Commands forwarded to V4 GUI |
| `/robot/punch_slots` | String (JSON) | Punch type assignments for V4 GUI slots |
| `/robot/system_enable` | String | "enable" / "disable" for V4 GUI motor control |
| `/robot/strike_feedback` | String (JSON) | Simulated strike completion when auto-executing |
| `/boxbunny/robot/strike_complete` | String (JSON) | Simulated completion for BoxBunny GUI consumption |

### Subscribed Topics

| Topic | Message Type | Description |
|---|---|---|
| `/boxbunny/robot/command` | RobotCommand | Incoming drill/sparring commands |
| `/boxbunny/robot/height` | HeightCommand | Height adjustment commands from session_manager |
| `/boxbunny/cv/person_direction` | String | Person tracking direction from cv_node |
| `/robot/strike_feedback` | String (JSON) | Completion feedback from V4 GUI |
| `/robot/strike_detected` | String (JSON) | Pad strike detection from V4 GUI (real IMU) |
| `motor_feedback` | Float64MultiArray | Raw Teensy positions/currents/IMU for live display |

### IMU Pad Mapping

The simulator uses the same pad index mapping as the real hardware, matching the configuration in `config/boxbunny.yaml`:

```python
# Teensy IMU indices → user-perspective pad names
# Physical wiring: Teensy index 1 = user's RIGHT, index 2 = user's LEFT
self._imu_pad_map = {0: "centre", 1: "right", 2: "left", 3: "head"}
```

This ensures that when the V4 GUI publishes `/robot/strike_detected` with `{"pad_index": 2, "peak_accel": 35.2}`, the simulator correctly identifies this as the LEFT pad and flashes the left button.

---

## 7. Typical Development Workflows

### Workflow 1: Pure Simulation (No Hardware)

```bash
# Terminal 1: Start the simulator
python3 tools/teensy_simulator.py

# Terminal 2: Start core nodes
ros2 launch boxbunny_core full_system.launch.py

# Use the simulator to:
# - Click pad buttons to simulate user strikes
# - Click punch buttons to simulate robot arm actions
# - Play combo presets to test drill sequences
# - Watch the GUI respond to simulated events
```

### Workflow 2: CV + IMU Fusion Testing

```bash
# Terminal 1: Start CV inference with ROS bridge
cd action_prediction
python3 ../notebooks/scripts/run_with_ros.py --no-video

# Terminal 2: Start punch_processor and imu_node
ros2 run boxbunny_core punch_processor &
ros2 run boxbunny_core imu_node

# Terminal 3: Start the simulator (for pad strikes)
python3 tools/teensy_simulator.py

# Terminal 4: Start fusion monitor
python3 notebooks/scripts/fusion_monitor.py

# Throw punches at camera. Click the correct pad on the simulator.
# Watch the fusion monitor confirm the match.
```

### Workflow 3: Hardware-in-the-Loop

```bash
# Terminal 1: Start V4 Arm Control GUI (connects to real Teensy)
# (separate application)

# Terminal 2: Start simulator with Forward to HW enabled
python3 tools/teensy_simulator.py
# Check "Forward to HW" checkbox

# Terminal 3: Start sparring engine
ros2 run boxbunny_core sparring_engine

# The sparring engine generates RobotCommands.
# The simulator forwards them to the V4 GUI.
# The V4 GUI executes strikes on real motors.
# Strike feedback flows back through the simulator.
# Teensy Live Data section shows real motor positions and IMU values.
```
