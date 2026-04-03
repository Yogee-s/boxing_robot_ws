# BoxBunny Training Modes and Sparring Engine

## 1. Training Modes Overview

BoxBunny supports six distinct training modes, each targeting different boxing skills. The mode is set when a session is started via the `StartSession` service and is broadcast to the entire system via the `SessionState` message.

| Mode | Code | Purpose | Robot Behaviour | Data Sources |
|------|------|---------|----------------|--------------|
| **Training** | `training` | Combo drill practice. User follows a prescribed punch sequence. | Stationary (no attacks). Pads light up to indicate target. | CV detection, IMU pad impacts, drill events |
| **Sparring** | `sparring` | Robot attacks user. User must defend and counter. | Attacks using Markov chain sequences. Reacts to user's blocks and idle periods. | CV detection, IMU (pad + arm), defense events, robot commands |
| **Free Training** | `free` | User-paced reactive mode. User punches pads, robot counters. | Purely reactive: counter-punches when user hits a pad. No timer-driven attacks. | IMU pad impacts, robot commands |
| **Reaction Test** | `reaction` | Measures reaction time to visual/audio cues. | Presents random stimuli; measures time to first detected movement. | CV pose estimation (`movement_delta`) |
| **Power Test** | `power` | Measures maximum punch force. | Stationary. | IMU accelerometer magnitude |
| **Stamina Test** | `stamina` | Measures sustained punch rate over a fixed period. | Stationary. | IMU pad impacts over time, fatigue index |


## 2. Sparring Engine

The sparring engine (`src/boxbunny_core/boxbunny_core/sparring_engine.py`) is a ROS 2 node that generates unpredictable robot attack sequences. It activates only during `sparring` and `free` mode sessions.

### 2.1 Punch Code System

Six punches are available, identified by numeric codes:

| Code | Punch | Name Constant |
|------|-------|---------------|
| `"1"` | Jab | `PunchType.JAB` |
| `"2"` | Cross | `PunchType.CROSS` |
| `"3"` | Left Hook | `PunchType.LEFT_HOOK` |
| `"4"` | Right Hook | `PunchType.RIGHT_HOOK` |
| `"5"` | Left Uppercut | `PunchType.LEFT_UPPERCUT` |
| `"6"` | Right Uppercut | `PunchType.RIGHT_UPPERCUT` |

### 2.2 Markov Chain Punch Selection

The robot's next punch is selected using a **first-order Markov chain** -- the probability of each punch depends only on the previous punch. This creates realistic, flowing combinations rather than purely random sequences.

**How it works:**

A 6x6 transition matrix `M` defines the probability of transitioning from punch `i` (row) to punch `j` (column):

```
        jab   cross  l_hook r_hook l_uc  r_uc
jab    [0.15   0.30   0.20   0.15  0.10  0.10]
cross  [0.25   0.10   0.25   0.15  0.10  0.15]
l_hook [0.20   0.25   0.10   0.15  0.15  0.15]
r_hook [0.20   0.20   0.15   0.10  0.15  0.20]
l_uc   [0.15   0.25   0.20   0.15  0.10  0.15]
r_uc   [0.20   0.25   0.15   0.20  0.10  0.10]
```

Each row sums to approximately 1.0. To select the next punch after a jab (row 0), sample from the distribution `[0.15, 0.30, 0.20, 0.15, 0.10, 0.10]` -- there is a 30% chance the robot follows with a cross, 20% for a left hook, etc.

The selection algorithm:

```python
def _select(self) -> int:
    matrix = STYLES[self._active_style]
    weights = list(matrix[self._cur_idx])   # row for current punch
    # Apply weakness bias
    for idx, miss_rate in self._weakness.items():
        weights[idx] += WEAKNESS_BIAS * miss_rate
    # Normalise
    total = sum(weights)
    weights = [w / total for w in weights]
    # Sample using cumulative distribution
    r = random.random()
    cumulative = 0.0
    for idx, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return idx
    return len(weights) - 1
```

### 2.3 Five Fighting Styles

Each style has a different transition matrix, producing characteristic punch patterns:

| Style | Character | Key Tendencies |
|-------|-----------|----------------|
| **Boxer** | Technically precise, balanced | High jab-to-cross transitions (30%). Even distribution of hooks and uppercuts. Classic 1-2 combinations. |
| **Brawler** | Aggressive, power-focused | Low jab usage (5--10%). Heavy emphasis on hooks (25%) and uppercuts (20--25%). Lots of power shot combinations. |
| **Counter-Puncher** | Reactive, precision-focused | Very high jab (25--30%) and cross (30%) probabilities. Minimal uppercuts (10%). Relies on straight punches. |
| **Pressure** | Relentless, volume-based | Extremely high jab probability (30--35%). Keeps coming forward with volume. Less variety in power shots. |
| **Switch** | Unpredictable, style-shifting | No fixed matrix. Every `switch_interval_s` seconds (default 20s), randomly switches to one of the other 4 styles. |

**Matrix example -- Brawler style:**

```
        jab   cross  l_hook r_hook l_uc  r_uc
jab    [0.10   0.15   0.25   0.25  0.10  0.15]
cross  [0.10   0.10   0.25   0.20  0.15  0.20]
l_hook [0.10   0.15   0.15   0.20  0.15  0.25]
r_hook [0.10   0.10   0.25   0.15  0.20  0.20]
l_uc   [0.05   0.15   0.20   0.20  0.15  0.25]
r_uc   [0.05   0.15   0.25   0.20  0.20  0.15]
```

Notice the brawler's tendency to chain hooks and uppercuts together, with only a 5--10% chance of returning to a jab.

### 2.4 Three Difficulty Tiers

Difficulty controls the attack interval (time between robot punches) and the speed of the physical arm movement:

| Difficulty | Attack Interval | Arm Speed | Speed (rad/s) |
|------------|----------------|-----------|----------------|
| **Easy** | 2.0 seconds | Slow | 5.0 |
| **Medium** | 1.2 seconds | Medium | 10.0 |
| **Hard** | 0.7 seconds | Fast | 20.0 |

The main tick loop (runs at 10 Hz) checks `time.time() - last_attack >= interval` before launching the next attack.

### 2.5 Idle Surprise Attacks

If the user stops punching for more than 3 seconds (`IDLE_THRESHOLD_S`), the sparring engine launches a surprise attack at 60% of the normal interval. This punishes passivity and trains the user to stay active.

```python
# In _tick():
if (now - self._last_user_punch) > IDLE_THRESHOLD_S and \
   (now - self._last_attack) > interval * 0.6:
    self._attack(now)
    return
```

### 2.6 Block Reactivity

When the user successfully blocks a robot punch (`set_user_blocked()` is called), the engine ensures the next attack is different from the blocked one. This prevents the robot from throwing the same blocked punch repeatedly:

```python
def _attack(self, now):
    nxt = self._select()  # Markov selection
    if self._blocked_last and nxt == self._cur_idx:
        # User blocked this exact punch -- pick a different one
        alts = [i for i in range(6) if i != nxt]
        nxt = random.choice(alts)
        self._blocked_last = False
    self._cur_idx = nxt
    # ... publish RobotCommand
```

### 2.7 Weakness Bias

The sparring engine accepts a weakness profile (`Dict[str, float]` mapping punch names to miss rates). For each punch type the user struggles to defend against, its selection probability is increased by `WEAKNESS_BIAS * miss_rate` (where `WEAKNESS_BIAS = 0.08`).

Example: If the user's miss rate against left hooks is 0.6 (they fail to defend 60% of the time), the left hook weight is increased by `0.08 * 0.6 = 0.048` in every row of the transition matrix. After renormalisation, this biases the robot toward exploiting the user's weaknesses.

### 2.8 Reactive Counter-Punches

When the user hits a pad (detected by IMU), the sparring engine can immediately fire a counter-punch. This is controlled by difficulty-scaled probability:

| Mode | Counter Probability | Behaviour |
|------|-------------------|-----------|
| Free Training | 100% (always) | Every pad strike triggers a counter |
| Sparring (Easy) | 30% | Occasional counters |
| Sparring (Medium) | 50% | Frequent counters |
| Sparring (Hard) | 80% | Almost always counters |

The counter-punch code is selected from the `pad_counter_strikes` mapping in config (see Section 3). A cooldown period (`counter_cooldown_ms`, default 1500ms) prevents rapid-fire counters.

Critically, when a counter-punch fires, the scheduled attack timer is reset (`self._last_attack = now`) to prevent a double-up (counter + scheduled attack within the same interval).


## 3. Free Training Mode

Free Training is a purely reactive mode with no timer-driven attacks. The robot only moves when the user hits a pad.

### 3.1 How It Differs from Sparring

| Aspect | Sparring | Free Training |
|--------|----------|---------------|
| Timer-driven attacks | Yes (Markov chain) | No |
| Counter-punches | Probabilistic (30--80%) | Always (100%) |
| Idle surprise | Yes (3s threshold) | No |
| Round timer | Yes | No (continuous) |
| Difficulty effect | Attack interval + speed | Speed only |

### 3.2 Pad-to-Counter-Punch Mapping

Configured in `config/boxbunny.yaml` under `free_training.pad_counter_strikes`:

```yaml
free_training:
  pad_counter_strikes:
    centre: ["1", "2"]        # jab or cross (randomly chosen)
    left: ["3"]               # left hook
    right: ["4"]              # right hook
    head: ["1", "2"]          # jab or cross
  counter_cooldown_ms: 1500   # minimum time between counters
  idle_return_s: 5.0          # return to guard after 5s idle
  speed: "medium"             # default counter-punch speed
```

When the user hits the centre pad, the robot randomly selects either a jab (code "1") or cross (code "2") as the counter-punch. Hitting the left pad always triggers a left hook.

### 3.3 Flow

```
User hits LEFT pad
    |
    v
IMU publishes PunchEvent (pad="left")
    |
    v
SparringEngine._on_imu_punch()
    |
    +-- Mode is "free"? Yes (100% counter)
    +-- Cooldown elapsed? (now - _ft_last_counter >= 1.5s)
    +-- Lookup: pad="left" -> strikes=["3"] -> pick "3" (left hook)
    |
    v
Publish RobotCommand {command_type: "punch", punch_code: "3", speed: "medium"}
    |
    v
robot_node translates to /robot/strike_command {slot: 3, duration: 5.0, speed: 10.0}
    |
    v
V4 GUI executes physical left hook strike
```


## 4. Speed Configuration

### 4.1 Speed Presets

Three named speed presets are defined in `constants.py`:

```python
class MotorSpeed:
    SLOW = 8.0       # rad/s
    MEDIUM = 15.0     # rad/s
    FAST = 25.0       # rad/s
    MAX = 30.0        # rad/s (hard cap for gear safety)

    PRESET_MAP = {"slow": SLOW, "medium": MEDIUM, "fast": FAST}
```

### 4.2 Speed Flow: Config to Motor

```
config/boxbunny.yaml
    free_training.speed: "medium"
         |
         v
SparringEngine reads via config_loader
    self._ft_speed = ft.speed  # "medium"
         |
         v
RobotCommand.speed = "medium"
         |
         v
robot_node._on_robot_command()
    speed = _SPEED_MAP["medium"]  # 10.0 rad/s
         |
         v
/robot/strike_command JSON
    {"slot": 3, "duration": 5.0, "speed": 10.0}
         |
         v
V4 GUI RosControlTab._poll_ros_commands()
    ros_speed = 10.0
         |
         v
_execute_ros_strike(ros_speed=10.0)
    effective_spd = min(float(ros_speed), 30.0)  # capped at MAX
         |
         v
Motors move at 10.0 rad/s
```

Note: The V4 GUI has its own speed spinbox (`spin_speed`) for manual control. When a ROS speed override is provided, it takes precedence. The hard cap of 30 rad/s is enforced at the motor command level.

### 4.3 robot_node Speed Mapping

The `robot_node` uses a slightly different speed mapping than `MotorSpeed.PRESET_MAP`:

```python
_SPEED_MAP = {"slow": 5.0, "medium": 10.0, "fast": 20.0}
```

These values are lower than the `MotorSpeed` constants because they represent the speed sent to the V4 GUI, which may apply its own dynamic speed calculation on top (see Section 5.4).


## 5. Robot Arm Control

### 5.1 Architecture

The robot arm is controlled through a multi-layer bridge:

```
BoxBunny System                V4 GUI               Hardware
+-----------------+       +-----------------+      +----------+
| sparring_engine |       | RosControlTab   |      |  Teensy  |
| drill_manager   | ----> | (Qt + threads)  | ---> |  MCU     |
| (RobotCommand)  |       | (strike FSM)    |      | (motors) |
+-----------------+       +-----------------+      +----------+
        |                        |                       |
  /boxbunny/robot/       /robot/strike_command    motor_commands
     command              (JSON String)          (Float64MultiArray)
```

The V4 GUI (`Boxing_Arm_Control/ros2_ws/unified_v4/unified_GUI_V4.py`) is responsible for all physical motor control, including:
- Calibration (joint limits, home positions)
- Strike library (recorded waypoints per punch type)
- Safety (current limiting, position error monitoring)
- Strike execution FSM (alignment, windup, apex, snap-back)

### 5.2 Strike Execution Flow: `_execute_ros_strike`

When the V4 GUI receives a strike command via ROS, it executes a multi-phase finite state machine on a background thread:

```
_execute_ros_strike(slot, strike_name, lib, arm_idx, duration, ros_speed)
    |
    |-- Phase 0: PREPARATION
    |   Read current motor positions
    |   Look up strike waypoints from library: windup, apex
    |   Calculate snap-back position: apex - snap% * (apex - windup)
    |   Determine effective speed: min(ros_speed, 30.0)
    |
    |-- Phase 1: TRANSIT (avoid centre)
    |   Compute approach vector: apex - current_position
    |   Compute strike vector: apex - windup
    |   Calculate angle between vectors
    |   IF angle >= alignment_threshold:
    |     Route through intermediate waypoints to avoid
    |     cutting through the centre space (where the user stands)
    |     Execute each transit waypoint sequentially
    |   Move to windup position
    |
    |-- Phase 2: STRIKE
    |   Single move command from windup to apex
    |   (This is the actual punch -- maximum speed)
    |
    |-- Phase 3: SNAP-BACK
    |   Single move command to recovery position
    |   Recovery = apex - snap% * (apex - windup)
    |   (Partial retraction toward windup, not full return)
    |
    |-- FEEDBACK
    |   Publish strike_feedback: {slot, strike, status, duration_actual}
    |   status = "completed" if actual <= budget, else "overtime"
```

### 5.3 Transit Waypoints (Avoid Centre)

When the arm needs to move from an unrelated position to a strike's windup, it may need to cross through the space in front of the user. The transit waypoint system prevents this:

1. Compute the approach vector (from current position to apex) and the strike vector (from windup to apex).
2. Calculate the angle between them. If the angle exceeds a threshold (configurable via `spin_align`), the arm is approaching from a significantly different direction.
3. Find intermediate waypoints (windup positions of other strikes) that lie between the current and target positions in the rotational axis.
4. Route through these waypoints in order, keeping the arm on a safe path around the perimeter.

```
   Current position          UNSAFE: direct path crosses
   (after right hook)        through user space
        X --------?---------> X  Left hook windup
                  |
                  | USER
                  | ZONE
                  |
   SAFE: route via intermediate waypoints

        X --> (jab windup) --> (cross windup) --> X  Left hook windup
```

### 5.4 Dynamic Speed and Current Limiting

**Dynamic speed calculation:**

When a duration budget is provided, the V4 GUI can automatically compute the minimum speed needed to complete all phases within that budget:

```python
def _compute_dynamic_speed(self, total_dist, duration, base_spd, ros_speed):
    settle_overhead = 0.3  # 3 phases x 0.1s settle time
    available_time = max(0.5, duration - settle_overhead)
    required_spd = total_dist / available_time
    max_spd = self.spin_max_speed.value()  # configurable cap
    effective = min(max(base_spd, required_spd), max_spd)
    return effective
```

**Current limiting safety:**

During every move, the system continuously monitors motor current:

```python
while time.time() - t0 < 8.0:
    c1 = abs(actual_current[motor_0])
    c2 = abs(actual_current[motor_1])
    if c1 > current_limit or c2 > current_limit:
        # EMERGENCY: disable motors immediately
        self.node.motor_enabled = False
        return False  # safety_abort
    # Check position convergence
    if position_error <= 0.2:
        return True  # arrived
    time.sleep(0.01)  # 100 Hz monitoring
```

The current limit is the minimum of the dynamic limit (`spin_current_limit` spinbox) and the calibration safety limit. If any motor exceeds the limit, all motors are immediately disabled and the strike is aborted with status `"safety_abort"`.

### 5.5 Move Execution Detail

Each `_execute_move` call:

1. Reads current position from `actual_pos[arm_offset : arm_offset + 2]`.
2. Computes proportional speeds for each joint: if one joint needs to travel further, it gets a higher speed to synchronise arrival.
3. Sends the target to the Teensy via `set_target_arm(arm_idx, pos0, pos1, spd0, spd1)`.
4. Polls at 100 Hz for:
   - Current limit breach -> abort
   - Position convergence (error <= 0.2 rad per joint) -> success
   - 8-second timeout -> failure


## 6. Person Tracking and Height

### 6.1 Person Direction from CV

The CV node determines which direction the user is facing/moving based on the horizontal position of their bounding box centre:

```
Camera frame (960 pixels wide)
|<--- LEFT --->|<--- CENTRE --->|<--- RIGHT --->|
0             288      480      672             960
              ^                  ^
              |    30% zone      |
         35% boundary       65% boundary
```

**Algorithm:**
1. `bbox_centre_x` from YOLO person detection.
2. Frame is divided: left zone < 35%, centre zone 35%--65%, right zone > 65%.
3. **20-pixel hysteresis** prevents rapid flickering at zone boundaries. The direction only changes if the position crosses at least 20 pixels beyond the boundary into the new zone.

```
Example with hysteresis:

  LEFT zone          CENTRE zone          RIGHT zone
  |                  |                    |
  0    268  288  308    460  480  500    652  672  692    960
       ^         ^      ^         ^      ^         ^
   L->C needs  C->L    C->R needs R->C
   to reach    needs   to reach   needs
   308 px      268 px  500 px     652 px
   (+20)       (-20)   (+20)      (-20)
```

### 6.2 Direction Publishing and Forwarding

```
cv_node publishes String "left" | "right" | "centre"
    on /boxbunny/cv/person_direction
         |
         v
robot_node._on_person_direction()
    Converts to uppercase: "LEFT" | "RIGHT" | "CENTRE"
    Publishes on /robot/yaw_cmd
         |
         v
Teensy firmware receives yaw_cmd
    Drives turning motor to face the user
```

The session manager also subscribes to person direction for data collection (direction timeline segments).

### 6.3 Height Auto-Adjustment

During the countdown phase of a session, the system auto-adjusts the robot's height to match the user:

```
Session enters "countdown" state
         |
         v
UserTracking message arrives (user_detected=True)
         |
         v
session_manager._on_user_tracking()
    IF state == "countdown" AND NOT height_adjusted:
         |
         v
    Compute target: 0.15 * 540 = 81 pixels from top
    (Head top should be at 15% of 540p frame)
         |
         v
    Publish HeightCommand:
        current_height_px = bbox_top_y
        target_height_px = 81
        action = "adjust"
         |
         v
    Set _height_adjusted = True (run once per session)
         |
         v
robot_node._on_height_command()
    error = current_height_px - target_height_px
    direction = "UP" if error > 0 else "DOWN"
    pwm = min(255, int(abs(error) * 2))
         |
         v
    Publish String "{direction}:{pwm}" on /robot/height_cmd
         |
         v
    Teensy MDDS10 driver moves lead screw
```

**Height configuration** (from `config/boxbunny.yaml`):

```yaml
height:
  ideal_top_fraction: 0.15    # head top at 15% of frame height
  deadband_px: 15.0           # no adjustment if error < 15 pixels
  max_iterations: 3           # max adjustment cycles
  settle_delay_ms: 500        # wait for motor to settle
  min_depth_m: 0.5            # ignore detections closer than 0.5m
  max_depth_m: 3.0            # ignore detections further than 3.0m
  no_person_timeout_s: 5.0    # give up if no person detected for 5s
```

### 6.4 Manual Height Control

In addition to auto-adjustment, height can be controlled manually from three interfaces:

1. **Desktop GUI HeightTab:** Physical UP/DOWN/STOP buttons with PWM speed slider (0--255). Press-and-hold interface with software ramp-down on release.

2. **Phone Dashboard:** Press-and-hold UP/DOWN buttons via `POST /api/remote/height` (see Dashboard documentation).

3. **V4 GUI HeightTab Controls:**
   - PWM speed slider (0--255)
   - Reverse direction checkbox (for motor wiring)
   - Ramp-down on stop: reduces PWM in 6 steps over ~300ms

### 6.5 V4 GUI Height Tab

The `HeightTab` class in `unified_GUI_V4.py` provides the physical interface:

```python
class HeightTab(QWidget):
    def _move(self, direction):
        pwm = self.sl_speed.value()       # 0-255 from slider
        if direction == "STOP":
            # Ramp down gradually to avoid mechanical shock
            threading.Thread(target=self._ramp_down, args=(last_dir, last_pwm))
            return
        elif direction == "UP":
            cmd = f"UP:{pwm}"             # e.g. "UP:180"
        else:
            cmd = f"DOWN:{pwm}"           # e.g. "DOWN:180"
        # Publish on /robot/height_cmd
        msg = String(); msg.data = cmd
        self.node.pub_height.publish(msg)

    def _ramp_down(self, direction, start_pwm):
        step = max(start_pwm // 6, 15)   # 6 steps, min 15
        current_pwm = start_pwm
        while current_pwm > 0:
            current_pwm = max(0, current_pwm - step)
            cmd = f"{direction}:{current_pwm}" if current_pwm > 0 else "STOP"
            self.node.pub_height.publish(msg)
            time.sleep(0.05)              # ~50ms between steps
```

### 6.6 User Tracking Data

The `UserTracking` message provides rich spatial data used across the system:

```
float64 timestamp
float32 bbox_centre_x        # horizontal centre of bounding box (pixels)
float32 bbox_centre_y        # vertical centre (pixels)
float32 bbox_top_y           # top edge of bbox (for height adjustment)
float32 bbox_width            # bbox width (pixels)
float32 bbox_height           # bbox height (pixels)
float32 depth                 # distance to user (metres, from RealSense depth)
float32 lateral_displacement  # horizontal shift from baseline (pixels)
float32 depth_displacement    # depth change from baseline (metres)
bool user_detected            # false if no person in frame
```

**Consumers of UserTracking:**
- `session_manager`: collects depth/lateral samples, triggers height auto-adjust
- `punch_processor`: uses lateral/depth displacement for slip detection
- Desktop GUI: displays depth and lateral position in the training view
