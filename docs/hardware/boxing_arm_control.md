# Boxing Arm Control (the `Boxing_Arm_Control/` submodule)

Everything that *swings* and *turns* on BoxBunny lives in this submodule. The main `boxing_robot_ws` repo drives sessions and interprets punches; `Boxing_Arm_Control/` is the low-level hardware layer: motors, IMUs, firmware, and the calibration/tuning GUI.

This doc is a **navigation layer** — if you need the exhaustive details, the submodule has its own READMEs and they are the source of truth. Links below.

---

## Where to find what

```
Boxing_Arm_Control/
├── README.md                              <- submodule top-level (calibration workflow, GUI tabs)
├── CHANGELOG.md                           <- hardware change log
├── teensy_firmware/
│   ├── teensy_firmware_V4/                <- active arm firmware (200 Hz unified loop)
│   │   └── README.md                      <- firmware build + flash instructions
│   └── 4_motors_test/                     <- standalone motor sanity test
└── ros2_ws/
    ├── unified_v4/                        <- V4 PyQt5 arm-calibration + control GUI
    │   ├── unified_GUI_V4.py              <- main app, ROS interface
    │   ├── homing_tab.py / strike_library_tab.py / strike_speed_tab.py
    │   ├── arm_kinematics.py              <- FK/IK
    │   └── README.md                      <- V4-specific topic contract
    ├── ble_control/                       <- base rotation + height Arduino (BLE)
    │   ├── ble_control.ino                <- firmware source
    │   ├── controller.html                <- standalone web controller (debug)
    │   └── README.md                      <- BLE pairing + command protocol
    └── _archive/                          <- V1/V2/V3 legacy, ignored by design
```

Primary reading:

- [`Boxing_Arm_Control/README.md`](../../Boxing_Arm_Control/README.md) — calibration workflow, every GUI tab, hardware table.
- [`Boxing_Arm_Control/ros2_ws/unified_v4/README.md`](../../Boxing_Arm_Control/ros2_ws/unified_v4/README.md) — the V4 arm GUI's full ROS contract (topics in, topics out).
- [`Boxing_Arm_Control/ros2_ws/ble_control/README.md`](../../Boxing_Arm_Control/ros2_ws/ble_control/README.md) — BLE UUIDs + command language.
- [hardware/ble_bridge.md](ble_bridge.md) — the Jetson-side companion to `ble_control/`.

---

## The two independent hardware subsystems

BoxBunny has **two separate embedded boards** that the Jetson talks to. They do not talk to each other.

### 1. Arm subsystem — Teensy 4.0 + 4× Damiao motors + 4× MPU6050 IMUs

```
┌───────────────────┐  USB/Serial    ┌──────────────┐  CAN 1 Mbps  ┌─────────────────────┐
│ Jetson Orin NX    │◄──────────────►│ Teensy 4.0   │◄────────────►│ 4× Damiao DM-J4310  │
│ micro-ROS agent   │   micro-ROS    │ Firmware V4  │              │ (MIT mode, 24 V)    │
│ + unified_GUI_V4  │                │ 200 Hz loop  │              │                     │
└───────────────────┘                │              │  I²C 400 kHz ├─────────────────────┤
         ▲                           └──────────────┘◄────────────►│ 4× MPU6050 pads     │
         │ ROS 2 topics                                             └─────────────────────┘
         ▼
   BoxBunny ROS graph
   (robot_node, punch_processor, imu_node, …)
```

- **Motors**: two arms × two joints each (shoulder pitch + shoulder roll). MIT-mode CAN control at 1 Mbps.
- **Pad IMUs**: four MPU6050s on a dual-bus I²C arrangement — one per pad (centre, left, right, head).
- **Firmware responsibility**: motor CAN comms, IMU reading, micro-ROS bridge to the Jetson at 200 Hz. Safety: current limit (3 A), watchdog, calibrated home on disable.
- **Jetson responsibility**: the V4 GUI (PyQt5) handles calibration, strike-library editing, and executes strikes from `/robot/strike_command`. `robot_node.py` in `boxbunny_core` is the thin bridge between the BoxBunny ROS graph and the V4 GUI.

Key topics on the arm side (full list in `ros2_ws/unified_v4/README.md`):

| Topic | Dir | Purpose |
|-------|-----|---------|
| `motor_commands` | → Teensy | `[pos0-3, speed0-3, enable]` heartbeat |
| `motor_feedback` | ← Teensy | 21-element state + IMU feedback at 200 Hz |
| `/robot/strike_command` | → V4 GUI | JSON `{slot, duration, speed}` |
| `/robot/strike_feedback` | ← V4 GUI | `{slot, strike, status, duration_actual}` |
| `/robot/strike_detected` | ← V4 GUI | Pad-IMU strike events with peak acceleration |
| `/robot/system_enable` | → V4 GUI | `"enable"` / `"disable"` |

### 2. Base subsystem — Arduino + base-rotation motor + height motor + BLE

```
┌───────────────────┐       BLE        ┌────────────────────────┐
│ Jetson Orin NX    │◄────────────────►│ Arduino (ble_control)  │
│ ble_bridge_node   │   bleak client    │ BoxBunny Base (BLE)    │
└───────────────────┘                  │ ┌────────┬──────────┐  │
                                       │ │ Base   │ Height   │  │
                                       │ │ motor  │ motor    │  │
                                       │ │ (CAN)  │ (MDDS10) │  │
                                       │ └────────┴──────────┘  │
                                       └────────────────────────┘
```

- **Base-rotation motor**: the torso yaw — turns the whole robot toward the user. CAN-driven, ±90° software limits in firmware.
- **Height motor**: lead-screw DC motor (MDDS10 H-bridge driver). Open-loop PWM with a refresh-window watchdog (250 ms) in firmware. No encoder on the lift.
- **BLE link**: single-owner. Only *one* Jetson process can be connected. `ble_bridge_node` is that process — everything else must go through it.

Full details: [hardware/ble_bridge.md](ble_bridge.md) and [`Boxing_Arm_Control/ros2_ws/ble_control/README.md`](../../Boxing_Arm_Control/ros2_ws/ble_control/README.md).

---

## Lifecycle — what boots and in what order

1. **Teensy** flashes from power-up, runs its 200 Hz loop, waits for micro-ROS agent.
2. **Arduino (base-control)** boots, starts BLE advertising as `BoxBunny Base`.
3. **Jetson / ROS graph** starts via `ros2 launch boxbunny_core boxbunny_full.launch.py`. That launch brings up:
   - The micro-ROS agent (for the Teensy) — usually started by the notebook cell or a helper script.
   - Every BoxBunny node, including `robot_node` and **`ble_bridge_node`**.
4. **V4 Arm GUI** (`unified_GUI_V4.py`) is started separately — it owns the Teensy side of the link. The V4 GUI's `ROS Control` tab auto-activates once all three config files exist (`arm_config.yaml`, `strike_library.json`, `ros_slots.json`).
5. **Main BoxBunny GUI** (touchscreen) comes up. `ble_bridge_node` auto-connects to `BoxBunny Base` in the background. Height and tracking just work.

---

## Calibration — where the config lives

Calibration is owned by the V4 GUI. Three files under `Boxing_Arm_Control/ros2_ws/unified_v4/data/`:

| File | What it stores |
|------|----------------|
| `arm_config.yaml` | Motor zero offsets, direction signs, soft limits per arm/joint |
| `strike_library.json` | Per-strike wind-up + apex waypoints (joint-space) |
| `ros_slots.json` | Which strike is assigned to which of the 6 slots + per-slot tuning |

Calibration steps live in [`Boxing_Arm_Control/README.md`](../../Boxing_Arm_Control/README.md) under "Calibration Workflow". The short version:

```
Calibration & Twin → Zero All Here → Pitch Scan → Direction Calibration
Strike Library     → Place wind-up + apex per strike (× 6)
ROS Control        → Assign slots 1–6 → Save
```

Once done, the V4 GUI loads everything on boot and `robot_node` can start firing strikes via slot number.

---

## How this submodule plugs into the main BoxBunny system

```
sparring_engine / drill_manager
         │
         │  RobotCommand { punch_code, speed, source }
         ▼
/boxbunny/robot/command                                 <- BoxBunny-side contract
         │
         ▼
robot_node (in boxbunny_core, this repo)
         │
         │  String JSON { slot, duration, speed }
         ▼
/robot/strike_command                                   <- V4 GUI contract
         │
         ▼
V4 Arm Control GUI (Boxing_Arm_Control submodule)
         │
         │  motor_commands (Float64MultiArray)
         ▼
Teensy firmware → Damiao motors
```

So there is a *hard boundary* between the two worlds:

- **BoxBunny-side contract** — `RobotCommand`, `HeightCommand`, `SessionState`, etc. Lives under `config/ros_topics.yaml` in this repo.
- **V4 GUI contract** — `/robot/strike_command`, `/robot/strike_feedback`, etc. Lives in [`Boxing_Arm_Control/ros2_ws/unified_v4/README.md`](../../Boxing_Arm_Control/ros2_ws/unified_v4/README.md).

`robot_node.py` is the translator between the two. If you rename a topic in one, check the other.

---

## Known cross-system gotchas

- **Two publishers on `/boxbunny/robot/strike_complete`** — the V4 GUI publishes a *simplified* payload (no `source` field) in parallel with `robot_node`'s version. This masked the sparring counter counter for a while (see [new_changes.md](../new_changes.md)). The GUI's sparring page now listens to `/boxbunny/robot/command` directly instead of relying on `strike_complete`.
- **BLE is single-owner** — running `notebooks/scripts/test_base_tracking.py` at the same time as the main GUI will fight for the link. Pick one. The standalone tool is kept for bench debugging only.
- **V4 GUI must be running for strikes to complete** — `sparring_engine` waits for `/robot/strike_feedback` to clear `_robot_busy`. If V4 isn't up, the engine has a 6 s fallback timeout (see [new_changes.md](../new_changes.md)) so it doesn't lock up, but the robot arm obviously won't swing.
- **Teensy firmware flash workflow** lives inside `teensy_firmware/teensy_firmware_V4/README.md` — not here, because it needs the Arduino IDE's Teensy board support package installed.

---

## Quick sanity checks

```bash
# Is the Teensy visible?
ls -l /dev/ttyACM*
# Is micro-ROS agent forwarding?
ros2 topic list | grep motor_
# Did the V4 GUI claim the strike_command topic?
ros2 topic info /robot/strike_command

# Is the base-rotation Arduino advertising?
python3 -c "import asyncio; from bleak import BleakScanner; \
  asyncio.run(BleakScanner.discover(timeout=5.0)) " | grep -i boxbunny

# Is ble_bridge_node alive?
ros2 topic echo /boxbunny/ble/status
```

If any of these are silent, start with power and USB connections before anything else. Full troubleshooting matrix: [deployment.md](../deployment.md).
