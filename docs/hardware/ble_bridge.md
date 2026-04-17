# BLE Bridge — Jetson ↔ Arduino (base rotation + height)

How the Jetson talks to the base-control Arduino. Covers both sides of the link: the firmware (`ble_control.ino`) and the Jetson-side ROS 2 node (`ble_bridge_node`).

If you just want the Arduino command language, see [`Boxing_Arm_Control/ros2_ws/ble_control/README.md`](../../Boxing_Arm_Control/ros2_ws/ble_control/README.md). This doc is the *integration* story.

---

## Why it exists

The base of the robot has two motors — a yaw rotation motor (turns the whole body toward the user) and a height motor (the lead-screw lift). Both need to be driven from the Jetson, but:

1. **BLE is single-owner.** Classic BLE peripherals accept one central at a time. Only one Jetson process can own the link.
2. **Two different command streams share it.** Height buttons from the GUI/phone, *and* live person-tracking rotation from CV.

So we built `ble_bridge_node` — one ROS 2 node that owns the BLE connection and multiplexes both streams onto it.

---

## The two sides

```
┌──────────────────────────────────┐      BLE (classic GATT)     ┌────────────────────────────┐
│  Jetson — ble_bridge_node        │ ◄─────────────────────────► │  Arduino Uno R4 WiFi       │
│  src/boxbunny_core/boxbunny_core │    service UUID 0x1820      │  ble_control.ino           │
│  /ble_bridge_node.py             │    CMD char   0x2a68        │  "BoxBunny Base" advert    │
│  (bleak, async)                  │    FB  char   0x2a69        │  500ms scan + connect      │
└──────────────────────────────────┘                              └────────────────────────────┘
```

| UUID | Role |
|------|------|
| `00001820-0000-1000-8000-00805f9b34fb` | Motor service |
| `00002a68-0000-1000-8000-00805f9b34fb` | CMD (Jetson → Arduino, write) |
| `00002a69-0000-1000-8000-00805f9b34fb` | Feedback (Arduino → Jetson, notify) |

---

## Command language

The Jetson writes ASCII strings to the CMD characteristic. Full list in [`ble_control/README.md`](../../Boxing_Arm_Control/ros2_ws/ble_control/README.md); the subset the bridge actually emits:

| Command | Action |
|---------|--------|
| `L:<RPM>` | Rotate left (yaw). Blocked at −90° hard limit. |
| `R:<RPM>` | Rotate right (yaw). Blocked at +90° hard limit. |
| `S` | Stop base motor. |
| `HUP:<PWM>` | Drive height motor up at PWM 0–255. |
| `HDOWN:<PWM>` | Drive height motor down at PWM 0–255. |
| `HSTOP` | Stop height motor. |

The Arduino publishes telemetry back via the feedback characteristic (`baseDeg,vel,current_A,busV,dir,limit`). `ble_bridge_node` logs this at debug level but does not act on it today.

---

## ROS-side interface (what the node publishes / subscribes / serves)

**Source:** [src/boxbunny_core/boxbunny_core/ble_bridge_node.py](../../src/boxbunny_core/boxbunny_core/ble_bridge_node.py)

Subscribes:
- `/boxbunny/robot/height` (`HeightCommand`) — translates to `HUP:`/`HDOWN:`/`HSTOP`.
- `/boxbunny/cv/person_direction` (`std_msgs/String`) — "left" / "right" / "centre".
- `/boxbunny/cv/user_tracking` (`UserTracking`) — depth + bbox for the primary user.

Publishes:
- `/boxbunny/ble/status` (`std_msgs/String`, 1 Hz, JSON) — `{connected, tracking, last_cmd, near_range_m}`. The GUI uses this to enable/disable the Start Tracking button.

Services:
- `/boxbunny/ble/tracking_start` (`std_srvs/srv/Trigger`) — enable CV-driven rotation forwarding.
- `/boxbunny/ble/tracking_stop` (`std_srvs/srv/Trigger`) — immediately send `S` and disable forwarding.

Tuning (in [config/boxbunny.yaml](../../config/boxbunny.yaml) → `ble_tracking:`):
- `near_range_m` (default `0.8`) — user must be closer than this to trigger rotation.
- `hysteresis_m` (default `0.15`) — extra slack before *disengaging* once engaged.
- `rpm` (default `500`) — rotation speed for `L:/R:` commands.

---

## Internal architecture

```
                           ┌─────────────────────────── ble_bridge_node ─────────────────────────────┐
ROS callbacks              │                                                                          │
(HeightCommand,            │  _on_height       ─┐                                                    │
 person_direction,         │  _on_person_dir   ─┤    threading.Lock            queue.Queue[str]       │
 user_tracking)      ─────►│  _on_user_track   ─┤    (protects state) ────────►(thread-safe handoff)  │
                           │  _srv_track_*     ─┘                                       │            │
                           │                                                            ▼            │
                           │  timers:                                          ┌────────────────┐    │
                           │   _deadman_tick (10 Hz, 500 ms height deadman)    │ BLE worker     │    │
                           │   _publish_status (1 Hz)                          │ thread         │    │
                           │                                                   │  bleak async   │    │
                           │                                                   │  scan/connect  │    │
                           │                                                   │  loop          │    │
                           │                                                   └────────────────┘    │
                           └──────────────────────────────────────────────────────────┬──────────────┘
                                                                                      │ write_gatt_char
                                                                                      ▼
                                                                              ┌───────────────┐
                                                                              │  Arduino BLE  │
                                                                              └───────────────┘
```

- A daemon thread runs an `asyncio` loop that hosts `bleak`. ROS callbacks never block on BLE I/O — they just push commands onto a `queue.Queue`.
- The BLE worker drains the queue at ~20 Hz (50 ms between writes).
- Scan → connect → serve → disconnect → back to scan, with a 2 s backoff.
- On disconnect the worker **drains the queue** so stale commands don't pour into the motor when the link comes back up.

---

## Safety layers (height motor)

The height motor has no encoder, no limit switches, and no position sensor. It is pure open-loop PWM. That means a runaway command could keep driving until the mechanical end-stop. There are **three defences** stacked:

### 1. Arduino refresh-window watchdog (firmware — the strongest one)

```cpp
// ble_control.ino, loop()
if (lastHeightCmdMs != 0 && (millis() - lastHeightCmdMs) > HEIGHT_WINDOW_MS) {
    analogWrite(HEIGHT_PWM_PIN, 0);
    lastHeightCmdMs = 0;
}
```

Every `HUP:`/`HDOWN:` command stamps `lastHeightCmdMs`. If no command arrives within `HEIGHT_WINDOW_MS` (currently **250 ms**), the firmware zeros the PWM.

Consequence: **the height motor can only run while commands are actively arriving**. Absence of signal *is* stop. BLE drop, Jetson crash, unplugged cable — all produce the same safe result.

The GUI publishes at 10 Hz (100 ms interval) while a button is held, so 250 ms is ~1 dropped packet of margin. Smooth under normal conditions, instantly safe under failure.

### 2. Jetson-side deadman (`ble_bridge_node`)

```python
# 10 Hz timer callback
if self._height_active and (now - self._last_height_cmd_ts > 0.5):
    self._enqueue("HSTOP")
    self._height_active = False
```

Belt-and-braces. If a publisher dies mid-motion (GUI crash, stuck Wi-Fi on the phone) but BLE is still up, the bridge fires `HSTOP` within 500 ms.

### 3. GUI `HSTOP` on button release

The GUI's press-and-hold pattern sends `HSTOP` on the mouse/touch release event. Instant stop the moment the user lifts their finger, regardless of the two watchdogs above.

### Base-motor safety

The yaw motor is protected by firmware hard limits at ±90° (`enforceLimits()` runs every loop iteration). No additional watchdog needed because the physical limit is guaranteed.

---

## Person-tracking gating (base rotation only)

When tracking is enabled, the bridge does **not** blindly forward every `person_direction` message. It gates by depth + hysteresis so background walkers don't drive the motor:

```
engage_range = near_range_m                 (default 0.8 m)
disengage_range = near_range_m + hysteresis (default 0.95 m)

if not engaged:
    engage only when depth ≤ 0.8 m
else:
    stay engaged up to 0.95 m, then disengage
```

A user standing at 2 m — even if they're the only person in frame — produces no rotation commands. They have to step into the engagement zone to start tracking.

(The CV side also already picks the largest + most centred bbox per frame, so the bridge sees one user at a time by construction.)

---

## Lifecycle & expected behaviour

| Event | What happens |
|-------|--------------|
| Jetson boot + `boxbunny_full.launch.py` | `ble_bridge_node` starts, begins scanning. |
| Arduino powered on | Starts advertising `BoxBunny Base`. |
| First match | Bridge connects, logs "BLE connected to BoxBunny Base", begins draining its queue. `/boxbunny/ble/status` flips to `{connected: true}`. |
| User presses height button in GUI | 10 Hz `HUP:200` / `HDOWN:200` stream. Motor moves. |
| Finger released | `HSTOP`. Motor stops. |
| GUI crashes mid-hold | Firmware stops motor within 250 ms (absence-of-signal watchdog). |
| BLE link drops | Bridge forces `_tracking_enabled = False` and will reconnect. On reconnect the stale command queue is drained before resuming. |
| Free-training / sparring session ends | `_safety_stop_tracking()` calls the `tracking_stop` service → `S` sent immediately. |
| Jetson shutdown (Ctrl-C) | Bridge sends final `HSTOP` + `S` before disconnecting. |

---

## Gotchas

- **Do not run `notebooks/scripts/test_base_tracking.py` while the main GUI is up.** Both try to own the BLE link. That script is intentionally kept for bench debugging; it is not expected to coexist with the real stack.
- **`bleak` version quirks.** `BleakScanner.discover()` has slightly different signatures across releases. If you upgrade the package, re-test the scan loop.
- **Arduino must be power-cycled after re-flashing.** The BLE stack sometimes gets stuck advertising stale service UUIDs until a full reboot.
- **Height motor with no command stream parked mid-travel** — because the watchdog zeros PWM, the lead-screw can back-drive a tiny amount under load. Not dangerous, but the user may see a few mm of sag if they release mid-lift.
- **`respawn=True` in the launch file.** If the node crashes (`bleak` raising), systemd/launch restarts it after 3 s. Check the log before assuming everything is fine — repeated crashes usually mean a missing `bleak` install or a BLE-adapter permission issue (`sudo usermod -aG bluetooth $USER`).

---

## Where the GUI talks to it

Two tracking toggles exist in the main GUI, both wired to the same services:

1. **Sparring Setup page** (`src/boxbunny_gui/boxbunny_gui/pages/sparring/sparring_config_page.py`) — pre-configure tracking before starting.
2. **Sparring / Free-training session pages** — toggle mid-session. Always auto-stops on `on_stop`, `on_timer_done`, and `on_leave` via `_safety_stop_tracking()`.

Both pages observe `/boxbunny/ble/status` via `GuiBridge.ble_status_changed` to enable/disable the Start Tracking button (disabled while BLE is offline) and to sync with the bridge's authoritative tracking state across page transitions.

Height control flows through the same bridge but via a different code path:

- Main GUI Settings page (press-and-hold height buttons) → `GuiBridge.publish_height_command(action)` → `/boxbunny/robot/height` (`HeightCommand`) → `ble_bridge_node` → BLE.
- Phone dashboard (`POST /api/remote/height`) → same topic → same bridge.

Both produce the identical `HUP:/HDOWN:/HSTOP` stream on the BLE link.
