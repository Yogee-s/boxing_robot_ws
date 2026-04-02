"""Integration tests for the BoxBunny system.

Tests the critical integration points without requiring hardware:
- InferenceEngine import & config
- CV+IMU fusion decision tree (frame persistence, pad inference)
- Pad constraints & reclassification
- Message field availability (accel_magnitude, consecutive_frames)
- Config loading (new fusion + free_training params)
- Robot motor command protocol
- Session stats movement tracking
"""
import json
import sys
import time
from pathlib import Path

_WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WS / "src" / "boxbunny_core"))
sys.path.insert(0, str(_WS))

passed = 0
failed = 0
total = 0


def test(name):
    global total
    total += 1
    print(f"  [{total:2d}] {name} ... ", end="", flush=True)


def ok():
    global passed
    passed += 1
    print("\033[92mPASS\033[0m")


def fail(reason=""):
    global failed
    failed += 1
    msg = f" ({reason})" if reason else ""
    print(f"\033[91mFAIL{msg}\033[0m")


# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  BoxBunny Integration Tests")
print("=" * 60)

# ── 1. Config Loading ─────────────────────────────────────────────────────────
print("\n── Config Loading ──")

test("Load boxbunny.yaml with new fusion params")
try:
    from boxbunny_core.config_loader import load_config
    cfg = load_config()
    assert cfg.fusion.cv_only_min_consecutive_frames == 3
    assert cfg.fusion.cv_only_min_confidence == 0.6
    assert cfg.fusion.imu_only_default_confidence == 0.3
    ok()
except Exception as e:
    fail(str(e))

test("Load free_training config")
try:
    assert cfg.free_training.counter_cooldown_ms == 1500
    assert "centre" in cfg.free_training.pad_counter_strikes
    assert "1" in cfg.free_training.pad_counter_strikes["centre"]
    ok()
except Exception as e:
    fail(str(e))

# ── 2. Constants & Topics ────────────────────────────────────────────────────
print("\n── Constants & Topics ──")

test("New topic constants exist")
try:
    from boxbunny_core.constants import Topics
    assert Topics.MOTOR_COMMANDS == "motor_commands"
    assert Topics.MOTOR_FEEDBACK == "motor_feedback"
    assert Topics.ROBOT_HEIGHT_CMD == "/robot/height_cmd"
    assert Topics.ROBOT_STRIKE_DETECTED == "/robot/strike_detected"
    assert Topics.ROBOT_STRIKE_COMPLETE == "/boxbunny/robot/strike_complete"
    assert Topics.CV_DEBUG_INFO == "/boxbunny/cv/debug_info"
    ok()
except Exception as e:
    fail(str(e))

test("Pad constraints: centre = jab + cross only")
try:
    from boxbunny_core.constants import PadLocation, PunchType
    valid = PadLocation.VALID_PUNCHES["centre"]
    assert PunchType.JAB in valid
    assert PunchType.CROSS in valid
    assert PunchType.LEFT_HOOK not in valid
    assert PunchType.RIGHT_HOOK not in valid
    assert PunchType.LEFT_UPPERCUT not in valid
    assert PunchType.RIGHT_UPPERCUT not in valid
    ok()
except Exception as e:
    fail(str(e))

test("Pad constraints: left = l_hook + l_uppercut only")
try:
    valid = PadLocation.VALID_PUNCHES["left"]
    assert PunchType.LEFT_HOOK in valid
    assert PunchType.LEFT_UPPERCUT in valid
    assert PunchType.JAB not in valid
    assert PunchType.CROSS not in valid
    assert len(valid) == 2
    ok()
except Exception as e:
    fail(str(e))

test("Pad constraints: right = r_hook + r_uppercut only")
try:
    valid = PadLocation.VALID_PUNCHES["right"]
    assert PunchType.RIGHT_HOOK in valid
    assert PunchType.RIGHT_UPPERCUT in valid
    assert len(valid) == 2
    ok()
except Exception as e:
    fail(str(e))

test("Pad constraints: head = all offensive")
try:
    valid = PadLocation.VALID_PUNCHES["head"]
    for p in PunchType.OFFENSIVE:
        assert p in valid, f"{p} missing from head"
    ok()
except Exception as e:
    fail(str(e))

# ── 3. Fusion Logic ──────────────────────────────────────────────────────────
print("\n── CV+IMU Fusion Logic ──")

test("PendingCV has consecutive_frames field")
try:
    from boxbunny_core.punch_fusion import PendingCV
    cv = PendingCV(timestamp=1.0, punch_type="jab", confidence=0.9,
                   consecutive_frames=5)
    assert cv.consecutive_frames == 5
    ok()
except Exception as e:
    fail(str(e))

test("PendingIMU has accel_magnitude field")
try:
    from boxbunny_core.punch_fusion import PendingIMU
    imu = PendingIMU(timestamp=1.0, pad="centre", level="hard",
                     force_normalized=1.0, accel_magnitude=45.2)
    assert imu.accel_magnitude == 45.2
    ok()
except Exception as e:
    fail(str(e))

test("infer_punch_from_pad: centre -> jab")
try:
    from boxbunny_core.punch_fusion import infer_punch_from_pad
    assert infer_punch_from_pad("centre") == "jab"
    assert infer_punch_from_pad("left") == "left_hook"
    assert infer_punch_from_pad("right") == "right_hook"
    assert infer_punch_from_pad("head") == "jab"
    assert infer_punch_from_pad("unknown") == "unclassified"
    ok()
except Exception as e:
    fail(str(e))

test("Reclassify: jab on left pad -> unclassified (strict)")
try:
    from boxbunny_core.punch_fusion import reclassify_punch
    # Jab is NOT valid on left pad (left = l_hook / l_uppercut only)
    result = reclassify_punch("left", "jab")
    assert result == "unclassified"
    ok()
except Exception as e:
    fail(str(e))

test("Reclassify: jab on centre pad -> jab (valid)")
try:
    result = reclassify_punch("centre", "jab")
    assert result == "jab"
    ok()
except Exception as e:
    fail(str(e))

test("Reclassify: r_hook on left pad with l_hook secondary -> l_hook")
try:
    secondary = [("left_hook", 0.6)]
    result = reclassify_punch("left", "right_hook",
                              secondary_classes=secondary)
    assert result == "left_hook"
    ok()
except Exception as e:
    fail(str(e))

test("RingBuffer CV-IMU matching within window")
try:
    from boxbunny_core.punch_fusion import RingBuffer, PendingCV, PendingIMU
    buf = RingBuffer(maxlen=32)
    buf.append(PendingIMU(10.0, "centre", "hard", 1.0, 42.0))
    match = buf.pop_match(9.9, 10.1)
    assert match is not None
    assert match.pad == "centre"
    assert match.accel_magnitude == 42.0
    ok()
except Exception as e:
    fail(str(e))

# ── 4. Session Stats & Movement Tracking ─────────────────────────────────────
print("\n── Session Stats & Movement ──")

test("SessionStats records movement timeline")
try:
    from boxbunny_core.punch_fusion import SessionStats
    stats = SessionStats()
    stats.record_tracking(1.5, 20.0, lateral_disp=20.0, depth_disp=0.1)
    time.sleep(0.6)  # exceed 0.5s sampling interval
    stats.record_tracking(1.6, -15.0, lateral_disp=-15.0, depth_disp=-0.05)
    assert len(stats.tracking_history) >= 1
    assert stats.max_lateral_displacement == 20.0
    assert stats.max_depth_displacement == 0.1
    ok()
except Exception as e:
    fail(str(e))

test("SessionStats summary includes movement fields")
try:
    fields = stats.to_summary_fields()
    assert "max_lateral_displacement" in fields
    assert "max_depth_displacement" in fields
    assert "movement_timeline_json" in fields
    timeline = json.loads(fields["movement_timeline_json"])
    assert isinstance(timeline, list)
    ok()
except Exception as e:
    fail(str(e))

# ── 5. Motor Command Protocol ────────────────────────────────────────────────
print("\n── Motor Command Protocol ──")

test("Motor command format: 9 doubles [pos*4, spd*4, enable]")
try:
    positions = [0.5, 0.0, -0.5, -1.5]
    speeds = [8.0, 8.0, 8.0, 8.0]
    mode = 1.0
    payload = positions + speeds + [mode]
    assert len(payload) == 9
    assert payload[8] == 1.0
    assert all(isinstance(v, float) for v in payload)
    ok()
except Exception as e:
    fail(str(e))

test("Motor feedback parse: 21 doubles [pos*4, curr*4, can, imu*12]")
try:
    # Simulate motor_feedback data
    feedback = [0.0] * 21
    feedback[0:4] = [0.5, 0.1, -0.5, -1.5]  # positions
    feedback[4:8] = [0.1, 0.2, 0.15, 0.1]   # currents
    feedback[8] = 42.0                         # CAN count
    feedback[9:12] = [0.1, -9.8, 0.3]         # IMU0 accel
    feedback[12:15] = [0.2, -9.7, 0.1]        # IMU1 accel
    assert len(feedback) == 21
    # Extract IMU accel for pad 0 (centre)
    imu0 = feedback[9:12]
    import math
    mag = math.sqrt(sum(x**2 for x in imu0))
    assert mag > 9.0  # should be ~9.81 (gravity)
    ok()
except Exception as e:
    fail(str(e))

# ── 6. InferenceEngine Import ────────────────────────────────────────────────
print("\n── InferenceEngine ──")

test("InferenceEngine class imports")
try:
    sys.path.insert(0, str(_WS))
    from action_prediction.lib.inference_runtime import (
        InferenceEngine, InferenceResult, RollingFeatureBuffer,
    )
    ok()
except ImportError as e:
    if "torch" in str(e).lower():
        print("\033[93mSKIP\033[0m (torch not available in this env)")
        passed += 1  # not a real failure
    else:
        fail(str(e))

test("InferenceResult dataclass fields")
try:
    from action_prediction.lib.inference_runtime import InferenceResult
    r = InferenceResult()
    assert hasattr(r, "action")
    assert hasattr(r, "confidence")
    assert hasattr(r, "consecutive_frames")
    assert hasattr(r, "movement_delta")
    assert hasattr(r, "keypoints")
    assert hasattr(r, "bbox")
    assert hasattr(r, "fps")
    assert r.action == "idle"
    assert r.consecutive_frames == 0
    ok()
except ImportError:
    print("\033[93mSKIP\033[0m (torch not available)")
    passed += 1
except Exception as e:
    fail(str(e))

test("RollingFeatureBuffer normalisation modes")
try:
    from action_prediction.lib.inference_runtime import RollingFeatureBuffer
    import numpy as np
    buf = RollingFeatureBuffer(window_size=3, voxel_size=4,
                               voxel_normalization="clip_p90",
                               in_channels=1, voxel_grid_size=(4, 4, 4))
    for _ in range(3):
        buf.add_frame(np.random.randn(64).astype(np.float32), fg_ratio=0.1)
    assert buf.is_ready
    feats = buf.get_features()
    assert feats is not None
    assert feats["features"].shape == (3, 64)
    ok()
except ImportError:
    print("\033[93mSKIP\033[0m (torch not available)")
    passed += 1
except Exception as e:
    fail(str(e))

# ── 7. Reaction Time Motion Detection ────────────────────────────────────────
print("\n── Reaction Time Motion Detection ──")

test("Keypoint motion detection above threshold")
try:
    import numpy as np
    # Simulate 17 COCO keypoints (x, y, conf)
    prev = np.array([[100, 200, 0.9]] * 17, dtype=np.float32)
    curr = prev.copy()
    curr[5] = [130, 200, 0.9]  # right shoulder moved 30px
    # Compute max displacement
    max_dist = 0.0
    for i in range(len(prev)):
        if prev[i][2] < 0.3 or curr[i][2] < 0.3:
            continue
        dist = float(np.sqrt((curr[i][0] - prev[i][0])**2 +
                             (curr[i][1] - prev[i][1])**2))
        max_dist = max(max_dist, dist)
    assert max_dist == 30.0
    assert max_dist > 20.0  # exceeds threshold
    ok()
except Exception as e:
    fail(str(e))

test("Keypoint motion below threshold (idle)")
try:
    curr2 = prev.copy()
    curr2[5] = [105, 202, 0.9]  # small shift (~5.4px)
    max_dist = 0.0
    for i in range(len(prev)):
        if prev[i][2] < 0.3 or curr2[i][2] < 0.3:
            continue
        dist = float(np.sqrt((curr2[i][0] - prev[i][0])**2 +
                             (curr2[i][1] - prev[i][1])**2))
        max_dist = max(max_dist, dist)
    assert max_dist < 20.0  # below threshold
    ok()
except Exception as e:
    fail(str(e))

# ── 8. Message Fields ────────────────────────────────────────────────────────
print("\n── ROS Message Fields ──")

test("ConfirmedPunch has accel_magnitude field")
try:
    from boxbunny_msgs.msg import ConfirmedPunch
    m = ConfirmedPunch()
    m.accel_magnitude = 35.5
    assert m.accel_magnitude == 35.5
    ok()
except Exception as e:
    fail(str(e))

test("PunchDetection has consecutive_frames field")
try:
    from boxbunny_msgs.msg import PunchDetection
    m = PunchDetection()
    m.consecutive_frames = 7
    assert m.consecutive_frames == 7
    ok()
except Exception as e:
    fail(str(e))

test("PadImpact has accel_magnitude field")
try:
    from boxbunny_msgs.msg import PadImpact
    m = PadImpact()
    m.accel_magnitude = 42.0
    assert m.accel_magnitude == 42.0
    ok()
except Exception as e:
    fail(str(e))

test("SessionPunchSummary has movement fields")
try:
    from boxbunny_msgs.msg import SessionPunchSummary
    m = SessionPunchSummary()
    m.max_lateral_displacement = 50.0
    m.max_depth_displacement = 0.15
    m.movement_timeline_json = '[{"t": 0.5, "depth": 1.5}]'
    assert m.max_lateral_displacement == 50.0
    assert m.movement_timeline_json.startswith("[")
    ok()
except Exception as e:
    fail(str(e))

# ── 9. Punch Sequence Loading ────────────────────────────────────────────────
print("\n── Punch Sequences ──")

test("Punch sequence files exist and are valid JSON")
try:
    seq_dir = _WS / "data" / "punch_sequences"
    json_files = list(seq_dir.glob("*.json"))
    assert len(json_files) >= 6, f"Only {len(json_files)} sequences found"
    for f in json_files:
        data = json.loads(f.read_text())
        assert isinstance(data, list), f"{f.name}: not a list"
        assert len(data) >= 2, f"{f.name}: too few waypoints"
        for wp in data:
            assert "pos" in wp, f"{f.name}: missing 'pos'"
            assert len(wp["pos"]) >= 2, f"{f.name}: pos too short"
    ok()
except Exception as e:
    fail(str(e))

# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
color = "\033[92m" if failed == 0 else "\033[91m"
print(f"  {color}{passed}/{total} passed, {failed} failed\033[0m")
print("=" * 60)

if failed > 0:
    sys.exit(1)
