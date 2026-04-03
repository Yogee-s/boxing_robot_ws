# BoxBunny Computer Vision Pipeline

## 1. Model Architecture

BoxBunny uses a custom **FusionVoxelPoseTransformerModel** for real-time boxing action recognition. The model fuses two complementary modalities -- 3D voxel motion features from depth data and 2D pose features from RGB -- through a transformer encoder to classify 8 boxing actions at 30fps.

### Action Classes

| Index | Class | Description |
|---|---|---|
| 0 | `jab` | Lead-hand straight punch (fast, centre pad) |
| 1 | `cross` | Rear-hand straight punch (powerful, centre pad) |
| 2 | `left_hook` | Left lateral punch (left pad) |
| 3 | `right_hook` | Right lateral punch (right pad) |
| 4 | `left_uppercut` | Left rising punch (left pad) |
| 5 | `right_uppercut` | Right rising punch (right pad) |
| 6 | `block` | Defensive guard position |
| 7 | `idle` | No action / standing |

**Best validation accuracy:** 96.6% (cross-person generalization -- trained on 3 people, validated on 1 unseen person, model version v5).

### Input Pipeline (Per Frame at 30fps)

```
Intel RealSense D435i
├── RGB stream (960x540 @ 30fps)
│   └── YOLO Pose (yolo26n-pose, TensorRT FP16, ~16ms)
│       └── 7 upper-body keypoints (nose, shoulders, elbows, wrists)
│           └── Pose Features (42 dimensions)
│
└── Depth stream (848x480 @ 30fps, aligned to colour)
    └── Voxel Extraction
        └── 12x12x12 person-centric 3D occupancy grid
            └── 2-channel temporal differencing
                └── Voxel Features (3,456 dimensions)
```

### Voxel Feature Extraction (3,456 dims)

The depth image is converted into a 12x12x12 person-centric 3D occupancy grid. Two temporal difference channels capture motion at different timescales:

| Channel | Content | Timescale | What It Captures |
|---|---|---|---|
| 0 | delta@2 frames | 67ms at 30fps | Fast motion -- punch onset, jab snap |
| 1 | delta@8 frames | 267ms at 30fps | Sustained motion -- full punch arc, hooks, uppercuts |

Each channel is a 12x12x12 = 1,728-voxel grid. Concatenated: 2 x 1,728 = 3,456 dimensions.

The voxel grid has three important properties:
- **Person-centric**: the grid follows the detected person's bounding box, so position in the frame does not affect features
- **Gravity-aligned**: the grid is corrected for camera tilt (configurable via `--camera-pitch`, default 5 degrees)
- **Depth-weighted**: closer voxels produce a stronger signal, reducing noise from background objects

### Pose Feature Extraction (42 dims)

From the YOLO Pose detection on RGB:

| Dimensions | Content | Purpose |
|---|---|---|
| 14 | Joint coordinates (x, y for 7 joints) | Absolute joint positions |
| 7 | Joint confidence scores | Detection reliability weighting |
| 2 | Arm extension ratios | How far each arm is extended from shoulder |
| 1 | Shoulder rotation | Body orientation (angled vs square) |
| 2 | Elbow angles (0=bent, 1=straight) | Hook vs jab discrimination |
| 14 | Joint velocities (dx, dy per joint) | Which hand is moving, direction, speed |
| 2 | Arm extension rate | Extending (punching) vs retracting (returning) |

### Model Architecture

```
Input: (batch, T=12 frames, 3498 dims per frame)
            [3456 voxel + 42 pose]
                    |
        +-----------+-----------+
        |                       |
   Voxel Branch            Pose Branch
   (3,456 dims)            (42 dims)
        |                       |
   Conv3D Stem             PoseEncoder MLP
   3x stride-2 convs       42 → 64 → 64
   16 → 32 → 64 channels   + confidence gating
        |                       |
   Flatten → 192-dim       64-dim embedding
        |                       |
        +----------- concat ----+
                    |
              256 → Linear → 192-dim
              + LayerNorm
                    |
           Positional Encoding (T=12)
                    |
           Transformer Encoder
           4 layers, 8 attention heads
           d_model=192, FFN=576
                    |
           Mean Pooling + Max Pooling → 384-dim
                    |
           Classifier Head
           384 → 192 → 8-class logits
```

**Training details:** Focal loss with gamma=2.0, minimal augmentation, cross-person validation protocol.

### GPU Optimisation

Both models are compiled to TensorRT FP16 engines for Jetson deployment:

| Model | Format | Inference Time | Notes |
|---|---|---|---|
| Action recognition | `best_model.trt` (TensorRT FP16) | ~8ms | Auto-generated from .pth via ONNX |
| YOLO Pose | `yolo26n-pose.engine` (TensorRT FP16, dynamic input) | ~16ms | Dynamic batch/resolution |
| **Total per frame** | | **~24ms** | **~42fps theoretical max** on Orin NX 16GB |

TensorRT engines are auto-generated on first run and cached. Fallback to raw PyTorch is available with `--no-optimize-gpu`.

---

## 2. Inference Pipeline

The inference pipeline is implemented in two layers:

1. **`action_prediction/lib/inference_runtime.py`** -- Headless inference engine. No GUI code. Handles model loading, feature extraction, inference, and post-processing.
2. **`src/boxbunny_core/boxbunny_core/cv_node.py`** -- ROS 2 wrapper that subscribes to camera topics, calls the inference engine, and publishes results.

### InferenceEngine (inference_runtime.py)

The `InferenceEngine` class encapsulates the full pipeline:

```python
@dataclass
class InferenceResult:
    action: str              # Post-processed prediction ("jab", "cross", ...)
    confidence: float        # 0.0-1.0 after EMA + hysteresis
    raw_action: str          # Raw model output before filtering
    consecutive_frames: int  # Consecutive frames with this prediction
    fps: float               # Current inference throughput
    keypoints: np.ndarray    # YOLO pose keypoints (7 joints)
    bbox: dict               # Bounding box {cx, cy, top_y, width, height, depth}
    movement_delta: float    # Pose change magnitude from previous frame

class InferenceEngine:
    def initialize(self) -> None:
        """Load models (YOLO Pose + action recognition), warm up GPU."""
        
    def process_frame(self, rgb: np.ndarray, depth: np.ndarray) -> InferenceResult:
        """Full pipeline: pose → voxels → inference → post-process."""
```

### Post-Processing Chain

Raw model logits go through a multi-stage filtering pipeline before being published. Each stage addresses a specific type of noise:

```
Raw model logits (8 classes)
        │
        ▼
[1] EMA Smoothing (alpha=0.35)
    - Exponential moving average across frames
    - Prevents single-frame spikes from causing false positives
    - alpha=0.35 means 65% weight on history, 35% on new frame
    - Lower alpha = smoother but more latent
        │
        ▼
[2] Hysteresis Filter (margin=0.12)
    - A new class must exceed the current class by margin
    - Prevents rapid oscillation between similar classes
    - Example: if "jab" is at 0.85, "cross" must reach 0.97 to take over
        │
        ▼
[3] Minimum Hold Frames (min_hold=3)
    - A class must be top-ranked for N consecutive frames before switching
    - Prevents brief flickers from propagating
        │
        ▼
[4] State Machine Filter
    - States: idle, jab, cross, left_hook, right_hook, ...
    - state_enter_consecutive: 2 frames to ENTER a new state
    - state_exit_consecutive: 2 frames of absence to EXIT
    - state_min_hold_steps: 2 minimum frames once entered
    - state_sustain_confidence: 0.78 threshold to sustain a state
    - state_peak_drop_threshold: 0.02 max drop from peak before exit
        │
        ▼
[5] Block Consecutive Filter (block_consecutive_needed=4)
    - Blocks require 4 consecutive high-confidence frames to confirm
    - Prevents brief guard positions from being classified as blocks
        │
        ▼
[6] Confidence Gate (min_confidence=0.4)
    - Below threshold → output "idle"
    - Prevents low-confidence predictions from reaching the fusion layer
        │
        ▼
Published as PunchDetection message
```

### cv_node.py: ROS Integration

The `CvNode` class wraps the inference engine as a ROS 2 node:

```python
class CvNode(Node):
    def __init__(self):
        # Subscribe to RealSense streams
        self.create_subscription(Image, "/camera/color/image_raw", self._on_color, 5)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self._on_depth, 5)
        
        # Publish results
        self._pub_detection = self.create_publisher(PunchDetection, "/boxbunny/cv/detection", 10)
        self._pub_pose = self.create_publisher(PoseEstimate, "/boxbunny/cv/pose", 10)
        self._pub_tracking = self.create_publisher(UserTracking, "/boxbunny/cv/user_tracking", 10)
        self._pub_person_direction = self.create_publisher(String, "/boxbunny/cv/person_direction", 10)
        
        # Run inference at 30Hz
        self.create_timer(1.0 / 30.0, self._inference_tick)
```

Key design decisions:
- **Lazy initialization**: The inference engine is loaded on the first frame, not at node startup. This allows the node to start quickly and fail gracefully if the model is missing.
- **Always-on inference**: Inference runs regardless of session state. The cv_node does not gate on SessionState -- it always publishes detections. Downstream consumers (punch_processor, session_manager) decide whether to use them.
- **Baseline management**: When a session starts (idle -> countdown), lateral and depth baselines are reset to the current position. Displacements are computed relative to this baseline for slip/dodge detection.

---

## 3. CV + IMU Fusion

The fusion system is the core innovation of BoxBunny. It combines two independent sensing modalities to produce high-confidence punch classifications that neither modality can achieve alone.

### Why Fusion Is Necessary

| Modality | Strengths | Weaknesses |
|---|---|---|
| **CV (Camera)** | Classifies punch type accurately (96.6%). Detects blocks and idle. Works without contact. | Cannot detect impact force. Latency from inference pipeline. Can misclassify similar punches (jab vs cross). No physical confirmation. |
| **IMU (Pads)** | Confirms physical contact. Measures impact force. No latency (hardware interrupt). Knows which pad was struck. | Cannot classify punch type -- only knows which pad was hit. Cannot detect blocks, slips, or dodges. |

**The fusion principle:** The CV model tells us WHAT punch was thrown. The IMU tells us WHERE it landed and HOW HARD. By combining them, we get confirmed punches with both classification and physical verification.

### Fusion Architecture

The fusion is implemented across two files:
- `punch_processor.py` -- the ROS node that orchestrates fusion
- `punch_fusion.py` -- helper functions, data structures, and the defense classifier

### How Fusion Works

```
                 CV Path                          IMU Path
                   │                                │
    PunchDetection │                    PunchEvent  │
    (30fps from    │                    (from       │
     cv_node)      │                     imu_node)  │
                   │                                │
                   ▼                                │
          ┌────────────────┐                        │
          │ CV Buffer       │                        │
          │ (800ms window)  │                        │
          │                │                        │
          │ Stores recent  │                        │
          │ non-idle,      │                        │
          │ non-block      │                        │
          │ predictions    │                        │
          │ with timestamp │                        │
          │ and confidence │                        │
          └────────┬───────┘                        │
                   │                                │
                   │   ┌───────────────────────┐    │
                   │   │ IMU Strike Trigger     │◄───┘
                   └──►│                       │
                       │ 1. Get pad name        │
                       │ 2. Lookup valid punches│
                       │    for this pad        │
                       │ 3. Search CV buffer    │
                       │    for valid matches   │
                       │ 4. Count frames per    │
                       │    valid punch type    │
                       │ 5. Pick dominant type  │
                       │    (most frames)       │
                       │ 6. Emit ConfirmedPunch │
                       │ 7. Clear CV buffer     │
                       └───────────────────────┘
```

### The IMU-Triggered Fusion Algorithm

When an IMU pad strike is detected, the punch_processor executes this algorithm:

```python
def _on_imu(self, msg: PunchEvent) -> None:
    """IMU strike detected -- use the pad to filter CV predictions."""
    pad = msg.pad
    valid = PadLocation.VALID_PUNCHES.get(pad)
    
    # Search the CV buffer for predictions valid on this pad
    counts: dict[str, int] = {}
    best_conf: dict[str, float] = {}
    for _ts, action, conf in self._cv_buffer:
        if valid is not None and action not in valid:
            continue  # False positive -- wrong punch for this pad
        counts[action] = counts.get(action, 0) + 1
        if conf > best_conf.get(action, 0.0):
            best_conf[action] = conf

    if not counts:
        return  # No valid CV predictions in buffer -- skip

    # Pick the punch type with the most frames (dominant prediction)
    cv_action = max(counts, key=lambda a: counts[a])
    cv_conf = best_conf[cv_action]
    
    # Clear buffer up to now (prevents double-matching)
    self._cv_buffer = [e for e in self._cv_buffer if e[0] > ts]
    
    # Emit confirmed punch
    self._emit(
        punch_type=cv_action, pad=pad, level=msg.level,
        force=msg.force_normalized, confidence=cv_conf,
        imu_confirmed=True, cv_confirmed=True,
    )
```

This design has several important properties:

1. **IMU is the trigger**: A punch is only confirmed when a pad is physically struck. CV predictions alone never produce a confirmed punch. This eliminates shadow-boxing false positives.
2. **Frame counting over peak confidence**: Instead of picking the CV prediction with the highest single-frame confidence, the algorithm counts how many frames each valid type appeared in the buffer. This filters out 1-frame classification errors -- if the model briefly classifies a jab as a cross for one frame, the frame count is 1 vs potentially 10+ for the correct classification.
3. **Buffer clearing prevents double-matching**: After a match, CV predictions up to the current timestamp are cleared, preventing the same CV prediction from matching a second rapid strike.

### Pad Constraint Table

The pad constraints are the key mechanism for filtering CV misclassifications. If the CV model says "left_hook" but the user struck the centre pad, the prediction is rejected because hooks cannot physically land on the centre pad.

```
┌───────────┬───────────────────────────────────────────┐
│  Pad      │  Valid Punch Types                        │
├───────────┼───────────────────────────────────────────┤
│  Centre   │  jab, cross                               │
│           │  (straight punches only)                  │
├───────────┼───────────────────────────────────────────┤
│  Left     │  left_hook, left_uppercut                 │
│           │  (left-side punches only)                 │
├───────────┼───────────────────────────────────────────┤
│  Right    │  right_hook, right_uppercut               │
│           │  (right-side punches only)                │
├───────────┼───────────────────────────────────────────┤
│  Head     │  jab, cross, left_hook, right_hook,       │
│           │  left_uppercut, right_uppercut             │
│           │  (all offensive punches valid)             │
└───────────┴───────────────────────────────────────────┘
```

These constraints are defined in `constants.py`:

```python
class PadLocation:
    VALID_PUNCHES = {
        LEFT: [PunchType.LEFT_HOOK, PunchType.LEFT_UPPERCUT],
        CENTRE: [PunchType.JAB, PunchType.CROSS],
        RIGHT: [PunchType.RIGHT_HOOK, PunchType.RIGHT_UPPERCUT],
        HEAD: PunchType.OFFENSIVE,  # all offensive punches
    }
```

### Reclassification Logic

When the primary CV prediction violates the pad constraint, the system attempts reclassification using secondary predictions from `punch_fusion.py`:

```python
def reclassify_punch(pad, cv_type, secondary_classes=None, min_conf=0.25):
    """Return a valid punch type for pad, reclassifying if needed."""
    valid = PadLocation.VALID_PUNCHES.get(pad)
    if cv_type in valid:
        return cv_type  # Primary prediction is valid
    
    # Try secondary classes (sorted by descending confidence)
    if secondary_classes:
        for cls_name, conf in secondary_classes:
            if cls_name in valid and conf >= min_conf:
                return cls_name
    
    return "unclassified"
```

### Defense Detection

When the robot throws a punch (via RobotCommand), the punch_processor opens a 500ms defense window. During this window, three data sources are collected:

1. **Arm IMU events**: Did the robot arm contact the user (struck) or miss?
2. **CV block detections**: Is the CV model detecting a block pose?
3. **User tracking snapshots**: Has the user moved laterally or in depth?

At window close, `classify_defense()` determines the outcome:

```python
def classify_defense(arm_events, cv_blocks, tracking_snapshots, ...):
    """Determine defense outcome."""
    # 1. Check arm contact
    struck = any(e.get("contact", False) for e in arm_events)
    if struck:
        return True, "hit"
    
    # 2. Arm missed -- check for block via CV
    has_block = any(b["confidence"] >= block_cv_min for b in cv_blocks)
    if has_block:
        return False, "block"
    
    # 3. Check tracking displacement for slip/dodge
    max_lateral = max(abs(t["lateral_displacement"]) for t in tracking_snapshots)
    max_depth = max(abs(t["depth_displacement"]) for t in tracking_snapshots)
    
    if max_lateral >= slip_lateral_px or max_depth >= slip_depth_m:
        return False, "slip"    # Large displacement
    if max_lateral >= dodge_lateral_px or max_depth >= dodge_depth_m:
        return False, "dodge"   # Moderate displacement
    
    return False, "unknown"
```

Defense detection thresholds (from `config/boxbunny.yaml`):

| Defense Type | Lateral Threshold | Depth Threshold |
|---|---|---|
| **Slip** | 40 pixels | 0.15 metres |
| **Dodge** | 20 pixels | 0.08 metres |
| **Block** | N/A (CV confidence >= 0.3) | N/A |
| **Hit** | N/A (arm contact = True) | N/A |

### Session Statistics (SessionStats)

The `SessionStats` dataclass in `punch_fusion.py` maintains running tallies during a session:

```python
@dataclass
class SessionStats:
    punch_counts: Dict[str, int]    # Per punch type
    force_sums: Dict[str, float]    # Per punch type (for averaging)
    pad_counts: Dict[str, int]      # Per pad
    total_punches: int
    confidence_sum: float
    imu_confirmed_count: int
    peak_force_level: str
    robot_punches_thrown: int
    robot_punches_landed: int
    defense_types: Dict[str, int]   # block/slip/dodge/hit/unknown
    depth_values: List[float]       # For average depth, range
    lateral_values: List[float]     # For movement analysis
    tracking_history: List[dict]    # Time-series at ~2Hz sampling
    rounds_completed: int
```

At session end, `to_summary_fields()` computes aggregate statistics and publishes a `SessionPunchSummary` message containing distributions, rates, and movement data.

---

## 4. Person Tracking

### UserTracking Message

The `cv_node` publishes a `UserTracking` message for every frame where inference runs. This message carries the YOLO bounding box information enriched with depth data:

| Field | Type | Description |
|---|---|---|
| `bbox_centre_x` | float32 | Bounding box centre X in pixels |
| `bbox_centre_y` | float32 | Bounding box centre Y in pixels |
| `bbox_top_y` | float32 | Top of bounding box -- used for height auto-adjustment |
| `bbox_width` | float32 | Bounding box width in pixels |
| `bbox_height` | float32 | Bounding box height in pixels |
| `depth` | float32 | Distance from camera to user in metres (D435i depth) |
| `lateral_displacement` | float32 | bbox_centre_x shift from baseline (pixels) |
| `depth_displacement` | float32 | Depth change from baseline (metres) |
| `user_detected` | bool | False if no person in frame |

**Baseline management**: When a session transitions from idle to countdown, the baseline (`_baseline_bbox_x`, `_baseline_depth`) is reset to the current position. All displacements are computed relative to this baseline, so slip/dodge detection measures movement from the user's starting stance, not absolute position.

### Person Direction Derivation

The `cv_node` computes a discrete left/right/centre direction from the bounding box centre X position. This drives the yaw motor to keep the robot facing the user.

```python
def _publish_person_direction(self, cx: float) -> None:
    """Publish left/right/centre based on bbox position with hysteresis."""
    w = self._frame_width  # 960 pixels
    
    # Centre zone = middle 30% of frame (336px to 624px)
    left_boundary = w * 0.35   # 336px
    right_boundary = w * 0.65  # 624px
    hysteresis = 20.0          # 20px past boundary before switching
```

The hysteresis prevents rapid oscillation when the user is near a boundary:

```
    0px              336px        624px              960px
    |     LEFT        |  CENTRE   |      RIGHT       |
    |                 |           |                   |
    |        ←20px hysteresis→    ←20px hysteresis→   |
    |           |     |     |     |     |              |
    |           316   336   |     624   644            |
    |           (enter L)   |     |   (enter R)        |
    |                 356   |   604                    |
    |              (exit L) |  (exit R)                |
```

Example: if currently "centre", the user must move past 316px (336 - 20) to switch to "left". Once in "left", they must move past 356px (336 + 20) to switch back to "centre".

The direction is forwarded by `robot_node` to `/robot/yaw_cmd` as "LEFT", "RIGHT", or "CENTRE" for the Teensy yaw motor.

### Height Auto-Adjustment

During the countdown phase, the `session_manager` monitors `bbox_top_y` and publishes `HeightCommand` messages to adjust the robot's height so the user's head is at approximately 15% from the top of the 540p frame:

```python
target_height_px = 0.15 * 540  # ~81 pixels from top
```

The `robot_node` converts the pixel error into a PWM direction and magnitude:

```python
error = msg.current_height_px - msg.target_height_px
pwm = min(255, int(abs(error) * 2))
direction = "UP" if error > 0 else "DOWN"
height_msg.data = f"{direction}:{pwm}"  # e.g., "UP:200"
```

Height adjustment configuration (from `config/boxbunny.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `ideal_top_fraction` | 0.15 | Target head position as fraction of frame height |
| `deadband_px` | 15.0 | No adjustment if error is within this range |
| `max_iterations` | 3 | Maximum adjustment cycles per countdown |
| `settle_delay_ms` | 500 | Wait time between adjustment iterations |
| `min_depth_m` / `max_depth_m` | 0.5 / 3.0 | Valid depth range for adjustment |

---

## 5. run_with_ros.py Bridge

The `notebooks/scripts/run_with_ros.py` script bridges the standalone action prediction GUI (`live_voxelflow_inference.py`) to the ROS 2 ecosystem. This is used for development and testing -- it allows running the full CV inference pipeline with its own Tkinter GUI while simultaneously publishing predictions to ROS for the fusion pipeline.

### Architecture

```
┌────────────────────────────────────────────────┐
│ run_with_ros.py                                │
│                                                │
│  ┌─────────────────────────────────────────┐   │
│  │ LiveVoxelGUI (Tkinter)                  │   │
│  │ - RealSense camera management            │   │
│  │ - Full inference pipeline                │   │
│  │ - Visual feedback (action bars, stats)   │   │
│  └─────────────┬───────────────────────────┘   │
│                │ app.smooth_probs (30fps)       │
│                ▼                                │
│  ┌─────────────────────────────────────────┐   │
│  │ _pub_loop() — 30Hz Tkinter timer        │   │
│  │                                          │   │
│  │ 1. Read raw (ungated) probabilities      │   │
│  │    from app.smooth_probs + app.labels    │   │
│  │ 2. argmax → action, confidence            │   │
│  │ 3. If conf < 0.2 → "idle"               │   │
│  │ 4. Publish PunchDetection to ROS          │   │
│  │ 5. Read YOLO bbox for person direction    │   │
│  │ 6. Publish person_direction to ROS        │   │
│  └─────────────┬───────────────────────────┘   │
│                │                                │
│  ┌─────────────▼───────────────────────────┐   │
│  │ _CVPub (ROS Node, background thread)     │   │
│  │ - Publisher: /boxbunny/cv/detection       │   │
│  │ - Publisher: /boxbunny/cv/person_direction│   │
│  │ - Runs rclpy.spin() in daemon thread      │   │
│  └─────────────────────────────────────────┘   │
└────────────────────────────────────────────────┘
```

### Key Design Decision: Raw Probabilities

The bridge publishes **raw (ungated) probabilities** rather than the GUI's gated predictions:

```python
# Use raw (ungated) probabilities so the CV buffer has real predictions.
# The inference GUI gates predictions below min-confidence to "idle",
# but we need the raw prediction for IMU pad fusion to filter properly.
probs = getattr(app, 'smooth_probs', None)
labels = getattr(app, 'labels', None)
idx = int(np.argmax(probs))
action = labels[idx]
conf = float(probs[idx])
if conf < 0.2:
    action = "idle"
    conf = 0.0
_ros_node.send(action, conf)
```

This is critical because the punch_processor's CV buffer needs to see the actual prediction (even at moderate confidence) so that when an IMU strike arrives, the buffer contains the correct action for pad-constraint filtering. If the bridge only published "idle" for sub-threshold predictions, the fusion system would miss many valid punches.

### Usage

```bash
cd action_prediction
python3 ../notebooks/scripts/run_with_ros.py \
    --checkpoint model/best_model.pth \
    --pose-weights model/yolo26n-pose.engine \
    --device cuda:0 \
    --no-video
```

This launches the Tkinter inference GUI and simultaneously publishes to ROS. The punch_processor, imu_node, and Teensy simulator (or real hardware) can then be run in parallel for a full fusion test.

### Fusion Monitor (fusion_monitor.py)

The `notebooks/scripts/fusion_monitor.py` is a companion Tkinter GUI that visualises the fusion pipeline output. It subscribes to:

- `/robot/strike_detected` -- from the V4 Arm Control GUI (gravity-calibrated IMU strikes)
- `/boxbunny/punch/confirmed` -- from punch_processor (fused output)
- `/boxbunny/cv/detection` -- from cv_node (for live CV state display)

The monitor shows:
- Each IMU strike event with pad name, peak acceleration, and force level
- Each confirmed punch with the CV classification, confidence, pad, and whether it was IMU/CV confirmed
- The current CV prediction state (for debugging when no strikes are happening)

This tool is used during integration testing to verify that the fusion pipeline is correctly matching CV predictions to IMU events and applying pad constraints.
