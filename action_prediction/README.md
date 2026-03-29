# Boxing Action Prediction — Standalone Deployment

Real-time boxing action recognition using Intel RealSense D435i (RGB-D) depth camera, combining 3D voxel motion features with 2D pose estimation through a transformer-based fusion model.

## What This Does

A person stands in front of a RealSense D435i camera and throws boxing punches. The system classifies each action in real-time at 30fps:

**8 classes:** `jab`, `cross`, `left_hook`, `right_hook`, `left_uppercut`, `right_uppercut`, `block`, `idle`

## How It Works

### Input Pipeline (30fps)
1. **RealSense D435i** captures synchronized RGB (960x540) + depth (848x480) at 30fps
2. **YOLO Pose** (yolo26n-pose, 320px) detects 7 upper-body keypoints from RGB every Nth frame
3. **Depth voxelization** converts the depth image into a 12x12x12 person-centric 3D occupancy grid

### Feature Extraction (per frame)

**Voxel features (3,456 dims)** — 2 channels of 12x12x12 = 1,728 voxels each:
| Channel | Content | What it captures |
|---|---|---|
| 0 | delta@2 frames (67ms at 30fps) | Fast motion — punch onset, jab snap |
| 1 | delta@8 frames (267ms at 30fps) | Sustained motion — full punch arc, hooks |

The voxel grid is person-centric (follows the person), gravity-aligned (corrected for camera tilt via IMU), and depth-weighted (closer = stronger signal). It captures the full 3D motion that 2D cameras miss — particularly the forward/backward axis critical for distinguishing jabs from hooks.

**Pose features (42 dims)** — from YOLO Pose detection on RGB:

| Dims | Content | Type | Purpose |
|---|---|---|---|
| 14 | Joint coordinates (x,y for 7 joints) | Static | Where each joint is |
| 7 | Joint confidence scores | Static | How reliable the detection is |
| 2 | Arm extension ratios | Static | How far each arm is extended |
| 1 | Shoulder rotation | Static | Body orientation |
| 2 | Elbow angles (0=bent, 1=straight) | Static | Hook vs jab discrimination |
| 14 | Joint velocities (dx,dy per joint) | Temporal | Which hand is moving, in what direction |
| 2 | Arm extension rate | Temporal | Extending vs retracting |

The 7 joints used: nose, left/right shoulder, left/right elbow, left/right wrist.

Pose is a secondary signal — useful when visible (especially during the wind-up phase before occlusion), but the model degrades gracefully to voxel-only when pose is noisy or missing (e.g., gloves blocking the body). This is achieved through confidence gating in the pose encoder.

### Model Architecture: `FusionVoxelPoseTransformerModel`

```
Input: (batch, T=12 frames, 3498 dims per frame)
                    |
        +-----------+-----------+
        |                       |
   Voxel branch            Pose branch
   (3,456 dims)            (42 dims)
        |                       |
   Reshape to              PoseEncoder MLP
   (B*T, 2, 12, 12, 12)    42 -> 64 -> 64
        |                  + confidence gating
   Conv3D Stem                  |
   3x stride-2 convs       64-dim embedding
   16 -> 32 -> 64 filters       |
        |                       |
   192-dim embedding            |
        |                       |
        +----------- concat ---+
                    |
              256 -> Linear -> 192-dim
              + LayerNorm
                    |
           Positional Encoding
                    |
           Transformer Encoder
           4 layers, 8 heads
           d=192, FFN=576
                    |
           Mean + Max pooling
           over T frames -> 384-dim
                    |
           Classifier Head
           LayerNorm -> Linear(96)
           -> GELU -> Linear(8)
                    |
              8-class logits
```

**Why this architecture:**
- **Conv3D stem** preserves spatial structure in the 12x12x12 voxel grid (vs flattening which loses 3D locality)
- **Confidence gating** in pose encoder automatically suppresses noisy/occluded frames
- **Causal transformer** each frame attends to current and past frames, matching real-time inference
- **Mean + max pooling** captures both the average motion pattern and the peak action moment

### Post-Processing

Raw model predictions are smoothed for stability:
1. **EMA smoothing** blends new predictions with recent history (configurable via `--ema-alpha`)
2. **Hysteresis** prevents flickering by requiring a margin to switch classes (`--hysteresis-margin`)
3. **Confidence gating** maps uncertain predictions to "idle" (`--min-confidence`)
4. **Optional state machine** requires sustained confidence before committing to a punch (`--use-action-state-machine`)

### GPU Optimization (`--optimize-gpu`)

On first run with this flag:
1. Model is exported to ONNX format (cached as `best_model.onnx`)
2. ONNX is compiled to a TensorRT FP16 engine (cached as `best_model.trt`)
3. Subsequent runs load the cached engine instantly

This gives ~2-4x inference speedup on NVIDIA Jetson (Orin Nano tested).

---

## Files Needed

To deploy on another machine, copy the `action_prediction/` folder. Everything needed is self-contained:

```
action_prediction/
    run.py                          <- Main entry point
    README.md                       <- This file
    live_voxelflow_inference.py     <- Inference engine + GUI
    lib/
        fusion_model.py             <- FusionVoxelPoseTransformerModel
        voxel_model.py              <- Conv3DStem, PositionalEncoding
        voxel_features.py           <- Voxel extraction from depth
        pose.py                     <- YOLO pose estimation wrapper
        __init__.py
    model/
        best_model.pth              <- Trained model checkpoint
        best_model.onnx             <- ONNX export (auto-generated)
        best_model.trt              <- TensorRT engine (auto-generated on Jetson)
        yolo26n-pose.pt             <- YOLO Pose weights
    __init__.py
```

## Setup

```bash
# Python 3.10+ required

# Install dependencies
pip install torch torchvision numpy opencv-python pyrealsense2 ultralytics

# Optional: TensorRT for Jetson GPU acceleration
# (auto-detected if available, not required)
```

## Usage

```bash
# Zero-config — all defaults work out of the box:
cd action_prediction
python run.py

# Defaults: model/best_model.pth, model/yolo26n-pose.pt,
#           --optimize-gpu (ONNX+TensorRT), --no-video

# With video feed enabled:
python run.py --show-video

# Tune responsiveness (for detecting repeated punches like jab-jab-jab):
python run.py --ema-alpha 0.65 --hysteresis-margin 0.04 --min-hold-frames 1

# Full control:
python run.py \
    --checkpoint model/best_model.pth \
    --pose-weights model/yolo26n-pose.pt \
    --device cuda:0 \
    --inference-interval 1 \
    --yolo-interval 5 \
    --downscale-width 384 \
    --temporal-smooth-window 1 \
    --min-confidence 0.7 \
    --ema-alpha 0.65 \
    --hysteresis-margin 0.04 \
    --min-hold-frames 1 \
    --processing-mode strict \
    --depth-res 848x480
```

## All Parameters

### Speed vs Accuracy
| Param | Default | Description |
|---|---|---|
| `--inference-interval` | 1 | Predict every Nth frame (1=every, 2=skip half) |
| `--yolo-interval` | 5 | YOLO pose every Nth frame (higher=faster) |
| `--downscale-width` | 384 | Feature resolution (256=fast, 384=balanced) |
| `--num-workers` | 1 | Parallel feature workers |

### Responsiveness (for repeated punches)
| Param | Default | Description |
|---|---|---|
| `--ema-alpha` | 0.65 | New prediction weight (0.35=smooth, 0.65=responsive, 1.0=raw) |
| `--hysteresis-margin` | 0.04 | Margin to switch class (0.12=sticky, 0.04=responsive, 0.0=instant) |
| `--min-hold-frames` | 1 | Hold prediction for N frames (3=sticky, 1=responsive) |
| `--temporal-smooth-window` | 1 | Smooth over N frames (1=raw, 3-5=stable) |
| `--min-confidence` | 0.7 | Below this -> idle (0.0=disabled, 0.9=strict) |

### GPU Optimization & Output
| Param | Default | Description |
|---|---|---|
| `--optimize-gpu` | **on** | Auto ONNX+TensorRT (FP16), cached after first run. Use `--no-optimize-gpu` to disable |
| `--no-video` | **on** | Video rendering disabled for max throughput. Use `--show-video` to enable |

### Camera
| Param | Default | Description |
|---|---|---|
| `--depth-res` | 848x480 | Depth stream resolution |
| `--rgb-res` | 960x540 | RGB stream resolution |
| `--processing-mode` | strict | strict=ordered frames, latest=low latency |
| `--camera-pitch` | 0 (auto) | Camera tilt in degrees (0=auto-detect via IMU) |

### State Machine (optional, disabled by default)
| Param | Default | Description |
|---|---|---|
| `--use-action-state-machine` | off | Enable causal action filtering |
| `--state-enter-consecutive` | 2 | Frames needed to commit to a punch |
| `--state-exit-consecutive` | 3 | Frames needed to return to idle |
| `--state-min-hold-steps` | 3 | Min hold before exit allowed |

## Output

### GUI Window
The system opens a tkinter window showing:
- **Prediction label** — current detected action (e.g., "JAB", "LEFT_HOOK", "IDLE")
- **Confidence** — model confidence percentage for the current prediction
- **FPS counters** — GUI FPS, camera FPS, and prediction Hz
- **Status indicator** — model loaded, camera ready, running
- **Probability bars** — per-class probability distribution (8 bars)
- **Camera feed** — live RGB with pose skeleton overlay (when `--show-video` is used)

### Prediction Classes
| Class | Description |
|---|---|
| `jab` | Lead hand straight punch |
| `cross` | Rear hand straight punch |
| `left_hook` | Left hand lateral punch (bent elbow) |
| `right_hook` | Right hand lateral punch (bent elbow) |
| `left_uppercut` | Left hand upward punch |
| `right_uppercut` | Right hand upward punch |
| `block` | Defensive guard position |
| `idle` | No action / guard stance |

### Input Requirements
- **Camera:** Intel RealSense D435i (RGB-D depth camera)
- **Position:** Person standing 1-2m from camera, upper body visible
- **Lighting:** Indoor lighting sufficient for RGB pose detection
- **Connection:** USB 3.0 to Jetson/PC

## Hardware Tested

- **NVIDIA Jetson Orin Nano** (16GB, JetPack 6.2, CUDA 12.4)
- Intel RealSense D435i
- Inference at ~15-30 prediction FPS depending on configuration
