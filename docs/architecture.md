# BoxBunny Architecture

## Nodes (ROS 2 Humble)

- `realsense2_camera` (external) – publishes camera streams
- `realsense_glove_tracker` – HSV glove detection + punch trigger (vision)
- `mpu6050_node` – IMU raw data
- `imu_punch_classifier` – classifies punch type from IMU
- `punch_fusion_node` – fuses vision punches + IMU punches into final events
- `reaction_drill_manager` – countdown, baseline capture, cue, reaction timing
- `punch_stats_node` – rolling analytics summary (counts, avg velocity)
- `trash_talk_node` – LLM coach output + service
- `boxbunny_gui` – GUI for drill, punch view, calibration, LLM

## Topics

- `/camera/color/image_raw` – RGB image
- `/camera/aligned_depth_to_color/image_raw` – depth image
- `/glove_detections` – glove boxes + distance + velocity
- `/punch_events_raw` – vision punches (from tracker)
- `/imu/data` – IMU raw data
- `/imu/punch` – IMU punch type
- `/punch_events` – fused punches
- `/drill_state` – drill state machine
- `/drill_countdown` – countdown integer seconds
- `/drill_events` – drill event markers
- `/drill_summary` – JSON summary (reaction times)
- `/punch_stats` – JSON stats (rolling window)
- `/trash_talk` – LLM outputs

## Data Flow

```
RealSense -> realsense_glove_tracker -> punch_events_raw
MPU6050  -> imu_punch_classifier     -> imu/punch
punch_events_raw + imu/punch -> punch_fusion_node -> punch_events
punch_events -> reaction_drill_manager + punch_stats_node + GUI + LLM
punch_stats  -> LLM (context)
```

## Configuration Files

- `src/boxbunny_vision/config/glove_tracker.yaml`
- `src/boxbunny_imu/config/imu.yaml`
- `src/boxbunny_fusion/config/fusion.yaml`
- `src/boxbunny_drills/config/drill.yaml`
- `src/boxbunny_analytics/config/analytics.yaml`
- `src/boxbunny_llm/config/llm.yaml`
- `src/boxbunny_gui/config/gui.yaml`
