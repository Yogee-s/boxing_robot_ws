# Customization Guide

## Toggle Major Subsystems

Launch args:

```bash
ros2 launch boxbunny_bringup boxbunny_system.launch.py enable_imu:=true enable_llm:=true enable_gui:=true
```

- `enable_imu`: start IMU nodes or not
- `enable_llm`: start LLM node or not
- `enable_gui`: start GUI or not

## Model Paths

Update these files to swap models:

- `src/boxbunny_llm/config/llm.yaml`
  - `model_path`: GGUF path for llama.cpp
  - `persona_examples_path`: short few‑shot YAML
  - `dataset_path`: larger example dataset for prompt guidance
- `src/boxbunny_vision/config/glove_tracker.yaml`
  - `pose_model_path`: pose model path (YOLO26n‑pose)

## LLM Modes

Modes supported:
- `coach`
- `encourage`
- `trash`
- `analysis`

Use the GUI LLM tab or call the service:

```bash
ros2 service call /llm/generate boxbunny_msgs/srv/GenerateLLM "{prompt: 'Analyze my punches', mode: 'analysis', context: 'stats'}"
```

## IMU Calibration

```bash
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch "{punch_type: 'jab_or_cross', duration_s: 2.5}"
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch "{punch_type: 'hook', duration_s: 2.5}"
ros2 service call /calibrate_imu_punch boxbunny_msgs/srv/CalibrateImuPunch "{punch_type: 'uppercut', duration_s: 2.5}"
```

## Performance Tuning

- Reduce `resize_scale` in `glove_tracker.yaml`
- Increase `process_every_n` for lower CPU
- Reduce RealSense FPS in `realsense.yaml`
