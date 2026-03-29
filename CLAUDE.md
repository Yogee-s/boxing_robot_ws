# BoxBunny — Developer Guidelines

## Critical Rules
1. **NEVER delete files** — archive to `_archive/`
2. **NEVER touch files outside** `boxing_robot_ws/`
3. **NEVER modify** `action_prediction/lib/fusion_model.py`, `pose.py`, `voxel_features.py`, `voxel_model.py`
4. All configurable values in YAML configs (`config/`) — no magic numbers in code
5. All ROS topic names configured in `config/ros_topics.yaml` — loaded by `constants.py`
6. No `print()` — use `logging` module
7. Max ~300 lines per file, type hints on all function signatures
8. Production code: docstrings, specific exception handling, structured logging

## Quick Reference
- **Config**: `config/boxbunny.yaml` (master), `config/ros_topics.yaml` (all ROS names)
- **Build**: `source /opt/ros/humble/setup.bash && colcon build --symlink-install`
- **Test**: `python3 -m pytest tests/ -v`
- **Demo users**: alex/boxing123, maria/boxing123, jake/boxing123, sarah/coaching123
