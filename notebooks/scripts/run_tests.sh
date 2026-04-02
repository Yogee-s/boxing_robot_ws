#!/usr/bin/env bash
# Run all pytest tests (excluding broken state_machine test).
set +e
cd /home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 -m pytest tests/ -v --tb=short -p no:launch-testing 2>&1
